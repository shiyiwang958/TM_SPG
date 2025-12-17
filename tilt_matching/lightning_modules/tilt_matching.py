import copy
import logging
import math
import os
from datetime import datetime
import itertools
import wandb

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model


class TiltMatchingModule(pl.LightningModule):
    def __init__(self, base_model, tokenizer, training_prompts_dataset, reward_funcs, **cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["base_model", "tokenizer", "training_prompts_dataset", "reward_funcs"], logger=False)
        self.tokenizer = tokenizer

        peft_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type=self.hparams.peft_task_type,
            lora_dropout=self.hparams.lora_dropout,
        )
        peft_wrapped = get_peft_model(base_model, peft_config)

        # Frozen teacher with LoRA
        self.base_model = copy.deepcopy(peft_wrapped)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

        # Trainable student with LoRA
        self.model = copy.deepcopy(peft_wrapped)

        # Load training prompts dataset and reward functions
        self.curr_prompt_counter = 0
        self.training_prompts_dataset = training_prompts_dataset
        self.training_prompts_dataset_len = len(training_prompts_dataset)
        self.reward_funcs = reward_funcs
        self.reward_weights = None  # Fine for now; will be all 1's as in d1/spg

        self.a = 0.0
        self.h = self.hparams.tm.h
        self.steps_per_h = self.hparams.tm.steps_per_h
        self.a_end = self.hparams.tm.a_end
        self.mask_id = 126336 # default from LLaDA
        self.checkpoint_freq = self.hparams.checkpoint_freq
        self.cv = self.hparams.tm.control_variate
        self.buffer = None
        self.buffer_rewards = None
        self.num_buffer_prompts = self.hparams.tm.num_buffer_prompts
        self.comps_per_prompt = self.hparams.tm.num_completions_per_prompt
        self.buffer_update_counter = 0
        self._step_counter = 0  # incremented at every call of on_train_batch_end
        self.dict_for_logs = {}

        # TODO: Dropouts?

        # LR Scheduling
        self.lr = self.hparams.learning_rate
        self.lr_scheduler_type = self.hparams.lr_scheduler_type
        self.lr_decay_ratio = self.hparams.lr_decay_ratio
        self.lr_warmup_ratio = getattr(self.hparams, "lr_warmup_ratio", 0)
        self.lr_min = getattr(self.hparams, "lr_min", 0.0)
        self._tm_sched_state = None

    def on_train_start(self):
        super().on_train_start()
        # Set up optimizer and LR
        self.tm_opt = self.optimizers()
        for g in self.tm_opt.param_groups:
            g["lr"] = self.lr
        self._init_tm_scheduler()

        self._update_buffer(self.base_model, self.num_buffer_prompts, self.comps_per_prompt)
        print(f"[DEBUG] Buffer initialized with shape {self.buffer.shape}")
    
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = AdamW(
            params,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay,
        )
        return opt
        
    def training_step(self, batch, batch_idx):
        """
        Perform one training step of Tilt Matching.
        The actual training logic is handled inside this method.

        Args:
            batch: Dummy batch (not used).
            batch_idx: Index of the batch (not used).
        """
        opt = self.tm_opt
        opt.zero_grad()
        loss = self._tm_step()
        self.manual_backward(loss)

        clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.hparams.max_grad_norm)
        opt.step()
        self._step_tm_scheduler()

        # Log current learning rate
        self.dict_for_logs["train/lr"] = opt.param_groups[0]["lr"]

        # At each h phase boundary, update a and base_model; save ckpt if necessary
        if (self._step_counter + 1) % self.steps_per_h == 0:
            self.a += self.h
            if self.a + self.h > self.a_end:
                self.h = self.a_end - self.a
            with torch.no_grad():
                self.base_model.load_state_dict(self.model.state_dict(), strict=True)
                for p in self.base_model.parameters():
                    p.requires_grad_(False)
            self.base_model.eval()
            print(f"Degree of tilt a = {self.a:.4f} at step {self.global_step}")

            if self.a >= self.a_end:
                print(f"Reached final a = {self.a_end:.2f}. Training Stopped", flush=True)
                self.trainer.should_stop = True

            # Reset LR for the new h phase
            for g in opt.param_groups:
                g["lr"] = self.lr
            self._init_tm_scheduler()

            # Reset buffer
            self._update_buffer(self.base_model, self.num_buffer_prompts, self.comps_per_prompt)
            print(f"[DEBUG] Buffer built at step {self._step_counter} with shape {self.buffer.shape}")
        # Partially refresh buffer
        elif (self._step_counter + 1) % self.hparams.tm.buffer_refresh_steps == 0:
            print(f"[DEBUG] Refreshing {self.hparams.tm.num_buffer_refresh} prompts at step {self._step_counter}")
            self._update_buffer(self.base_model, self.hparams.tm.num_buffer_refresh, self.comps_per_prompt)
        
        self.log("ckpt_a", self.a, on_step=True, on_epoch=False, sync_dist=True)

    def _tm_step(self):
        num_buffer_prompts, comps_per_prompt, L = self.buffer.shape
        num_batch_prompts = self.hparams.tm.num_batch_prompts
        B = num_batch_prompts * comps_per_prompt
        gen_length = self.hparams.max_completion_length

        # Draw a batch from the buffer
        prompts_idx = torch.randperm(num_buffer_prompts, device=self.device)[:num_batch_prompts]
        x1s = self.buffer[prompts_idx].reshape(B, L)           # [B, L]
        rwds = self.buffer_rewards[prompts_idx].reshape(B, -1) # [B, num_reward_funcs]

        # Aggregate rewards from multiple functions
        if self.reward_weights is None:
            weights = torch.ones(rwds.shape[1], device=self.device, dtype=rwds.dtype)
        else:
            weights = self.reward_weights.to(device=self.device, dtype=rwds.dtype)
        rwd = torch.nansum(rwds * weights.unsqueeze(0), dim=1) # [B,]
        
        # Create x_t's by masking the x_1's
        num_to_mask = torch.randint(low=1, high=gen_length+1, size=(x1s.shape[0],), device=self.device)
        xts, mask_indices = self._build_interpolant(x1s, num_to_mask, self.hparams.block_length)

        # Get model predictions and compute loss
        temp = self.hparams.sampling_temperature
        with torch.no_grad():
            old_logits = self.base_model(xts).logits # [B, L, V]
        V = old_logits.shape[-1]
        x1_equals_v = F.one_hot(x1s.long(), num_classes = V) # [B, L, V]
        curr_logits = self.model(xts).logits
        if temp > 0.0:
            old_logits  /= temp
            curr_logits /= temp
        old_probs = F.softmax(old_logits, dim=-1) # [B, L, V]
        
        loss_type = self.hparams.tm.loss_type
        if loss_type == "itm":
            hr = self.h * rwd # [B,]
            target = old_probs + x1_equals_v * torch.expm1(hr).view(-1, 1, 1) # [B, L, V]
            per_sample_losses = -(target * F.log_softmax(curr_logits, dim=-1)).sum(dim=-1) # [B, L]
            loss = per_sample_losses[mask_indices.bool()].mean()
        elif loss_type == "etm":
            raise NotImplementedError("ETM loss not implemented yet")
        elif loss_type == "sg-itm":
            raise NotImplementedError("stopgrad-ITM loss not implemented yet")
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}")

        log_dict = {
            f"train/loss": loss,
            f"train/a": self.a,
            f"train/h": self.h,
            f"train/drift_gap_kl": self._kl_from_logits(old_logits, curr_logits, mask_indices),
            f"train/rwd_max": rwd.max(),
            f"train/rwd_min": rwd.min(),
            f"train/rwd_mean": rwd.mean(),
            f"train/rwd_std": rwd.std(),
        }
        
        def _to_scalar(x):
            if isinstance(x, torch.Tensor):
                x = x.detach()
                if x.numel() == 1:
                    return x.item()
                raise ValueError("Expected a scalar tensor for logging")
            return x
        
        for k, v in log_dict.items():
            self.dict_for_logs[k] = _to_scalar(v)

        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.dict_for_logs or self._step_counter % self.hparams.metrics_log_every != 0:
            self._step_counter += 1
            return
        # log all at once
        try:
            # Correct for min/max style metrics (sync_dist=True averages across ranks)
            if "train/rwd_min" in self.dict_for_logs:
                local_min = torch.tensor(self.dict_for_logs["train/rwd_min"], device=self.device)
                global_min = self.all_gather(local_min).min().item()
                self.dict_for_logs["train/rwd_min"] = global_min
            if "train/rwd_max" in self.dict_for_logs:
                local_max = torch.tensor(self.dict_for_logs["train/rwd_max"], device=self.device)
                global_max = self.all_gather(local_max).max().item()
                self.dict_for_logs["train/rwd_max"] = global_max
        except Exception:
            pass
        self.log_dict(self.dict_for_logs, on_step=True, on_epoch=False, sync_dist=True)
        self.dict_for_logs = {}
        self._step_counter += 1

    def _prepare_prompts(self, num_dinstinct_prompts, num_completions_per_prompts):
        """
        Select `num_dinstinct_prompts` prompts from `self.training_prompts_dataset`,
        starting at `self.curr_prompt_counter`, repeat each prompt for
        `num_completions_per_prompts` times, and return a tensor of token IDs
        ready to be passed to `self._generate`.

        Returns: input_ids: torch.Tensor
            Shape: [num_dinstinct_prompts * num_completions_per_prompts, prompt_length]
        """
        # ---- 1. Choose distinct prompt indices (with wrap-around) ----
        indices = []
        for offset in range(num_dinstinct_prompts):
            indices.append((self.curr_prompt_counter + offset) % self.training_prompts_dataset_len)
        self.curr_prompt_counter += num_dinstinct_prompts
        self.curr_prompt_counter %= self.training_prompts_dataset_len
        # Remember which dataset rows were used, for reward computation later
        self._last_prompt_indices = indices

        # ---- 2. Extract structured prompts from the dataset ----
        # For get_sudoku_questions_new, each element looks like:
        #   {"prompt": [{"role": "user", "content": "..."}], "puzzle": ..., "solution": ...}
        structured_prompts = [self.training_prompts_dataset[i]["prompt"] for i in indices]

        # ---- 3. Convert structured prompts to plain text and tokenize ----
        prompts_text = []
        for sp in structured_prompts:
            if isinstance(sp, str):
                text = sp # already a plain string
            elif isinstance(sp, list):
                # Typical case for Sudoku / GSM8K / math: [{"role": "...", "content": "..."}]
                text = self.tokenizer.apply_chat_template(sp, tokenize=False, add_generation_prompt=True)
            else:
                raise TypeError(f"Unsupported prompt type {type(sp)} in training_prompts_dataset")
            prompts_text.append(text)

        input_ids = self.tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_prompt_length,
            padding_side="left",
            add_special_tokens=False,
        )["input_ids"].to(self.device)

        return input_ids.repeat_interleave(num_completions_per_prompts, dim=0)

    def _update_buffer(self, model, num_buffer_updates, num_completions_per_prompt):
        """
        Partially update the replay buffer of generated sequences and their rewards.
        - selects `num_buffer_updates` *distinct buffer rows* (along the first
          dimension of `self.buffer`) starting at `self.buffer_update_counter`
          (with wrap-around),
        - generates new completions for fresh prompts for those rows,
        - recomputes rewards for those new samples,
        - writes them into `self.buffer` and `self.buffer_rewards`,
        - and advances `self.buffer_update_counter`.

        Shapes:
          buffer shape: [num_buffer_prompts, num_completions_per_prompt, prompt_len + completion_len]
          buffer_rewards shape: [num_buffer_prompts, num_completions_per_prompt, num_reward_funcs]
        """
        build_or_refresh = "building" if num_buffer_updates == self.num_buffer_prompts else "refreshing"
        print(f"{build_or_refresh} sample buffer ...")
        buffer_start_time = datetime.now()
        device = self.device

        # ---- 1. Prepare prompts as token IDs ----
        if num_buffer_updates == self.num_buffer_prompts:
            update_rows = list(range(self.num_buffer_prompts))
            self.buffer_update_counter = 0
        else:
            update_rows = [
                (self.buffer_update_counter + u) % self.num_buffer_prompts
                for u in range(num_buffer_updates)
            ]
            self.buffer_update_counter += num_buffer_updates
            self.buffer_update_counter %= self.num_buffer_prompts
        prompt_ids = self._prepare_prompts(num_buffer_updates, num_completions_per_prompt)
        total_batch, prompt_len = prompt_ids.shape

        # ---- 2. Run diffusion generation to get prompt+completion sequences ----
        gen_length = self.hparams.max_completion_length
        outputs = []
        chunk_size = max(1, min(self.hparams.tm.buffer_chunk_size, total_batch))
        for start in range(0, total_batch, chunk_size):
            end = min(start + chunk_size, total_batch)
            with torch.no_grad():
                chunk_completion_ids = self._generate(
                    model=model,
                    prompt=prompt_ids[start:end],
                    steps=self.hparams.diffusion_steps,
                    gen_length=gen_length,
                    block_length=self.hparams.block_length,
                    temperature=self.hparams.sampling_temperature,
                    cfg_scale=self.hparams.cfg_scale,
                    remasking=self.hparams.remasking_strategy,
                ) # [chunk_size, seq_len]
            outputs.append(chunk_completion_ids)
        prompt_completion_ids = torch.cat(outputs, dim=0) # [total_batch, seq_len]
        seq_len = prompt_completion_ids.size(1) # seq_len = prompt_len + gen_length
        #TODO: DELETE
        assert seq_len == prompt_len + gen_length, (
            f"Expected seq_len={prompt_len + gen_length}, got {seq_len}"
        )

        # ---- 3. Reshape into [num_updates, num_completions, seq_len] and update corresponding rows ----
        new_buffer_block = prompt_completion_ids.view(num_buffer_updates, -1, seq_len)
        if self.buffer is None:
            self.buffer = new_buffer_block
        else:
            self.buffer[update_rows, :, :] = new_buffer_block

        # ---- 4. Decode completions to text for reward computation ----
        completion_ids = prompt_completion_ids[:, prompt_len:]  # [total_batch, gen_length]
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # ---- 5. Build reward inputs: prompts, completions, and extra dataset columns ----
        data_keys = [key for key in self.training_prompts_dataset[0].keys() if key != "prompt"]
        # For each generated sample we need:
        #   - a structured prompt (list of chat messages)
        #   - a structured completion (list with one assistant message)
        #   - one entry per dataset column (e.g. "answer", "puzzle", "solution", "target", "numbers")
        prompts_for_rewards = []
        completions_for_rewards = []
        reward_kwargs = {key: [] for key in data_keys}

        for row_idx in self._last_prompt_indices:
            row = self.training_prompts_dataset[row_idx]
            base_prompt = row["prompt"]  # list[{"role": ..., "content": ...}, ...]

            for _ in range(num_completions_per_prompt):
                # Structured prompt for this completion
                prompts_for_rewards.append(base_prompt)

                # Copy all extra fields for this completion
                for key in data_keys:
                    reward_kwargs[key].append(row[key])
        # TODO: DELETE
        assert len(prompts_for_rewards) == total_batch
        for key in data_keys:
            assert len(reward_kwargs[key]) == total_batch

        # Turn plain completions into chat-style completions [{"role": "assistant", "content": "..."}]
        completions_for_rewards = []
        for text in completions_text:
            completions_for_rewards.append([{"role": "assistant", "content": text}])

        # ---- 6. Compute rewards for every sequence in the buffer ----
        num_funcs = len(self.reward_funcs)
        rewards_per_func = torch.zeros(total_batch, num_funcs, device=device)

        for j, reward_func in enumerate(self.reward_funcs):
            # We mirror diffu_grpo_trainer:
            # reward_func(prompts=..., completions=..., step=..., run_name=..., **reward_kwargs)
            scores = reward_func(
                prompts=prompts_for_rewards,
                completions=completions_for_rewards,
                step=self._step_counter,
                **reward_kwargs,
            )
            rewards_per_func[:, j] = torch.tensor(scores, device=device, dtype=torch.float32)
        # Store as shape [num_buffer_updates, num_completions_per_prompt, num_funcs]
        new_rewards_block = rewards_per_func.view(num_buffer_updates, -1, num_funcs)
        if self.buffer_rewards is None:
            self.buffer_rewards = new_rewards_block
        else:
            self.buffer_rewards[update_rows, :, :] = new_rewards_block

        buffer_end_time = datetime.now()
        buffer_build_time = (buffer_end_time - buffer_start_time).total_seconds()
        print(f"Finished {build_or_refresh} sample buffer, took {buffer_build_time}")
    
    def _init_tm_scheduler(self):
        """Initialize a per-h-phase, linear LR scheduler with warmup.
        First ramp up from 0 to self.lr over warmup steps,
        keep it constant for a while,
        then linearly anneal from self.lr -> self.lr_min over the remaining steps.
        """
        schedule_type = self.lr_scheduler_type
        opt = self.tm_opt

        self._tm_sched_state = None

        if opt is None or schedule_type is None or schedule_type == "constant":
            return
        if schedule_type != "linear":
            raise NotImplementedError("Only linear LR schedule is implemented")

        assert self.lr_warmup_ratio + self.lr_decay_ratio <= 1.0
        total_steps = self.steps_per_h
        warmup_steps = math.floor(self.lr_warmup_ratio * total_steps)
        decay_steps = math.floor(self.lr_decay_ratio * total_steps)
        const_steps = total_steps - warmup_steps - decay_steps

        base_lrs = [pg["lr"] for pg in opt.param_groups]
        scale = self.lr_min / self.lr
        min_lrs = [lr * scale for lr in base_lrs]

        for pg in opt.param_groups:
            pg["lr"] = 0.0 # start from 0

        self._tm_sched_state = {
            "step": 0,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "const_steps": const_steps,
            "decay_steps": decay_steps,
            "base_lrs": base_lrs,
            "min_lrs": min_lrs,
        }

    def _step_tm_scheduler(self):
        """Advance the warmup + plateau + anneal schedule once per batch."""
        state = self._tm_sched_state
        opt = self.tm_opt

        if state is None or opt is None:
            return

        step = state["step"]
        total_steps = state["total_steps"]
        warmup = state["warmup_steps"]
        const_steps = state["const_steps"]
        decay = state["decay_steps"]
        base_lrs = state["base_lrs"]
        min_lrs = state["min_lrs"]

        # If we've already passed the planned window, clamp to min_lrs
        if step >= total_steps:
            for pg, min_lr in zip(opt.param_groups, min_lrs):
                pg["lr"] = float(min_lr)
            return

        # 1) WARMUP: linear from 0 -> base_lr over warmup_steps
        if warmup > 0 and step < warmup:
            # frac in (0, 1], so LR > 0 from the first step
            frac = float(step + 1) / float(warmup)
            for pg, base_lr in zip(opt.param_groups, base_lrs):
                pg["lr"] = float(base_lr) * frac

        # 2) CONSTANT: keep LR at base_lr
        elif step < warmup + const_steps:
            for pg, base_lr in zip(opt.param_groups, base_lrs):
                pg["lr"] = float(base_lr)

        # 3) ANNEAL: linear from base_lr -> min_lr over anneal_steps
        elif decay > 0:
            # k goes from 0 to anneal_steps-1 over the anneal window
            k = step - warmup - const_steps
            if decay == 1:
                frac = 1.0
            else:
                frac = float(k) / float(decay - 1)

            for pg, base_lr, min_lr in zip(opt.param_groups, base_lrs, min_lrs):
                pg["lr"] = float(base_lr + (min_lr - base_lr) * frac)

        # If decay == 0 and we're past warmup+const_steps, just keep base_lrs
        else:
            for pg, base_lr in zip(opt.param_groups, base_lrs):
                pg["lr"] = float(base_lr)

        state["step"] = step + 1
    
    def _kl_from_logits(self, logits_A, logits_B, mask_indices):
        log_A = F.log_softmax(logits_A, dim=-1) # [B, L, V]
        log_B = F.log_softmax(logits_B, dim=-1)
        kl = F.kl_div(log_A, log_B, reduction='none', log_target=True).sum(-1) # [B, L]
        return kl[mask_indices.bool()].float().mean()

    def _generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.amp.autocast("cuda", enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    # Handle classifier-free guidance more efficiently
                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)

                        # Get logits in a single forward pass
                        logits = model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x).logits

                    # Apply Gumbel noise for sampling
                    logits_with_noise = self._add_gumbel_noise(
                        logits, temperature=temperature, dtype=dtype
                    )
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    del logits_with_noise

                    # Handle remasking strategy
                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(dtype), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                        )
                    elif remasking == "random":
                        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                    else:
                        raise NotImplementedError(remasking)

                    # Ensure we don't process tokens beyond the current block
                    x0_p[:, end_idx:] = float("-inf")

                    # Update masked tokens
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, float("-inf"))

                    # Select tokens to transfer based on confidence
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        num_tokens = num_transfer_tokens[j, i].item()
                        if num_tokens > 0:
                            _, select_index = torch.topk(confidence[j], k=num_tokens)
                            transfer_index[j, select_index] = True

                    x[transfer_index] = x0[transfer_index]
                    del x0, confidence, transfer_index

            return x

    def _get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)
    
    def _add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise
    
    def _build_interpolant(self, x1s, num_to_mask, block_size):
        """
        Given a batch of fully generated sequences x_1, build partially masked x_t.
        Args:
            x1s: Tensor of shape [B, L] where L = max_prompt_length + gen_length.
                This is the batch of x_1's (prompt + completion, no masks).
            num_to_mask: Tensor of shape [B] with values in [1, gen_length].
                This is the number of tokens to mask out for each sample.
            block_size: int. Must divide gen_length. Used to be consistent with the
                        block-wise left-to-right generation schedule.
        Returns:
            xts: Tensor of shape [B, L], the partially masked sequences at time t.
            mask_indices: BoolTensor of shape [B, L], True where tokens are masked.
        """
        device = x1s.device
        B, L = x1s.shape
        prompt_len = self.hparams.max_prompt_length
        gen_len = self.hparams.max_completion_length
        num_blocks = gen_len // block_size

        # Sanity checks
        assert (num_to_mask <= gen_len).all() and (num_to_mask >= 1).all()
        assert L == prompt_len + gen_len
        assert gen_len % block_size == 0

        xts = x1s.clone()
        mask_indices = torch.zeros_like(xts, dtype=torch.bool, device=device)

        # How many whole blocks to mask, and how many extra tokens in the next block
        full_blocks = (num_to_mask - 1) // block_size     # [B]
        remainder   = (num_to_mask - 1) % block_size + 1  # [B]

        # For each sample b, fully mask blocks with id >= num_blocks - full_blocks[b]
        comp_pos = torch.arange(gen_len, device=device)                  # [gen_len]
        block_ids = (comp_pos // block_size).unsqueeze(0).expand(B, -1)  # [B, gen_len]
        full_blocks_threshold = (num_blocks - full_blocks).unsqueeze(1)  # [B, 1]
        full_blocks_to_mask = block_ids >= full_blocks_threshold         # [B, gen_len]

        # Random masking within the "current" block (partial block)
        scores = torch.rand(B, block_size, device=device)                   # [B, block_size]
        ranks = scores.argsort(dim=1).argsort(dim=1)                        # [B, block_size]
        masks_within_block = ranks < remainder.unsqueeze(1)                 # [B, block_size]
        partial_block_start = (full_blocks_threshold - 1) * block_size      # [B, 1]
        idx = partial_block_start + torch.arange(block_size, device=device) # [B, block_size]
        partial_to_mask = torch.zeros(B, gen_len, dtype=torch.bool, device=device)
        partial_to_mask.scatter_(1, idx, masks_within_block)                # [B, gen_len] bools
        to_mask = full_blocks_to_mask | partial_to_mask                     # [B, gen_len] bools

        # Apply mask to completions region
        completion_region = xts[:, prompt_len:]
        completion_region = torch.where(
            to_mask,
            torch.full_like(completion_region, self.mask_id),
            completion_region,
        ) # [B, gen_len]
        xts[:, prompt_len:] = completion_region

        # Record masked positions in the full sequence
        mask_indices[:, prompt_len:] = to_mask

        return xts, mask_indices
