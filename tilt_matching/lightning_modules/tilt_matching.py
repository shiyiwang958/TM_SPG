import copy
import logging
import math
import os
from datetime import datetime
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
import itertools
import wandb

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, get_peft_model_state_dict, set_peft_model_state_dict


class TiltMatchingModule(pl.LightningModule):
    def __init__(self, base_model, tokenizer, training_prompts_dataset, reward_funcs, **cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["base_model", "tokenizer", "training_prompts_dataset", "reward_funcs"], logger=False)
        self.tokenizer = tokenizer

        self.student_adapter_name = "student"
        self.teacher_adapter_name = "teacher"

        peft_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type=self.hparams.peft_task_type,
            lora_dropout=self.hparams.lora_dropout,
        )

        peft_wrapped = get_peft_model(base_model, peft_config, adapter_name=self.student_adapter_name)
        peft_wrapped.add_adapter(self.teacher_adapter_name, peft_config)
        student_state = get_peft_model_state_dict(peft_wrapped, adapter_name=self.student_adapter_name)
        set_peft_model_state_dict(peft_wrapped, student_state, adapter_name=self.teacher_adapter_name)

        for name, param in peft_wrapped.named_parameters():
            if f".{self.teacher_adapter_name}" in name:
                param.requires_grad = False

        peft_wrapped.set_adapter(self.student_adapter_name)
        self.model = peft_wrapped

        self.curr_prompt_counter = 0
        self.training_prompts_dataset = training_prompts_dataset
        self.training_prompts_dataset_len = len(training_prompts_dataset)
        self.reward_funcs = reward_funcs
        self.reward_weights = None

        self.a = 0.0
        self.h = self.hparams.tm.h
        self.steps_per_h = self.hparams.tm.steps_per_h
        self.a_end = self.hparams.tm.a_end
        self.mask_id = 126336
        self.checkpoint_freq = self.hparams.checkpoint_freq
        self.cv = self.hparams.tm.control_variate
        self.buffer = None
        self.buffer_rewards = None
        self.num_buffer_prompts = self.hparams.tm.num_buffer_prompts
        self.comps_per_prompt = self.hparams.tm.num_completions_per_prompt
        self.buffer_update_counter = 0
        self._step_counter = 0
        self.dict_for_logs = {}

        self.lr = self.hparams.learning_rate
        self.lr_scheduler_type = self.hparams.lr_scheduler_type
        self.lr_decay_ratio = self.hparams.lr_decay_ratio
        self.lr_warmup_ratio = getattr(self.hparams, "lr_warmup_ratio", 0)
        self.lr_min = getattr(self.hparams, "lr_min", 0.0)
        self._tm_sched_state = None

    @contextmanager
    def _use_adapter(self, adapter_name: str):
        prev = self.model.active_adapter
        self.model.set_adapter(adapter_name)
        try:
            yield
        finally:
            self.model.set_adapter(prev)

    def state_dict(self, destination=None, keep_vars=False):
        destination = OrderedDict() if destination is None else destination

        model_adapter_state = get_peft_model_state_dict(self.model, adapter_name=self.student_adapter_name)
        base_adapter_state = get_peft_model_state_dict(self.model, adapter_name=self.teacher_adapter_name)

        for key, value in model_adapter_state.items():
            tensor = value if keep_vars else value.detach()
            destination[f"model_adapter.{key}"] = tensor.to("cpu")

        for key, value in base_adapter_state.items():
            tensor = value if keep_vars else value.detach()
            destination[f"base_adapter.{key}"] = tensor.to("cpu")

        return destination

    def load_state_dict(self, state_dict, strict=True):
        model_prefix = "model_adapter."
        base_prefix = "base_adapter."

        model_adapter_state = {}
        base_adapter_state = {}
        unexpected_keys = []

        for key, value in state_dict.items():
            if key.startswith(model_prefix):
                model_adapter_state[key[len(model_prefix):]] = value
            elif key.startswith(base_prefix):
                base_adapter_state[key[len(base_prefix):]] = value
            else:
                unexpected_keys.append(key)

        if model_adapter_state:
            missing_model, unexpected_model = set_peft_model_state_dict(
                self.model,
                model_adapter_state,
                adapter_name=self.student_adapter_name,
            )
        else:
            missing_model, unexpected_model = [], []

        if base_adapter_state:
            missing_base, unexpected_base = set_peft_model_state_dict(
                self.model,
                base_adapter_state,
                adapter_name=self.teacher_adapter_name,
            )
        else:
            missing_base, unexpected_base = [], []

        def _relevant_missing(key: str) -> bool:
            # For LoRA checkpoints we only expect adapter weights; ignore base weights.
            return "lora" in key.lower() or "ranknum" in key.lower()

        missing_keys = [k for k in missing_model if _relevant_missing(k)]
        missing_keys.extend(k for k in missing_base if _relevant_missing(k))
        unexpected_keys.extend(list(unexpected_model))
        unexpected_keys.extend(list(unexpected_base))

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                f"missing keys: {missing_keys}; unexpected keys: {unexpected_keys}"
            )

        IncompatibleKeys = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
        return IncompatibleKeys(missing_keys, unexpected_keys)

    def on_train_start(self):
        super().on_train_start()
        # Set up optimizer and LR
        self.tm_opt = self.optimizers()
        for g in self.tm_opt.param_groups:
            g["lr"] = self.lr
        self._init_tm_scheduler()

        self._update_buffer(self.model, self.num_buffer_prompts, self.comps_per_prompt)
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
        self._step_tm_scheduler()
        opt = self.tm_opt
        opt.zero_grad()
        loss = self._tm_step()
        self.manual_backward(loss)

        # Gradient clipping
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm_before = clip_grad_norm_(params, float('inf')).item()
        grad_norm_after = clip_grad_norm_(params, self.hparams.max_grad_norm).item()
        grad_clipped = float(grad_norm_before > self.hparams.max_grad_norm + 1e-6)

        opt.step()
        print(f"current a is {self.a:.4f}")
        print(f"global step is {self.global_step}")

        # Log current learning rate and grad norms
        self.dict_for_logs["train/lr"] = opt.param_groups[0]["lr"]
        self.dict_for_logs['grads/grad_norm_before'] = grad_norm_before
        self.dict_for_logs['grads/grad_norm_after'] = grad_norm_after
        self.dict_for_logs['grads/grad_clipped'] = grad_clipped

        # At each h phase boundary, update a and the teacher adapter; save ckpt if necessary
        if (self.global_step + 1) % self.steps_per_h == 0:
            self.a += self.h
            if self.a + self.h > self.a_end:
                self.h = self.a_end - self.a
            with torch.no_grad():
                adapter_state = get_peft_model_state_dict(self.model, adapter_name=self.student_adapter_name)
                set_peft_model_state_dict(self.model, adapter_state, adapter_name=self.teacher_adapter_name)
                for name, p in self.model.named_parameters():
                    if f".{self.teacher_adapter_name}" in name:
                        p.requires_grad_(False)
            print(f"Degree of tilt a = {self.a:.4f} at step {self.global_step}")

            if self.a >= self.a_end:
                print(f"Reached final a = {self.a_end:.2f}. Training Stopped", flush=True)
                self.trainer.should_stop = True

            # Reset LR for the new h phase
            for g in opt.param_groups:
                g["lr"] = self.lr
            self._init_tm_scheduler()

            # Reset buffer
            self._update_buffer(self.model, self.num_buffer_prompts, self.comps_per_prompt)
            print(f"[DEBUG] Buffer built at step {self.global_step} with shape {self.buffer.shape}")
            self.monitor_sudoku()
            
        # Partially refresh buffer
        elif (self.global_step + 1) % self.hparams.tm.buffer_refresh_steps == 0:
            print(f"[DEBUG] Refreshing {self.hparams.tm.num_buffer_refresh} prompts at step {self.global_step}")
            self._update_buffer(self.model, self.hparams.tm.num_buffer_refresh, self.comps_per_prompt)
        
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
        with torch.no_grad(), self._use_adapter(self.teacher_adapter_name):
            old_logits = self._new_forward(self.model, xts, gen_length) # [B, gen_length, V]
            # old_logits = self.model(xts).logits # [B, L, V]
        V = old_logits.shape[-1]
        x1_equals_v = F.one_hot(x1s.long()[:, -gen_length:], num_classes = V) # [B, gen_length, V]
        with self._use_adapter(self.student_adapter_name):
            curr_logits = self._new_forward(self.model, xts, gen_length) # [B, gen_length, V]
        # curr_logits = self.model(xts).logits
        if temp > 0.0:
            old_logits  /= temp
            curr_logits /= temp
        old_probs = F.softmax(old_logits, dim=-1) # [B, gen_length, V]
        
        loss_type = self.hparams.tm.loss_type
        hr = self.h * rwd # [B,]
        if loss_type == "itm":
            target = self.cv * old_probs + x1_equals_v * (1 - self.cv + torch.expm1(hr)).view(-1, 1, 1) # [B, gen_length, V]
        elif loss_type == "etm":
            target = (1 - hr) * old_probs + x1_equals_v * hr.view(-1, 1, 1) # [B, gen_length, V]
        elif loss_type == "sg-itm":
            curr_probs = F.softmax(curr_logits, dim=-1) # [B, gen_length, V]
            target = self.cv * old_probs + x1_equals_v * (1 - self.cv + torch.expm1(hr)).view(-1, 1, 1) - torch.expm1(hr) * curr_probs.detach()
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}")
        per_sample_losses = -(target * F.log_softmax(curr_logits, dim=-1)).sum(dim=-1) # [B, gen_length]
        loss = per_sample_losses[mask_indices.bool()].mean()

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
        if not self.dict_for_logs or self.global_step % self.hparams.metrics_log_every != 0:
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

        prev_adapter = model.active_adapter
        model.set_adapter(self.teacher_adapter_name)

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
        print(f"Finished {build_or_refresh} reward buffer, took {buffer_build_time}")

        # restore adapter
        model.set_adapter(prev_adapter)
    
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
    
    def monitor_sudoku(self, num_completions=3):
        """
        Have each rank evaluate a single Sudoku whose index matches its global rank.
        """
        print("Checking teacher model on fixed sodokus...")

        rank = getattr(self.trainer, "global_rank", 0)
        world_size = getattr(self.trainer, "world_size", 1)
        
        # prepare prompts for this rank's assigned sudoku
        monitor_start_time = datetime.now()
        monitored_rows = [self.training_prompts_dataset[rank]]
        monitored_prompts = [row["prompt"] for row in monitored_rows]
        monitored_prompt_text = []
        for sp in monitored_prompts:
            if isinstance(sp, str):
                text = sp # already a plain string
            elif isinstance(sp, list):
                # Typical case for Sudoku / GSM8K / math: [{"role": "...", "content": "..."}]
                text = self.tokenizer.apply_chat_template(sp, tokenize=False, add_generation_prompt=True)
            else:
                raise TypeError(f"Unsupported prompt type {type(sp)} in training_prompts_dataset")
            monitored_prompt_text.append(text)
        
        input_ids = self.tokenizer(
            text = monitored_prompt_text,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True,
            max_length = self.hparams.max_prompt_length,
            padding_side = "left",
            add_special_tokens = False,
        )["input_ids"].to(self.device)
        prompt_len =input_ids.shape[1]

        input_ids = input_ids.repeat_interleave(num_completions, dim=0)
        
        with torch.no_grad(), self._use_adapter(self.teacher_adapter_name):
            monitored_answers = self._generate(
                model = self.model,
                prompt = input_ids,
                steps =  self.hparams.diffusion_steps,
                gen_length=self.hparams.max_completion_length,
                block_length=self.hparams.block_length,
                temperature=self.hparams.sampling_temperature,
                cfg_scale=self.hparams.cfg_scale,
                remasking=self.hparams.remasking_strategy,
            ) # [num_soduku * num_completions, seq_len]

        monitored_answers_text = monitored_answers[:, prompt_len:]
        monitored_answers_text = self.tokenizer.batch_decode(monitored_answers_text, skip_special_tokens=True)
        # [num_soduku * num_completions, gen_length]

        # check if each answer is correct
        reward_func = self.reward_funcs[0] # reward_func = sudoku_reward_func

        data_keys = [key for key in monitored_rows[0].keys() if key != "prompt"]
        prompts_for_rewards = []
        reward_kwargs = {key: [] for key in data_keys}

        for row in monitored_rows:
            base_prompt = row["prompt"]
            for _ in range(num_completions):
                prompts_for_rewards.append(base_prompt)
                for key in data_keys:
                    reward_kwargs[key].append(row[key])

        completions_for_rewards = []
        for text in monitored_answers_text:
            completions_for_rewards.append([{"role": "assistant", "content": text}])

        scores = reward_func(
            prompts=prompts_for_rewards,
            completions=completions_for_rewards,
            **reward_kwargs,
        )
        monitor_end_time = datetime.now()
        print(f"Finished checking. Time taken: {monitor_end_time - monitor_start_time}")

        # Gather completions and scores across ranks to let rank 0 log all results.
        gathered_answers = self.all_gather(monitored_answers)            # [world, B, seq_len]
        scores_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)
        gathered_scores = self.all_gather(scores_tensor)                # [world, B]

        if rank != 0:
            return

        answers_flat = gathered_answers.reshape(-1, gathered_answers.shape[-1])
        scores_flat = gathered_scores.reshape(-1).tolist()
        decoded = self.tokenizer.batch_decode(answers_flat[:, prompt_len:], skip_special_tokens=True)

        if wandb.run is not None:
            log_sodoku = {}
            for r in range(world_size):
                start = r * num_completions
                end = start + num_completions
                table = wandb.Table(columns=["puzzle", "completion", "score"])
                puzzle_text = self.training_prompts_dataset[r].get("puzzle", "")
                for idx in range(start, end):
                    table.add_data(
                        puzzle_text,
                        decoded[idx],
                        float(scores_flat[idx]),
                    )
                log_sodoku[f"sudoku_rank_{r}"] = table
            wandb.log(log_sodoku, step=self.global_step)

    
    def _kl_from_logits(self, logits_A, logits_B, mask_indices):
        log_A = F.log_softmax(logits_A, dim=-1)
        log_B = F.log_softmax(logits_B, dim=-1)
        kl = F.kl_div(log_A, log_B, reduction='none', log_target=True).sum(-1)
        return kl[mask_indices.bool()].float().mean()
    
    def on_save_checkpoint(self, checkpoint: dict):
        print(f"saving checkpoint at a = {self.a:.4f}")
        checkpoint["tilt"] = {"a": self.a, "h": self.h}
        checkpoint["hparams"] = copy.deepcopy(self.hparams)
        checkpoint["prompt_counter"] = self.curr_prompt_counter
        checkpoint["_step_counter"] = self._step_counter
        
    
    def on_load_checkpoint(self, checkpoint: dict):
        tilt = checkpoint.get("tilt", None)
        self.a = tilt.get("a", 0.0)
        self.h = tilt.get("h", 2.5e-3)
        self.curr_prompt_counter = checkpoint.get("prompt_counter", 0)
        self._step_counter = checkpoint.get("_step_counter", 0)

        hparams = checkpoint.get("hparams", None)
        self.__dict__["hparams"] = hparams
        self.__dict__["_hparams"] = hparams
        
        

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
            prompt_len = prompt.shape[1]
            x = torch.full((bs, prompt_len + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, :prompt_len] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt_len + num_block * block_length
                end_idx = prompt_len + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x[:, prompt_len:] == mask_id # [B, gen_len]

                    # Handle classifier-free guidance more efficiently
                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)

                        # Get logits in a single forward pass
                        # logits = model(x_).logits
                        logits = self._new_forward(model, x_, gen_length) # [2*B, gen_len, V]
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = self._new_forward(model, x, gen_length) # [B, gen_len, V]
                        # logits_old_suffix = model(x).logits[:, -gen_length:, :] # [B, gen_len, V]
                        # diff = (logits_old_suffix - logits).abs()
                        # if diff.max().item() > 1e-8:
                        #     print("[BUG] Large discrepancy between new_forward and model(x):")
                        #     print("max_abs:", diff.max().item())
                        #     print("max_rel:", (diff / (logits_old_suffix.abs() + 1e-4)).max().item())

                    # Apply Gumbel noise for sampling
                    logits_with_noise = self._add_gumbel_noise(
                        logits, temperature=temperature, dtype=dtype
                    )
                    x0 = torch.argmax(logits_with_noise, dim=-1) # [B, gen_len]
                    del logits_with_noise

                    # Handle remasking strategy
                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(dtype), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                        ) # [B, gen_len]
                    elif remasking == "random":
                        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device) # [B, gen_len]
                    else:
                        raise NotImplementedError(remasking)
                    del logits

                    # Ensure we don't process tokens beyond the current block
                    x0_p[:, end_idx-prompt_len:] = float("-inf")

                    # Update masked tokens
                    x0 = torch.where(mask_index, x0, x[:, prompt_len:])
                    confidence = torch.where(mask_index, x0_p, float("-inf"))

                    # Select tokens to transfer based on confidence
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        num_tokens = num_transfer_tokens[j, i].item()
                        if num_tokens > 0:
                            _, select_index = torch.topk(confidence[j], k=num_tokens)
                            transfer_index[j, select_index] = True

                    x[:, prompt_len:][transfer_index] = x0[transfer_index]
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
            mask_indices: BoolTensor of shape [B, gen_length], True where tokens are masked.
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
        mask_indices = full_blocks_to_mask | partial_to_mask                # [B, gen_len] bools

        # Apply mask to completions region
        completion_region = xts[:, prompt_len:]
        completion_region = torch.where(
            mask_indices,
            torch.full_like(completion_region, self.mask_id),
            completion_region,
        ) # [B, gen_len]
        xts[:, prompt_len:] = completion_region

        return xts, mask_indices

    def _unwrap_llada_core(self, m: torch.nn.Module):
        """
        Get the core LLaDAModel (with .transformer and .config).
        """
        assert isinstance(m, PeftModelForCausalLM)
        lm = m.base_model # peft.tuners.lora.model.LoraModel
        core = getattr(lm, "model", None) # LLaDAModelLM
        if core is None or not hasattr(core.base_model, "transformer"):
            raise ValueError("Expected a LLaDA HF model with .model.transformer")
        return core.base_model # LLaDAModel
    
    def _llada_hidden_no_logits(self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        Run the LLaDA stack up to final layer norm, but DO NOT compute logits yet.
        Args:
            model: PeftModelForCausalLM
            input_ids: [B, L]
            attention_mask: [B, L] with 1 = real token, 0 = pad (HF convention).
        Returns:
            hidden: [B, L, d_model]  (post-ln_f)
        """
        core = self._unwrap_llada_core(model)
        cfg = core.config
        tfm = core.transformer

        # MDM constraints (same as in LLaDAModel.forward)
        assert not cfg.alibi, "Alibi is not supported for LLaDA MDM."
        assert cfg.rope, "Rope must be enabled for LLaDA-8B-Instruct."
        # We don't use KV cache, consistent with MDM constraints.
        use_cache = False
        past_key_values = None

        batch_size, seq_len = input_ids.shape
        past_length = 0

        # ---- Embeddings ----  (lines 2079–2086)
        x = tfm.wte(input_ids)  # [B, L, d_model]
        if cfg.input_emb_norm:
            x = x * (cfg.d_model ** 0.5)

        # No positional embeddings when RoPE is used. (2088–2099 is skipped because rope=True)

        # Embedding dropout (2101–2105)
        x = tfm.emb_drop(x)

        # ---- Attention mask → additive bias ---- (2107–2118)
        if attention_mask is not None and 0.0 in attention_mask:
            # [B, 1, 1, L], 0 for keep, -inf for pad
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        attention_bias = None

        # ---- Merge attention_mask with default bidirectional bias ---- (2122–2179)
        if (
            attention_mask is not None
            or cfg.alibi
            or past_key_values is not None
            or attention_bias is not None
        ):
            if attention_bias is None and cfg.alibi:
                # (we never hit this because cfg.alibi is False for LLaDA-8B-Instruct)
                raise RuntimeError("ALiBi path should be disabled for LLaDA-8B-Instruct")
            elif attention_bias is None:
                # default: bidirectional bias (zeros)
                attention_bias = core.get_bidirectional_attention_bias(past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]

            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask

            # Avoid -inf + -inf → NaNs (2173–2179)
            attention_bias.masked_fill_(attention_bias == float("-inf"), torch.finfo(attention_bias.dtype).min)
        # else: attention_bias stays None

        # ---- Transformer blocks / block groups ---- (2188–2279)
        if cfg.block_group_size == 1:
            for block_idx, block in enumerate(tfm.blocks):
                from configuration_llada import ActivationCheckpointingStrategy
                # (optional) hidden state logging
                # all_hidden_states.append(x)

                layer_past = None  # no KV cache for MDM
                strat = core.activation_checkpointing_strategy

                use_ckpt = (
                    strat == ActivationCheckpointingStrategy.whole_layer
                    or (strat == ActivationCheckpointingStrategy.one_in_two   and block_idx % 2 == 0)
                    or (strat == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
                    or (strat == ActivationCheckpointingStrategy.one_in_four  and block_idx % 4 == 0)
                )

                if use_ckpt:
                    x, _ = core._activation_checkpoint_fn(
                        block,
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                    )
                else:
                    x, _ = block(
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                    )
        else:
            for group_idx, block_group in enumerate(tfm.block_groups):
                # all_hidden_states.append(x)
                layers_past = None  # no KV cache
                x, _ = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache,
                )

        # We do **not** use last_logits_only here; we want full [B, L, d_model].

        # ---- Final layer norm (2286–2290) ----
        x = tfm.ln_f(x)  # [B, L, d_model]

        return x
    
    def _llada_logits_on_suffix(self,
        model: torch.nn.Module,
        hidden: torch.Tensor,  # [B, L, d_model] from llada_hidden_no_logits
        gen_len: int,
    ) -> torch.Tensor:
        """
        Compute logits **only** for the last `gen_len` positions of each sequence.

        Args:
            model: PeftModelForCausalLM or bare LLaDA HF model.
            hidden: [B, L, d_model] (post-ln_f).
            gen_len: number of completion tokens at the end of the sequence.

        Returns:
            logits_suffix: [B, gen_len, V]
        """
        lm = model.base_model
        core = self._unwrap_llada_core(model)
        cfg = core.config

        B, L, d_model = hidden.shape
        assert gen_len <= L, f"gen_len={gen_len} cannot exceed sequence length L={L}"
        hidden_suffix = hidden[:, -gen_len:, :]  # [B, gen_len, d_model]

        # Get the output embedding / projection the same way HF does.
        out_module = lm.get_output_embeddings()  # nn.Embedding or nn.Linear 

        if isinstance(out_module, torch.nn.Embedding):
            # Weight tying case: logits = F.linear(x, wte.weight)
            weight = out_module.weight          # [V, d_model]
            bias = None
            logits = F.linear(hidden_suffix, weight, bias)
        elif isinstance(out_module, torch.nn.Linear):
            # Non-tying case: use ff_out directly
            logits = out_module(hidden_suffix)  # [B, gen_len, V]
        else:
            raise TypeError(
                f"Unsupported output embeddings module type: {type(out_module)} "
                "(expected nn.Embedding or nn.Linear)."
            )

        if getattr(cfg, "scale_logits", False):
            logits = logits * (1.0 / math.sqrt(cfg.d_model))

        return logits  # [B, gen_len, V]

    def _new_forward(self, model, x, gen_length):
        # x: [B, L]
        hidden = self._llada_hidden_no_logits(model, x, attention_mask=None)
        return self._llada_logits_on_suffix(model, hidden, gen_length)  # [B, gen_len, V]
