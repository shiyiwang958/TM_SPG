import copy
import logging
import os
from datetime import datetime
import itertools
from webbrowser import get
import wandb
from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
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

        self.a = 0.0
        self.h = self.hparams.tm.h
        self.steps_per_h = self.hparams.tm.steps_per_h
        self.a_end = self.hparams.tm.a_end
        self.mask_id = 126336 # default from LLaDA
        self.checkpoint_freq = getattr(self.hparams.training, "checkpoint_freq", 0.3)
        self.cv = self.hparams.tm.control_variate
        self.buffer = None
        self.buffer_rewards = None
        self.num_buffer_distinct_prompts = self.hparams.tm.num_buffer_distinct_prompts
        self.buffer_update_counter = 0
        self._step_counter = 0
        # TODO: Dropouts?

        self.base_lr = float(getattr(self.hparams.etm, "learning_rate", 1e-3))
        self.lr_schedule = getattr(self.hparams.etm, "lr_schedule", None)
        self.lr_warmup_steps = int(getattr(self.hparams.etm, "lr_warmup_steps", 0))  # warmup per h-phase
        self.min_lr = float(getattr(self.hparams.etm, "min_lr", 0.0))               # target at end of decay in each phase
        self._etm_scheduler = None
        self._etm_sched_state = None

        # TODO: grad-clipping

    def on_train_start(self):
        super().on_train_start()
        # Set up optimizer and LR
        self.tm_opt = self.optimizers()
        for g in self.tm_opt.param_groups:
            g["lr"] = self.hparams.tm.base_lr #TODO: implement LR
        self._init_tm_scheduler()
        # TODO: move training_prompts to device
        # franklin: GPT says train_prompts are plain texts and don't have to move to device. Only torch tensors need to be on device.

        print("Building initial buffer...")
        buffer_start_time = datetime.now()
        self.update_buffer(self.num_buffer_distinct_prompts, self.hparams.tm.num_completions_per_prompt)
        buffer_end_time = datetime.now()
        buffer_build_time = (buffer_end_time - buffer_start_time).total_seconds()
        print(f"Finish building sample buffer, took {buffer_build_time}")

    def _prepare_prompts(self, num_dinstinct_prompts, num_completions_per_prompts):
        """
        Select `num_dinstinct_prompts` prompts from `self.training_prompts_dataset`,
        starting at `self.current_prompt_counter`, repeat each prompt for
        `num_completions_per_prompts` times, and return a tensor of token IDs
        ready to be passed to `self.generate`.

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
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )["input_ids"].to(self.device)

        # ---- 4. Optional prompt-length truncation (keep last tokens) ----
        max_prompt_length = getattr(self.hparams, "max_prompt_length", -1)
        if max_prompt_length > 0:
            input_ids = input_ids[:, -max_prompt_length:]

        return input_ids.repeat_interleave(num_completions_per_prompts, dim=0)

    def update_buffer(self, num_buffer_updates, num_completions_per_prompt):
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
          buffer shape: [num_buffer_distinct_prompts, num_completions_per_prompt, prompt_len + completion_len]
          buffer_rewards shape: [num_buffer_distinct_prompts, num_completions_per_prompt, num_reward_funcs]
        """
        device = next(self.model.parameters()).device

        # ---- 1. Prepare prompts as token IDs ----
        if num_buffer_updates == self.num_buffer_distinct_prompts:
            update_rows = list(range(self.num_buffer_distinct_prompts))
            self.buffer_update_counter = 0
        else:
            update_rows = [
                (self.buffer_update_counter + u) % self.num_buffer_distinct_prompts
                for u in range(num_buffer_updates)
            ]
            self.buffer_update_counter += num_buffer_updates
            self.buffer_update_counter %= self.num_buffer_distinct_prompts
        prompt_ids = self._prepare_prompts(num_buffer_updates, num_completions_per_prompt)
        total_batch, prompt_len = prompt_ids.shape

        # ---- 2. Run diffusion generation to get prompt+completion sequences ----
        gen_length = self.hparams.max_completion_length
        with torch.no_grad():
            prompt_completion_ids = self.generate(
                model=self.base_model,
                prompt=prompt_ids,
                steps=self.hparams.diffusion_steps,
                gen_length=gen_length,
                block_length=self.hparams.block_length,
                temperature=self.hparams.sampling_temperature, #franklin: Temp = 1.0 is much better!
                cfg_scale=self.hparams.cfg_scale,
                remasking=self.hparams.remasking_strategy,
            ) # [total_batch, prompt_len + gen_length]
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
        rewards_per_func = torch.zeros(total_batch, num_funcs, dtype=torch.float32, device=device)

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

    # franklin: maybe getting the model running first and then set up LR scheduler later.
    # otherwise the code is too much to handle at once.

    def generate(
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
        with torch.cuda.amp.autocast(enabled=True):
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
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
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
                            logits_with_noise = self.add_gumbel_noise(
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
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

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

    def get_num_transfer_tokens(self, mask_index, steps):
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
    
    def add_gumbel_noise(self, logits, temperature, dtype):
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



#TODO: dtype; device