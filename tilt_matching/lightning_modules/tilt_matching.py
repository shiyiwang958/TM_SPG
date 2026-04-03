import copy
import logging
import math
import os
from datetime import datetime
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
import itertools
import re
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, get_peft_model_state_dict, set_peft_model_state_dict
from data_utils import SYSTEM_PROMPT


class DTMModule(pl.LightningModule):
    def __init__(self, base_model, tokenizer, train_set, validation_set, reward_funcs, **cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["base_model", "tokenizer", "train_set", "validation_set", "reward_funcs"], logger=False)
        self.tokenizer = tokenizer

        # --- Set up student and teacher models with PEFT LoRA adapters ---
        peft_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type=self.hparams.peft_task_type,
            lora_dropout=self.hparams.lora_dropout,
        )
        peft_wrapped = get_peft_model(base_model, peft_config, adapter_name="student")
        peft_wrapped.add_adapter("teacher", peft_config)
        student_state = get_peft_model_state_dict(peft_wrapped, adapter_name="student")
        set_peft_model_state_dict(peft_wrapped, student_state, adapter_name="teacher")

        for name, param in peft_wrapped.named_parameters():
            if ".teacher" in name:
                param.requires_grad = False
        peft_wrapped.set_adapter("student")
        self.model = peft_wrapped

        # --- Initialize dataset, reward functions, and buffer-related variables ---
        self.curr_prompt_counter = 0
        self.train_set = train_set
        self.train_set_len = len(train_set)
        self.validation_set = validation_set
        self.reward_funcs = reward_funcs

        self.buffer = None
        self.buffer_rewards = None
        self._rebuild_buffer_next_phase = False
        self.num_buffer_prompts = self.hparams.tm.num_buffer_prompts
        self.comps_per_prompt = self.hparams.tm.num_completions_per_prompt
        self.buffer_update_counter = 0
        self._grad_accum_counter = 0
        self._step_counter = 0
        self._eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if self._eos_id is None:
            raise ValueError("tokenizer.eos_token_id must be set to exclude post-EOS positions from the loss.")

        # --- Set up DTM hyperparameters ---
        self.a = 0.0
        self.h = self.hparams.tm.h
        self.steps_per_h = self.hparams.tm.steps_per_h
        self.a_end = self.hparams.tm.a_end
        self.mask_id = 126336
        self.rwd_shift = self.hparams.tm.rwd_shift
        self.cv = self.hparams.tm.control_variate

        self.lr = self.hparams.learning_rate
        self.lr_scheduler_type = self.hparams.lr_scheduler_type
        self.lr_decay_ratio = self.hparams.lr_decay_ratio
        self.lr_warmup_ratio = getattr(self.hparams, "lr_warmup_ratio", 0)
        self.lr_min = getattr(self.hparams, "lr_min", 0.0)
        self._tm_sched_state = None

        # EMA over student adapter weights (per h-phase)
        self.use_ema = getattr(self.hparams.tm, "use_ema", True)
        self.ema_decay = getattr(self.hparams.tm, "ema_decay", 0.999)
        self._ema_shadow = None

        # --- Set up control variate estimation and logging ---
        self._cv_num_accum = None  # sum of w*<pi_theta-delta, delta-pi_a> over masked positions
        self._cv_den_accum = None  # sum of ||delta-pi_a||^2 over masked positions
        self._cv_ema_beta = float(getattr(self.hparams.tm, "control_variate_ema", 0.05))
        self._track_cv = bool(getattr(self.hparams.tm, "track_cv", False))
        self._compute_cv = bool(getattr(self.hparams.tm, "learned_cv", False) or self._track_cv)
        self._use_sar_discounted_future_loss = bool(getattr(self.hparams.tm, "use_sar_discounted_future_loss", False))
        self._sar_future_discount_alpha = float(getattr(self.hparams.tm, "sar_future_discount_alpha", 1.0))
        if self._use_sar_discounted_future_loss and not (0.0 <= self._sar_future_discount_alpha <= 1.0):
            raise ValueError(
                "tm.sar_future_discount_alpha must be in [0, 1] when tm.use_sar_discounted_future_loss is enabled"
            )
        self.dict_for_logs = {}
        self.ckpt_counter = 0
        self.log_student_steps = self.hparams.tm.student_log_steps
        self.log_student = getattr(self.hparams.tm, "log_student", True)
        self.log_temperature = getattr(self.hparams.tm, "log_temperature", 0.0) 
        self.student_logs_per_prompt = getattr(self.hparams.tm, "student_logs_per_prompt", 1)
        # for micro-step metric accumulation:
        self._micro_log_sums = {}
        self._micro_log_counts = {}
        self._micro_log_mins = {}
        self._micro_log_maxs = {}

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

        model_adapter_state = get_peft_model_state_dict(self.model, adapter_name="student")
        base_adapter_state = get_peft_model_state_dict(self.model, adapter_name="teacher")

        for key, value in model_adapter_state.items():
            tensor = value if keep_vars else value.detach()
            destination[f"model_adapter.{key}"] = tensor.to("cpu")

        for key, value in base_adapter_state.items():
            tensor = value if keep_vars else value.detach()
            destination[f"base_adapter.{key}"] = tensor.to("cpu")

        return destination

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Load adapter weights saved via `state_dict`.
        Expects keys with prefixes `model_adapter.` (student) and `base_adapter.` (teacher).
        
        Args:
            state_dict: Dictionary containing adapter weights
            strict: If True, raise error on missing or unexpected keys

        student_copy_over: If True, load student adapter weights into both student and teacher.
            If False, load normally (student from model_adapter, teacher from base_adapter).
        
        Returns a dict mirroring torch's load_state_dict with any missing or unexpected keys.
        """
        # Allow control via hparams if not explicitly specified
        student_copy_over = getattr(self.hparams, "student_copy_over", False)
        
        expected_student = set(get_peft_model_state_dict(self.model, adapter_name="student").keys())
        expected_teacher = set(get_peft_model_state_dict(self.model, adapter_name="teacher").keys())

        student_state = OrderedDict()
        teacher_state = OrderedDict()
        unexpected_keys = []

        for key, value in state_dict.items():
            if key.startswith("model_adapter."):
                bare = key[len("model_adapter."):]
                student_state[bare] = value.to(self.model.device)
                # If student_copy_over is True, also load into teacher
                if student_copy_over:
                    teacher_state[bare] = value.to(self.model.device) 
            elif key.startswith("base_adapter."):
                bare = key[len("base_adapter."):]
                # Only load base_adapter into teacher if student_copy_over is False
                if not student_copy_over:
                    teacher_state[bare] = value.to(self.model.device)
            else:
                unexpected_keys.append(key)

        set_peft_model_state_dict(self.model, student_state, adapter_name="student")
        set_peft_model_state_dict(self.model, teacher_state, adapter_name="teacher")

        missing_student = list(expected_student - set(student_state.keys()))
        # Only check for missing teacher keys if we're actually loading them
        missing_teacher = list(expected_teacher - set(teacher_state.keys()))
        missing_keys = [f"model_adapter.{k}" for k in missing_student] + [f"base_adapter.{k}" for k in missing_teacher]

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Error(s) in loading state_dict: missing keys {missing_keys}; unexpected keys {unexpected_keys}"
            )

        return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

    def on_train_start(self):
        super().on_train_start()
        if getattr(self.hparams, "eval", False):
            self.logging_student(self.model, self.student_logs_per_prompt)

        global_world_size = getattr(self.trainer, "world_size", 1)
        global_rank = getattr(self.trainer, "global_rank", 0)
        logical_world_size = min(global_world_size, self.hparams.world_size)
        logical_rank = global_rank % logical_world_size
        self.g = torch.Generator(device=self.device)
        self.g.manual_seed(12345 + logical_rank)

        self._start_step = getattr(self, "global_step", 0)
        
        # Set up optimizer and LR
        self.tm_opt = self.optimizers()
        for g in self.tm_opt.param_groups:
            g["lr"] = self.lr
        self._init_tm_scheduler()

        self._update_buffer(self.model, self.num_buffer_prompts, self.comps_per_prompt)
        print(f"[DEBUG] Buffer initialized with shape {self.buffer.shape}")
        # Initialize EMA for the current h-phase
        self._reset_ema()
    
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
    
    def on_train_batch_start(self, batch, batch_idx):
        # If we scheduled a full buffer rebuild at the end of the previous h-phase,
        # do it now (i.e., at the start of the new h-phase), after checkpointing.
        if getattr(self, "_rebuild_buffer_next_phase", False):
            self._update_buffer(self.model, self.num_buffer_prompts, self.comps_per_prompt)
            print(f"[DEBUG] Buffer built at start of new h-phase (global step {self.global_step})")
            self._rebuild_buffer_next_phase = False
    
    # ---------------------------
    # Micro-step metric averaging
    # ---------------------------
    @staticmethod
    def _to_scalar(x):
        """Convert a 0-dim tensor (or python scalar) to a python float for accumulation."""
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.numel() == 1:
                return x.item()
            raise ValueError("Expected a scalar tensor for logging/accumulation")
        return x

    @staticmethod
    def _extract_boxed_answer(text):
        """Return the last \boxed{...} content from a string, or None if missing."""
        if text is None:
            return None
        matches = re.findall(r"\\boxed\{(.*?)\}", str(text), re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()
    
    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    def _reset_micro_log_accum(self):
        self._micro_log_sums = {}
        self._micro_log_counts = {}
        self._micro_log_mins = {}
        self._micro_log_maxs = {}

    def _accumulate_micro_log_dict(self, log_dict):
        """Accumulate per-microstep metrics into running sums/mins/maxes for the current window."""
        for k, v in log_dict.items():
            val = self._to_scalar(v)
            if not isinstance(val, (int, float)):
                raise ValueError(f"Non-numeric metric {k}={type(val)} cannot be accumulated")
            val = float(val)

            # Keep intuitive semantics for *_min and *_max metrics across micro-steps
            if k.endswith("_min"):
                prev = self._micro_log_mins.get(k, float("inf"))
                self._micro_log_mins[k] = min(prev, val)
            elif k.endswith("_max"):
                prev = self._micro_log_maxs.get(k, -float("inf"))
                self._micro_log_maxs[k] = max(prev, val)
            else:
                self._micro_log_sums[k] = self._micro_log_sums.get(k, 0.0) + val
                self._micro_log_counts[k] = self._micro_log_counts.get(k, 0) + 1

    def _finalize_micro_log_dict(self):
        """Compute the averaged metrics for the current accumulation window."""
        out = {}
        for k, s in self._micro_log_sums.items():
            c = self._micro_log_counts.get(k, 1)
            # Keep total sums for *_sum / *_count metrics; average everything else.
            if k.endswith(("_sum", "_count")):
                out[k] = s
            else:
                out[k] = s / float(c)
        out.update(self._micro_log_mins)
        out.update(self._micro_log_maxs)
        return out

    # ---------------------------
    # EMA (student adapter only)
    # ---------------------------
    def _student_param_iter(self):
        state = get_peft_model_state_dict(self.model, adapter_name="student")
        for key, value in state.items():
            yield key, value

    def _reset_ema(self):
        self._ema_shadow = {}
        for key, param in self._student_param_iter():
            self._ema_shadow[key] = param.detach().clone()

    @torch.no_grad()
    def _sync_phase_models_from_ema(self):
        student_state = get_peft_model_state_dict(self.model, adapter_name="student")

        if self.use_ema:
            if self._ema_shadow is None:
                raise RuntimeError("EMA shadow is missing at phase boundary.")
            missing_keys = sorted(set(student_state.keys()) - set(self._ema_shadow.keys()))
            if missing_keys:
                raise RuntimeError(f"EMA shadow is missing keys at phase boundary: {missing_keys}")
            set_peft_model_state_dict(self.model, self._ema_shadow, adapter_name="student")
            set_peft_model_state_dict(self.model, self._ema_shadow, adapter_name="teacher")
        else:
            set_peft_model_state_dict(self.model, student_state, adapter_name="teacher")

        for name, p in self.model.named_parameters():
            if ".teacher" in name:
                p.requires_grad_(False)
        self.model.set_adapter("student")

    def _reset_phase_optimizer_state(self):
        if self.tm_opt is None:
            return
        self.tm_opt.state.clear()
    
    @torch.no_grad()
    def _update_ema(self):
        if self._ema_shadow is None:
            self._reset_ema()
            return
        decay = float(self.ema_decay)
        with torch.no_grad():
            for key, param in self._student_param_iter():
                if key not in self._ema_shadow:
                    self._ema_shadow[key] = param.detach().clone()
                else:
                    self._ema_shadow[key].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def training_step(self, batch, batch_idx):
        """
        Perform one training micro-step of Tilt Matching.

        We use manual optimization, so we implement gradient accumulation ourselves:
          - run `_tm_step()` every call (new sampled batch each micro-step)
          - accumulate grads for `grad_accum_steps` micro-steps
          - run a single optimizer update + LR schedule step on the last micro-step

        Note: When using DDP, we disable gradient synchronization on non-update
        micro-steps to avoid an all-reduce every backward pass.

        Args:
            batch: Dummy batch (not used, for Lightning interface only).
            batch_idx: Index of the batch (not used, for Lightning interface only).
        """
        opt = self.tm_opt
        accum = self.hparams.tm.grad_accum_steps

        # At the start of a new accumulation window:
        if (self._grad_accum_counter % accum) == 0:
            if getattr(self.hparams.tm, "mix_batches", False):
                total_sample_needed = self.hparams.tm.num_batch_prompts * accum
                self._accum_prompts_idx = torch.randperm(self.buffer.shape[0] * self.comps_per_prompt, device=self.device, generator=self.g)[:total_sample_needed]              
            else:
                total_prompts_needed = self.hparams.tm.num_batch_prompts * accum
                self._accum_prompts_idx = torch.randperm(self.buffer.shape[0], device=self.device, generator=self.g)[:total_prompts_needed]
            self._reset_micro_log_accum()
            if self._compute_cv:
                self._cv_num_accum = torch.zeros((), device=self.device)
                self._cv_den_accum = torch.zeros((), device=self.device)
            else:
                self._cv_num_accum = None
                self._cv_den_accum = None

        loss, micro_log_dict = self._tm_step()
        self._accumulate_micro_log_dict(micro_log_dict)
        loss_scaled = loss / float(accum)

        # Backward (avoid DDP grad sync on non-update micro-steps)
        self._grad_accum_counter += 1
        is_update_step = (self._grad_accum_counter % accum) == 0
        if not is_update_step:
            with self.trainer.model.no_sync():
                self.manual_backward(loss_scaled)
            self.dict_for_logs = {}
            return
        else:
            self.manual_backward(loss_scaled)

        # ---- Perform gradient update on the student and step optimizer ----
        self._step_tm_scheduler()
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm_before = clip_grad_norm_(params, self.hparams.max_grad_norm).item()
        grad_norm_after = clip_grad_norm_(params, float("inf")).item()
        grad_clipped = float(grad_norm_before > self.hparams.max_grad_norm + 1e-6)

        opt.step()
        opt.zero_grad(set_to_none=True)
        if self.use_ema:
            self._update_ema()

        # ---- Update control variate once per global step (after grad accumulation, synced across GPUs) ----
        # Aggregate across all GPUs
        if self._compute_cv:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(self._cv_num_accum, op=dist.ReduceOp.SUM)
                dist.all_reduce(self._cv_den_accum, op=dist.ReduceOp.SUM)

            c_batch = (-self._cv_num_accum / self._cv_den_accum.clamp_min(1e-12))
            cv_old = torch.tensor(float(self.cv), device=self.device)
            cv_new = (1.0 - self._cv_ema_beta) * cv_old + self._cv_ema_beta * c_batch
            self.cv = float(cv_new.item())

        # Build averaged metrics (over micro-steps) for this optimizer/global step
        self.dict_for_logs = self._finalize_micro_log_dict()

        # Log current learning rate and grad norms
        self.dict_for_logs["train/lr"] = opt.param_groups[0]["lr"]
        self.dict_for_logs["grads/grad_norm_before"] = grad_norm_before
        self.dict_for_logs["grads/grad_norm_after"] = grad_norm_after
        self.dict_for_logs["grads/grad_clipped"] = grad_clipped
        if self._compute_cv:
            self.dict_for_logs["train/cv"] = self.cv

        # At each h phase boundary, sync teacher/student to the EMA state.
        if (self.global_step - self._start_step) % self.steps_per_h == 0:
            self.a += self.h
            if self.a + self.h > self.a_end:
                self.h = self.a_end - self.a
            with torch.no_grad():
                self._sync_phase_models_from_ema()
            self._reset_phase_optimizer_state()
            print(f"Model weights copied. Degree of tilt a = {self.a:.4f} at global step {self.global_step}")

            if self.a >= self.a_end:
                print(f"Reached final a = {self.a_end:.2f}. Training Stopped", flush=True)
                self.trainer.should_stop = True

            # Reset LR for the new h phase
            for g in opt.param_groups:
                g["lr"] = self.lr
            self._init_tm_scheduler()
            # Reset EMA for the new h-phase
            self._reset_ema()

        self.log("ckpt_a", self.a, on_step=True, on_epoch=False, sync_dist=True)

    def _tm_step(self):
        num_buffer_prompts, comps_per_prompt, L = self.buffer.shape
        num_batch_prompts = self.hparams.tm.num_batch_prompts
        gen_length = self.hparams.max_completion_length

        # Draw a batch from the buffer
        if getattr(self.hparams.tm, "mix_batches", False):
            B = num_batch_prompts
            start_idx = (self._grad_accum_counter % self.hparams.tm.grad_accum_steps) * B
            prompts_idx = self._accum_prompts_idx[start_idx:start_idx + B]
            flat_buffer = self.buffer.reshape(-1, L)
            flat_rewards = self.buffer_rewards.reshape(-1, self.buffer_rewards.shape[-1])
            x1s = flat_buffer[prompts_idx]           # [B, L]
            del flat_buffer
            rwds = flat_rewards[prompts_idx]         # [B, num_reward_funcs]
            del flat_rewards
        else:
            B = num_batch_prompts * comps_per_prompt
            start_idx = (self._grad_accum_counter % self.hparams.tm.grad_accum_steps) * num_batch_prompts
            prompts_idx = self._accum_prompts_idx[start_idx:start_idx + num_batch_prompts]
            x1s = self.buffer[prompts_idx].reshape(B, L)           # [B, L]
            rwds = self.buffer_rewards[prompts_idx].reshape(B, -1) # [B, num_reward_funcs]

        # Aggregate rewards from multiple functions
        weights = torch.ones(rwds.shape[1], device=self.device, dtype=rwds.dtype)
        rwd = torch.nansum(rwds * weights.unsqueeze(0), dim=1) # [B,]
        if self.hparams.dataset == "gsm8k":
            correct_frac = torch.isclose(rwds[:, -1], 2.0 * torch.ones_like(rwd), atol=1e-6, rtol=0.0).float().mean()
        elif self.hparams.dataset == "math":
            correct_frac = torch.isclose(rwds[:, 0], 2.0 * torch.ones_like(rwd), atol=1e-6, rtol=0.0).float().mean()
        else:
            correct_frac = torch.isclose(rwd, self.hparams.max_rwd * torch.ones_like(rwd), atol=1e-6, rtol=0.0).float().mean()
        
        # Create x_t's by masking the x_1's
        num_to_mask = torch.randint(low=1, high=gen_length+1, size=(x1s.shape[0],), device=self.device)
        itpl_block_len = getattr(self.hparams.tm, "itpl_block_length", self.hparams.block_length)
        xts, mask_indices, active_block_mask = self._build_interpolant(x1s, num_to_mask, itpl_block_len)
        aux_mask, loss_weights = self._build_loss_weights(
            completion_ids=x1s[:, -gen_length:],
            mask_indices=mask_indices,
            active_block_mask=active_block_mask,
            block_size=itpl_block_len,
            dtype=torch.float32,
        )

        # Get model predictions and compute loss
        temp = self.hparams.sampling_temperature
        with torch.no_grad(), self._use_adapter("teacher"):
            self.model.eval()
            old_logits = self._new_forward(self.model, xts, gen_length) # [B, gen_length, V]
        V = old_logits.shape[-1]
        x1_equals_v = F.one_hot(x1s.long()[:, -gen_length:], num_classes = V) # [B, gen_length, V]
        with self._use_adapter("student"):
            self.model.train()   
            curr_logits = self._new_forward(self.model, xts, gen_length) # [B, gen_length, V]
        if temp > 0.0 and self.hparams.tm.rescale_logits:
            old_logits  /= temp
            curr_logits /= temp
        old_probs = F.softmax(old_logits, dim=-1) # [B, gen_length, V]
        with torch.no_grad():
            curr_probs_ng = F.softmax(curr_logits, dim=-1)  # [B, gen_length, V]
        
        # shift reward for minimizing gradient variance for loss computation
        hr = self.h * (rwd + self.rwd_shift) # [B,]

        if self._compute_cv:
            # learnable/diagnostic control variate accumulation
            with torch.no_grad():
                w = torch.exp(hr).view(-1, 1)  # [B,1]

                # A = delta - pi_a, B = pi_theta - delta
                A = x1_equals_v - old_probs       # [B,L,V]
                Bv = curr_probs_ng - x1_equals_v  # [B,L,V]

                dot = (Bv * A).sum(dim=-1)  # [B,L]
                den = (A * A).sum(dim=-1)   # [B,L]

                self._cv_num_accum += (w * dot)[aux_mask].sum()
                self._cv_den_accum += den[aux_mask].sum()

        active_cv = self.cv if getattr(self.hparams.tm, "learned_cv", False) else self.hparams.tm.control_variate

        loss_type = self.hparams.tm.loss_type
        if loss_type == "itm":
            target = active_cv * old_probs + x1_equals_v * (1 - active_cv + torch.expm1(hr)).view(-1, 1, 1) # [B, gen_length, V]
        elif loss_type == "etm":
            target = (1 - hr) * old_probs + x1_equals_v * hr.view(-1, 1, 1) # [B, gen_length, V]
        elif loss_type == "sg-itm":
            curr_probs = F.softmax(curr_logits, dim=-1) # [B, gen_length, V]
            target = active_cv * old_probs + x1_equals_v * (1 - active_cv + torch.expm1(hr)).view(-1, 1, 1) - torch.expm1(hr) * curr_probs.detach()
        elif loss_type == "final-phase":
            if self.hparams.dataset == "gsm8k":
                target = x1_equals_v * rwds[:, -1].view(-1, 1, 1) # [B, gen_length, V]
            elif self.hparams.dataset == "math":
                target = x1_equals_v * rwds[:, 0].view(-1, 1, 1) # [B, gen_length, V]
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}")
        
        per_position_losses = -(target * F.log_softmax(curr_logits, dim=-1)).sum(dim=-1) # [B, gen_length]
        loss_weights = loss_weights.to(per_position_losses.dtype)
        per_row_losses = (per_position_losses * loss_weights).sum(dim=1)  # [B]
        per_row_weight = loss_weights.sum(dim=1).clamp_min(1e-12)
        loss = (per_row_losses / per_row_weight).mean()

        log_dict = {
            f"train/loss": loss,
            f"train/a": self.a,
            f"train/h": self.h,
            f"train/drift_gap_kl": self._kl_from_logits(old_logits, curr_logits, aux_mask),
            f"train/rwd_max": rwd.max(),
            f"train/rwd_min": rwd.min(),
            f"train/rwd_mean": rwd.mean(),
            f"train/rwd_std": rwd.std(),
            f"train/correct_frac": correct_frac,
            f"charts/step_counter": self._step_counter,
        }
        
        return loss, log_dict
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (self._grad_accum_counter % self.hparams.tm.grad_accum_steps) == 0:
            if (self.global_step - self._start_step + 1) % self.hparams.ckpt_freq == 0:
                self.ckpt_counter += 1
                self.log("ckpt_counter", self.ckpt_counter, on_step=True, on_epoch=False, sync_dist=True)
            if (self.global_step - self._start_step) % self.log_student_steps == 0 and self.log_student:
                self.logging_student(self.model, self.student_logs_per_prompt)
            if (self.global_step - self._start_step) % self.steps_per_h == 0:
                self._rebuild_buffer_next_phase = True
                print(f"[DEBUG] Scheduled buffer rebuild for next h-phase at global step {self.global_step}")
            # Partially refresh buffer
            elif (self.global_step - self._start_step) % self.hparams.tm.buffer_refresh_steps == 0:
                print(f"[DEBUG] Refreshing {self.hparams.tm.num_buffer_refresh} prompts at global step {self.global_step}")
                self._update_buffer(self.model, self.hparams.tm.num_buffer_refresh, self.comps_per_prompt)
        if not self.dict_for_logs or (self.global_step - self._start_step - 1) % self.hparams.metrics_log_every != 0:
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
    
    def on_save_checkpoint(self, checkpoint: dict):
        print(f"saving checkpoint at global step = {self.global_step}")
        checkpoint["tilt_a"] = self.a
        checkpoint["prompt_counter"] = self.curr_prompt_counter
        checkpoint["tm_sched_state"] = copy.deepcopy(getattr(self, "_tm_sched_state", None))
        checkpoint["step_counter"] = self._step_counter + 1 # since ckpt saving is before on_train_batch_end
        checkpoint["ckpt_counter"] = self.ckpt_counter
        
    def on_load_checkpoint(self, checkpoint: dict):
        self.a = checkpoint.get("tilt_a", 0.0)
        self.curr_prompt_counter = checkpoint.get("prompt_counter", 0)
        self._tm_sched_state = checkpoint.get("tm_sched_state", None)
        self.cv = float(checkpoint.get("cv", getattr(self, "cv", 0.0)))
        self._step_counter = checkpoint.get("step_counter", 0)
        self._resuming_from_ckpt = True
        self.ckpt_counter = checkpoint.get("ckpt_counter", 0)

    def _prepare_prompts(self, num_dinstinct_prompts, num_completions_per_prompts):
        """
        Select `num_dinstinct_prompts` prompts from `self.train_set`,
        starting at `self.curr_prompt_counter`, repeat each prompt for
        `num_completions_per_prompts` times, and return a tensor of token IDs
        ready to be passed to `self._generate`.

        Returns: input_ids: torch.Tensor
            Shape: [num_dinstinct_prompts * num_completions_per_prompts, prompt_length]
        """
        # Get DDP info (defaults to 1 if not distributed)
        # We only want 8 unique prompt groups. Ranks separated by 8 (0/8, 1/9, ...)
        # will share the same prompts when running with 16 GPUs.
        global_world_size = getattr(self.trainer, "world_size", 1)
        global_rank = getattr(self.trainer, "global_rank", 0)
        logical_world_size = min(global_world_size, self.hparams.world_size)
        logical_rank = global_rank % logical_world_size

        # ---- 1. Choose distinct prompt indices (with wrap-around) ----
        indices = []
        for offset in range(num_dinstinct_prompts):
            idx = (
                self.curr_prompt_counter
                + (offset * logical_world_size)
                + logical_rank
            ) % self.train_set_len
            indices.append(idx)
        self.curr_prompt_counter += (num_dinstinct_prompts * logical_world_size)
        self.curr_prompt_counter %= self.train_set_len
        # Remember which dataset rows were used, for reward computation later
        self._last_prompt_indices = indices

        # ---- 2. Extract structured prompts from the dataset ----
        # For get_countdown_questions, each element looks like:
        #   {"prompt": [{"role": "user", "content": "..."}], "target": ..., "numbers": ...}
        structured_prompts = [self.train_set[i]["prompt"] for i in indices]

        # ---- 3. Convert structured prompts to plain text and tokenize ----
        prompts_text = []
        for sp in structured_prompts:
            if isinstance(sp, str):
                text = sp # already a plain string
            elif isinstance(sp, list):
                # Typical case for Sudoku / GSM8K / math: [{"role": "...", "content": "..."}]
                text = self.tokenizer.apply_chat_template(sp, tokenize=False, add_generation_prompt=True)
            else:
                raise TypeError(f"Unsupported prompt type {type(sp)} in train_set")
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

        # # Debug prompt length
        # prompt_input = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        # print(f"Prompts are: {prompt_input[:2]} ...")

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
        
        device = self.device

        prev_adapter = model.active_adapter
        model.set_adapter("teacher")
        model.eval()
        print(f"{build_or_refresh} sample buffer ...")
        buffer_start_time = datetime.now()

        # ---- 1. Prepare prompts as token IDs ----
        if num_buffer_updates == self.num_buffer_prompts:
            update_rows = list(range(self.num_buffer_prompts))
            self.buffer_update_counter = 0
            self.buffer = None
            self.buffer_rewards = None
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
        chunk_size = max(1, min(self.hparams.tm.buffer_chunk_size, total_batch))
        gen_length = self.hparams.max_completion_length
        seq_len = prompt_len + gen_length
        # pre-allocate
        prompt_completion_ids = torch.empty(
            (total_batch, seq_len),
            device=prompt_ids.device,
            dtype=prompt_ids.dtype,
        )
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
                )  # [end-start, seq_len]

            prompt_completion_ids[start:end].copy_(chunk_completion_ids)

        # ---- 3. Reshape into [num_updates, num_completions, seq_len] and update corresponding rows ----
        new_buffer_block = prompt_completion_ids.view(num_buffer_updates, -1, seq_len)
        if self.buffer is None:
            self.buffer = new_buffer_block
        else:
            self.buffer[update_rows, :, :] = new_buffer_block

        # ---- 4. Decode completions to text for reward computation ----
        completion_ids = prompt_completion_ids[:, prompt_len:]  # [total_batch, gen_length]
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # print_mask = (torch.rand(completion_ids.shape[0], device=completion_ids.device) < 0.1)
        # for i in range(completion_ids.shape[0]):
        #     if print_mask[i]:
        #         print(f"Completion {i+1}/{completion_ids.shape[0]}: {completions_text[i]}")
        
        # ---- 5. Build reward inputs: prompts, completions, and extra dataset columns ----
        data_keys = [key for key in self.train_set[0].keys() if key != "prompt"]
        # For each generated sample we need:
        #   - a structured prompt (list of chat messages)
        #   - a structured completion (list with one assistant message)
        #   - one entry per dataset column (e.g. "answer", "target", "numbers")
        prompts_for_rewards = []
        completions_for_rewards = []
        reward_kwargs = {key: [] for key in data_keys}

        for row_idx in self._last_prompt_indices:
            row = self.train_set[row_idx]
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
            rewards_per_func[:, j] = torch.tensor(scores, device=device, dtype=torch.float32).clamp(min=0.0)
        
        #DEBUG: for a given prompt, with prob 0.1 print all the generated completions and their rewards
        do_print = torch.rand(num_buffer_updates, device=device) < getattr(self.hparams.tm, "buffer_student_print_prob", 0.1)
        for i in range(num_buffer_updates):
            if not do_print[i]:
                continue
            
            # Get the prompt index in the dataset
            row_idx = self._last_prompt_indices[i]
            row = self.train_set[row_idx]
        
            if self.hparams.dataset == "math" or self.hparams.dataset == "gsm8k":
                prompt = row.get("prompt", None)
                prompt_text = None
                if isinstance(prompt, list) and prompt:
                    prompt_text = prompt[0].get("content", None)
                elif isinstance(prompt, str):
                    prompt_text = prompt
                if prompt_text:
                    parts = prompt_text.split("\n\n", 1)
                    question = parts[1] if len(parts) > 1 else prompt_text
                else:
                    question = None
                answer = row.get("answer", None)
                
                print(prompt_text, flush=True)
                print(f"\n{'='*80}", flush=True)
                print(f"Model = Teacher", flush=True)
                print(f"Question: {question}", flush=True)
                if self.hparams.dataset == "math":
                    boxed = self._extract_boxed_answer(answer)
                    print(f"Ground truth answer (boxed): {boxed}", flush=True)
                elif self.hparams.dataset == "gsm8k":
                    print(f"Ground truth answer: {answer}", flush=True)
            elif self.hparams.dataset == "countdown":
                target = row.get("target", None)
                numbers = row.get("numbers", None)
                print(f"Target: {target} | Numbers: {numbers}", flush=True)
            print(f"{'='*80}\n", flush=True)
            
            # Print all completions for this prompt
            for comp_idx in range(num_completions_per_prompt):
                global_idx = i * num_completions_per_prompt + comp_idx
                completion_text = completions_text[global_idx]
                
                # Extract the equation/answer from completion
                equation = None
                answer_pattern = r"<answer>(.*?)</answer>"
                matches = re.findall(answer_pattern, completion_text, re.DOTALL)
                if matches:
                    equation = matches[-1].strip()
                
                # Get the reward for this completion
                if self.hparams.dataset == "gsm8k":
                    reward_val = rewards_per_func[global_idx, -1].item() if num_funcs > 0 else 0.0
                else:
                    reward_val = rewards_per_func[global_idx, 0].item() if num_funcs > 0 else 0.0
                
                print(f"  Completion {comp_idx + 1}/{num_completions_per_prompt}:", flush=True)
                print(f"  --------------------------------", flush=True)
                print(f"  Extracted equation: {equation}", flush=True)
                print(f"  Reward: {reward_val:.4f}", flush=True)
                print(f"  Full completion: {completion_text}", flush=True)  # truncate long completions
                print()


        # Store as shape [num_buffer_updates, num_completions_per_prompt, num_funcs]
        new_rewards_block = rewards_per_func.view(num_buffer_updates, -1, num_funcs)
        if self.buffer_rewards is None:
            self.buffer_rewards = new_rewards_block
        else:
            self.buffer_rewards[update_rows, :, :] = new_rewards_block
        
        avg_rwd = float(new_rewards_block.mean() * new_rewards_block.shape[-1])

        print(f"[EVAL] average reward = {avg_rwd:.3f}")
        
        if getattr(self, "_rebuild_buffer_next_phase", False) and getattr(self.hparams.tm, "rwd_shift_auto", True):
            self.rwd_shift = - self.all_gather(torch.tensor(avg_rwd, device=self.device)).mean().item()
            print(f"New phase: setting rwd_shift = {self.rwd_shift:.3f}")

        buffer_end_time = datetime.now()
        buffer_build_time = (buffer_end_time - buffer_start_time).total_seconds()
        print(f"Finished {build_or_refresh} reward buffer, took {buffer_build_time}")

        # restore adapter
        model.set_adapter(prev_adapter)
        model.train()

    def logging_student(self, model, num_completions_per_prompt):
        """
        Evaluate the student model on the validation set.
        - Distributes validation examples across GPUs
        - Each GPU evaluates a subset of the validation set
        - Generates completions and computes rewards
        - Each prompt is completed only once (num_completions_per_prompt=1)
        """
        device = self.device

        prev_adapter = model.active_adapter
        model.set_adapter("student")
        model.eval()
        print(f"Start Logging Student ...")
        buffer_start_time = datetime.now()

        # ---- 1. Distribute validation set across GPUs ----
        global_world_size = getattr(self.trainer, "world_size", 1)
        global_rank = getattr(self.trainer, "global_rank", 0)
        
        total_val_examples = len(self.validation_set)
        examples_per_gpu = total_val_examples // global_world_size
        start_idx = global_rank * examples_per_gpu
        
        # Last GPU takes any remainder
        if global_rank == global_world_size - 1:
            end_idx = total_val_examples
        else:
            end_idx = start_idx + examples_per_gpu
        
        # Get this GPU's subset of validation examples
        val_subset_indices = list(range(start_idx, end_idx))
        num_val_prompts = len(val_subset_indices)
        print(f"[GPU {global_rank}/{global_world_size}] Evaluating {num_val_prompts} validation examples (indices {start_idx} to {end_idx-1})")
        
        # ---- 2. Prepare prompts from validation set ----
        if self.hparams.dataset == "countdown":
            structured_prompts = []
            targets = []
            numbers_list = []
        
            for idx in val_subset_indices:
                val_entry = self.validation_set[idx]
                target = int(val_entry["output"])
                numbers_str = val_entry["input"]
                numbers = [int(num) for num in numbers_str.split(",")]
                
                # Create structured prompt matching countdown.py format
                prompt_text = f"{SYSTEM_PROMPT}\nUsing only the numbers {numbers}, create an arithmetic expression that evaluates to exactly {target}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
                structured_prompts.append([{"role": "user", "content": prompt_text}])
                targets.append(target)
                numbers_list.append(numbers)
        else:
            structured_prompts = [self.validation_set[i]["prompt"] for i in val_subset_indices]
        
        prompts_text = [self.tokenizer.apply_chat_template(sp, tokenize=False, add_generation_prompt=True) for sp in structured_prompts]
        prompt_ids = self.tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
            add_special_tokens=False,
        )["input_ids"].to(device)
        
        # Repeat each prompt for num_completions_per_prompt completions
        prompt_ids = prompt_ids.repeat_interleave(num_completions_per_prompt, dim=0)
        total_batch, prompt_len = prompt_ids.shape

        # ---- 3. Run diffusion generation to get prompt+completion sequences ----
        chunk_size = max(1, min(self.hparams.tm.buffer_chunk_size, total_batch)//4)
        gen_length = self.hparams.max_completion_length
        seq_len = prompt_len + gen_length
        # pre-allocate
        prompt_completion_ids = torch.empty(
            (total_batch, seq_len),
            device=prompt_ids.device,
            dtype=prompt_ids.dtype,
        )
        for start in range(0, total_batch, chunk_size):
            end = min(start + chunk_size, total_batch)
            with torch.no_grad():
                chunk_completion_ids = self._generate(
                    model=model,
                    prompt=prompt_ids[start:end],
                    steps=self.hparams.diffusion_steps,
                    gen_length=gen_length,
                    block_length=self.hparams.block_length,
                    temperature=self.log_temperature,
                    cfg_scale=self.hparams.cfg_scale,
                    remasking='low_confidence',
                )  # [end-start, seq_len]

            prompt_completion_ids[start:end].copy_(chunk_completion_ids)
            del chunk_completion_ids
        
        # ---- 4. Decode completions to text for reward computation ----
        completion_ids = prompt_completion_ids[:, prompt_len:]  # [total_batch, gen_length]
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # ---- 5. Build reward inputs: prompts, completions, and extra dataset columns ----
        data_keys = [key for key in self.train_set[0].keys() if key != "prompt"]
        if self.hparams.dataset == "countdown":
            reward_kwargs = {"target": [], "numbers": []}
            prompts_for_rewards = structured_prompts
            for i in range(num_val_prompts):
                # Add target and numbers for reward computation
                reward_kwargs["target"].append(targets[i])
                reward_kwargs["numbers"].append(numbers_list[i])
        else:
            reward_kwargs = {key: [] for key in data_keys}
            prompts_for_rewards = [self.validation_set[i]["prompt"] for i in val_subset_indices]
            for row_idx in val_subset_indices:
                # Copy all extra fields for this completion
                for key in data_keys:
                    reward_kwargs[key].append(self.validation_set[row_idx][key])

        # Turn plain completions into chat-style completions [{"role": "assistant", "content": "..."}]
        completions_for_rewards = [[{"role": "assistant", "content": text}] for text in completions_text]

        # ---- 6. Compute rewards for every generated completion ----
        num_funcs = len(self.reward_funcs)
        rewards_per_func = torch.zeros(total_batch, num_funcs, device=device)
        reward_kwargs["buffer_print_samples"] = int(getattr(self.hparams.tm, "buffer_print_samples", 0))
        reward_kwargs["rank"] = int(getattr(self.trainer, "global_rank", 0))

        for j, reward_func in enumerate(self.reward_funcs):
            scores = reward_func(
                prompts=prompts_for_rewards,
                completions=completions_for_rewards,
                **reward_kwargs,
            )
            rewards_per_func[:, j] = torch.tensor(scores, device=device, dtype=torch.float32)

        do_print = torch.rand(num_val_prompts, device=device) < getattr(self.hparams.tm, "buffer_student_print_prob", 0.1)
        for i in range(num_val_prompts):
            if not do_print[i]:
                continue
            
            row_idx = val_subset_indices[i]
            row = self.validation_set[row_idx]

            print(f"\n{'='*80}", flush=True)
            print(f"Model = Student", flush=True)
            if self.hparams.dataset == "math" or self.hparams.dataset == "gsm8k":
                prompt = row.get("prompt", None)
                prompt_text = None
                if isinstance(prompt, list) and prompt:
                    prompt_text = prompt[0].get("content", None)
                elif isinstance(prompt, str):
                    prompt_text = prompt
                if prompt_text:
                    parts = prompt_text.split("\n\n", 1)
                    question = parts[1] if len(parts) > 1 else prompt_text
                else:
                    question = None
                answer = row.get("answer", None)
                
                print(f"Question: {question}", flush=True)
                if self.hparams.dataset == "math":
                    boxed = self._extract_boxed_answer(answer)
                    print(f"Ground truth answer (boxed): {boxed}", flush=True)
                elif self.hparams.dataset == "gsm8k":
                    print(f"Ground truth answer: {answer}", flush=True)
            elif self.hparams.dataset == "countdown":
                target = row.get("target", None)
                numbers = row.get("numbers", None)
                print(f"Target: {target} | Numbers: {numbers}", flush=True)
            print(f"{'='*80}\n", flush=True)    

            matches = re.findall(r"<answer>(.*?)</answer>", completions_text[i], re.DOTALL)
            if matches:
                equation = matches[-1].strip()
            else:
                equation = None
            

            if self.hparams.dataset == "gsm8k":
                reward_val = rewards_per_func[i, -1].item() if num_funcs > 0 else 0.0
            else:
                reward_val = rewards_per_func[i, 0].item() if num_funcs > 0 else 0.0

            print(f"  --------------------------------", flush=True)
            print(f"  Extracted equation: {equation}", flush=True)
            print(f"  Reward: {reward_val:.4f}", flush=True)
            print(f"  Full completion: {completions_text[i]}", flush=True)  # truncate long completions
            print()

        # Free memory after reward computation
        del prompts_for_rewards
        del completions_for_rewards
        del reward_kwargs

        if self.hparams.dataset == "gsm8k":
            correct_nums_eval = torch.isclose(rewards_per_func[:, -1], 2.0 * torch.ones_like(rewards_per_func[:, -1]), atol=1e-3, rtol=0.0).float().sum().item()
            format_eval = 0 # TODO: add
        elif self.hparams.dataset == "math":
            correct_nums_eval = torch.isclose(rewards_per_func[:, 0], 2.0 * torch.ones_like(rewards_per_func[:, 0]), atol=1e-3, rtol=0.0).float().sum().item()
            format_eval = torch.isclose(rewards_per_func[:, 1], 0.375 * torch.ones_like(rewards_per_func[:, 1]), atol=1e-3, rtol=0.0).float().sum().item()  
        else:
            correct_nums_eval = torch.isclose(rewards_per_func, self.hparams.max_rwd * torch.ones_like(rewards_per_func), atol=1e-3, rtol=0.0).float().sum().item()
            format_eval = 0 # TODO: add
        total_rwd_eval = rewards_per_func.sum(dim=-1).sum().item()
        local_counts = torch.tensor([correct_nums_eval, format_eval, total_rwd_eval, num_val_prompts * num_completions_per_prompt], device=self.device, dtype=torch.long)
        gathered = self.all_gather(local_counts)  # shape: (world_size, 4) for DDP
        global_correct = gathered[:, 0].sum()
        global_format = gathered[:, 1].sum()
        global_rwd = gathered[:, 2].sum()
        global_total = gathered[:, 3].sum()
        
        global_format_score = (global_format.float() / global_total.float())
        global_acc = (global_correct.float() / global_total.float())
        global_rwd_avg = (global_rwd.float() / global_total.float())

        print(f"[EVAL] Global accuracy is {global_acc:.4f}, global average reward is {global_rwd_avg:.4f}, global format score is {global_format_score:.4f}")
        self.dict_for_logs["eval/format_score"] = global_format_score.item()
        self.dict_for_logs["eval/correct_frac"] = global_acc.item()
        self.dict_for_logs["eval/avg_rwd"] = global_rwd_avg.item()
        # print(f"[EVAL] At global step {self.global_step}, for gpu {self.global_rank}, student correctness fraction: {global_acc:.4f}, avg reward: {global_rwd_avg:.4f}")
        
        buffer_end_time = datetime.now()
        buffer_build_time = (buffer_end_time - buffer_start_time).total_seconds()
        print(f"Logging Student finished, took {buffer_build_time}")

        # restore adapter
        model.set_adapter(prev_adapter)
        model.train()

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
        log_A = F.log_softmax(logits_A, dim=-1)
        log_B = F.log_softmax(logits_B, dim=-1)
        kl = F.kl_div(log_A, log_B, reduction='none', log_target=True).sum(-1)
        return kl[mask_indices].float().mean()

    def _build_loss_weights(self, completion_ids, mask_indices, active_block_mask, block_size, dtype):
        use_sar = bool(getattr(self.hparams.tm, "use_sar_active_block_norm", False))
        aux_mask = active_block_mask if use_sar else mask_indices
        pre_eos_mask = self._build_pre_eos_mask(completion_ids)

        if not (use_sar and self._use_sar_discounted_future_loss):
            return aux_mask, aux_mask.to(dtype=dtype) * pre_eos_mask.to(dtype=dtype)

        if not torch.all(active_block_mask.any(dim=1)):
            raise RuntimeError("Expected every sample to have at least one masked token in the SAR active block.")

        gen_length = mask_indices.shape[1]
        if gen_length % block_size != 0:
            raise ValueError("SAR discounted future loss requires the generation length to be divisible by block_size.")

        device = mask_indices.device
        block_ids = torch.arange(gen_length, device=device, dtype=torch.long) // block_size  # [L]
        active_block_ids = (active_block_mask.to(torch.long) * block_ids.unsqueeze(0)).amax(dim=1)  # [B]
        block_offsets = block_ids.unsqueeze(0) - active_block_ids.unsqueeze(1)  # [B, L]
        masked_offsets = torch.where(mask_indices, block_offsets, torch.zeros_like(block_offsets))

        alpha = torch.tensor(self._sar_future_discount_alpha, device=device, dtype=dtype)
        loss_weights = mask_indices.to(dtype) * torch.pow(alpha, masked_offsets.to(dtype)) * pre_eos_mask.to(dtype)
        return aux_mask, loss_weights

    def _build_pre_eos_mask(self, completion_ids):
        eos_hits = completion_ids.eq(self._eos_id)
        seen_eos_before = eos_hits.to(torch.int32).cumsum(dim=1) - eos_hits.to(torch.int32)
        return seen_eos_before == 0

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
            mask_indices: [B, gen_length] BoolTensor, True where tokens are masked.
            partial_to_mask: [B, gen_length] BoolTensor, True where tokens are masked in the partial block.
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

        return xts, mask_indices.bool(), partial_to_mask.bool()

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
