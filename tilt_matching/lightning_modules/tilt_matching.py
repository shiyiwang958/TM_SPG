import copy
import logging
import math
import os
from datetime import datetime
from collections import OrderedDict, namedtuple, Counter
from contextlib import contextmanager, nullcontext
import itertools
import wandb
import re
import bitsandbytes as bnb
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, get_peft_model_state_dict, set_peft_model_state_dict
from data_utils import ID_TO_TYPE
from generate import generate

class TiltMatchingModule(pl.LightningModule):
    def __init__(self, base_model, tokenizer, training_prompts_dataset, test_prompts_dataset, reward_funcs, **cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["base_model", "tokenizer", "training_prompts_dataset", "test_prompts_dataset", "reward_funcs"], logger=False)
        self.tokenizer = tokenizer

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

        self.curr_prompt_counter = 0
        self.training_prompts_dataset = training_prompts_dataset
        self.training_prompts_dataset_len = len(self.training_prompts_dataset)
        self.test_prompts_dataset = test_prompts_dataset
        print(f"[DEBUG] Training prompts dataset length: {self.training_prompts_dataset_len}")
        self.reward_funcs = reward_funcs
        self.reward_weights = None

        self.a = 0.0
        self.h = self.hparams.tm.h
        self.steps_per_h = self.hparams.tm.steps_per_h
        self.a_end = self.hparams.tm.a_end
        self.mask_id = 126336
        self.checkpoint_freq = self.hparams.checkpoint_freq
        self._step_counter = 0
        self.cv = self.hparams.tm.control_variate
        self._start_step = 0
        # Control-variate (c) online estimation buffers (accumulated over grad_accum micro-steps)
        self._cv_num_accum = None  # sum of w*<pi_theta-delta, delta-pi_a> over masked positions
        self._cv_den_accum = None  # sum of ||delta-pi_a||^2 over masked positions
        self._cv_ema_beta = float(getattr(self.hparams.tm, "control_variate_ema", 0.05))
        self.rwd_shift = float(getattr(self.hparams.tm, "rwd_shift", 0.0))
        self.buffer = None
        self.buffer_rewards = None
        self.level_and_type = None
        self._rebuild_buffer_next_phase = False
        self.num_buffer_prompts = self.hparams.tm.num_buffer_prompts
        self.comps_per_prompt = self.hparams.tm.num_completions_per_prompt
        self.buffer_update_counter = 0
        self.ckpt_counter = 0
        self._recent_buffer_rwd = []
        self._grad_accum_counter = 0
        # --- Per h-phase total-reward distribution stats (accumulated across _update_buffer calls) ---
        self._phase_total_reward_counts = Counter()  # maps total_reward_value -> count
        self._phase_total_reward_n = 0               # total number of samples counted this phase
        self.dict_for_logs = {}
        # micro-step metric accumulation
        self._micro_log_sums = {}
        self._micro_log_counts = {}
        self._micro_log_mins = {}
        self._micro_log_maxs = {}

        self.lr = self.hparams.learning_rate
        self.lr_scheduler_type = self.hparams.lr_scheduler_type
        self.lr_decay_ratio = self.hparams.lr_decay_ratio
        self.lr_warmup_ratio = getattr(self.hparams, "lr_warmup_ratio", 0)
        self.lr_min = getattr(self.hparams, "lr_min", 0.0)
        self._tm_sched_state = None

        # --- Student LoRA EMA (used for eval_student and teacher sync) ---
        self.student_ema_enabled = bool(getattr(self.hparams.tm, "use_student_ema", True))
        self.student_ema_beta = float(getattr(self.hparams.tm, "student_ema_beta", 0.99))
        self.student_ema_start_step = int(getattr(self.hparams.tm, "student_ema_start_step", 0))
        self._student_ema_state = None  # OrderedDict[str, Tensor] (student adapter EMA)
        self._student_ema_loaded_cpu = None  # temp storage when resuming from ckpt

    @contextmanager
    def _use_adapter(self, adapter_name: str):
        prev = self.model.active_adapter
        self.model.set_adapter(adapter_name)
        try:
            yield
        finally:
            self.model.set_adapter(prev)

    def _clone_adapter_state(self, adapter_name: str) -> OrderedDict:
        """Clone the PEFT adapter state dict so it can be safely restored after swaps."""
        sd = get_peft_model_state_dict(self.model, adapter_name=adapter_name)
        return OrderedDict((k, v.detach().clone()) for k, v in sd.items())

    def _init_student_ema(self) -> None:
        self._student_ema_state = self._clone_adapter_state("student")

    def _reset_student_ema(self) -> None:
        """Start EMA fresh from the *current* student adapter weights.

        We call this at h-phase boundaries so the next phase's EMA does not mix
        weights from the previous phase.
        """
        if not self.student_ema_enabled:
            return
        self._student_ema_loaded_cpu = None
        self._student_ema_state = None
        self._init_student_ema()

    def _maybe_init_student_ema(self) -> None:
        if not self.student_ema_enabled:
            return
        if self._student_ema_state is not None:
            return
        if self._student_ema_loaded_cpu is not None:
            # Loaded from ckpt on CPU; move to the current device
            self._student_ema_state = OrderedDict((k, t.to(self.device)) for k, t in self._student_ema_loaded_cpu.items())
            self._student_ema_loaded_cpu = None
        else:
            self._init_student_ema()

    @torch.no_grad()
    def _update_student_ema(self) -> None:
        """Update EMA of *student* LoRA adapter weights (in-place).

        Called once per optimizer/global step *after* opt.step().
        """
        if not self.student_ema_enabled:
            return
        if int(self.global_step) < int(self.student_ema_start_step):
            return

        # lazy init
        if self._student_ema_state is None:
            self._maybe_init_student_ema()
            return

        beta = float(self.student_ema_beta)
        cur = get_peft_model_state_dict(self.model, adapter_name="student")
        for k, v in cur.items():
            if k not in self._student_ema_state:
                self._student_ema_state[k] = v.detach().clone()
            else:
                self._student_ema_state[k].mul_(beta).add_(v.detach(), alpha=(1.0 - beta))

    @contextmanager
    def _use_student_ema_weights(self):
        """Temporarily swap student adapter weights to their EMA version."""
        if not self.student_ema_enabled:
            yield
            return

        if self._student_ema_state is None:
            self._maybe_init_student_ema()
        if self._student_ema_state is None:
            yield
            return

        prev_student = self._clone_adapter_state("student")
        try:
            set_peft_model_state_dict(self.model, self._student_ema_state, adapter_name="student")
            yield
        finally:
            set_peft_model_state_dict(self.model, prev_student, adapter_name="student")


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
        Returns a dict mirroring torch's load_state_dict with any missing or unexpected keys.
        """
        expected_student = set(get_peft_model_state_dict(self.model, adapter_name="student").keys())
        expected_teacher = set(get_peft_model_state_dict(self.model, adapter_name="teacher").keys())

        student_state = OrderedDict()
        teacher_state = OrderedDict()
        unexpected_keys = []

        for key, value in state_dict.items():
            if key.startswith("model_adapter."):
                bare = key[len("model_adapter."):]
                student_state[bare] = value.to(self.model.device)
            elif key.startswith("base_adapter."):
                bare = key[len("base_adapter."):]
                teacher_state[bare] = value.to(self.model.device)
            else:
                unexpected_keys.append(key)

        set_peft_model_state_dict(self.model, student_state, adapter_name="student")
        set_peft_model_state_dict(self.model, teacher_state, adapter_name="teacher")

        missing_student = list(expected_student - set(student_state.keys()))
        missing_teacher = list(expected_teacher - set(teacher_state.keys()))
        missing_keys = [f"model_adapter.{k}" for k in missing_student] + [f"base_adapter.{k}" for k in missing_teacher]

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Error(s) in loading state_dict: missing keys {missing_keys}; unexpected keys {unexpected_keys}"
            )

        return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

    def on_train_start(self):
        super().on_train_start()

        self.tm_opt = self.optimizers()

        resuming = bool(getattr(self, "_resuming_from_ckpt", False))

        if resuming:
            self._start_step = self.global_step

        self._grad_accum_counter = 0

        if self.hparams.dataset == "math":
            self._init_math_lat_dict()

        self._do_first_eval = getattr(self.hparams.tm, "do_first_eval", False)
        if self._do_first_eval:
            rwds_eval = self._eval_student(self.hparams.tm.eval_num_prompts) # [N_eval, num_reward_funcs]
            if self.hparams.dataset == "gsm8k":
                correct_frac_eval = torch.isclose(rwds_eval[:, -1], 2.0 * torch.ones_like(rwds_eval[:, -1]), atol=1e-6, rtol=0.0).float().mean()
                avg_rwd_eval = rwds_eval.sum(dim=-1).mean() # [N_eval,]
                self.dict_for_logs["eval/correct_frac"] = correct_frac_eval
                self.dict_for_logs["eval/avg_rwd"] = avg_rwd_eval
                print(f"[EVAL] At global step {self.global_step}, for gpu {self.global_rank}, student correctness fraction: {correct_frac_eval:.4f}, avg reward: {avg_rwd_eval:.4f}")
            if self.hparams.dataset == "math":
                correct_frac_eval = torch.isclose(rwds_eval[:, 0], 2.0 * torch.ones_like(rwds_eval[:, 0]), atol=1e-6, rtol=0.0).float().mean()
                avg_rwd_eval = rwds_eval.sum(dim=-1).mean() # [N_eval,]
                self.dict_for_logs["eval/correct_frac"] = correct_frac_eval
                self.dict_for_logs["eval/avg_rwd"] = avg_rwd_eval
                print(f"[EVAL] At global step {self.global_step}, for gpu {self.global_rank}, student correctness fraction: {correct_frac_eval:.4f}, avg reward: {avg_rwd_eval:.4f}")

        if not resuming:
            # fresh run behavior
            self._grad_accum_counter = 0
            for g in self.tm_opt.param_groups:
                g["lr"] = self.lr
            self._init_tm_scheduler()
            self._maybe_init_student_ema()
            return

        # ----- resume behavior -----
        # DO NOT overwrite optimizer lr (Lightning already restored it)

        # If scheduler state wasn't in the ckpt (older ckpts), reconstruct it without changing LR:
        if self._tm_sched_state is None:
            saved_lrs = [pg["lr"] for pg in self.tm_opt.param_groups]
            self._init_tm_scheduler()  # this would set lrs=0; fix below
            for pg, lr in zip(self.tm_opt.param_groups, saved_lrs):
                pg["lr"] = lr

            # best-effort: infer position inside h-phase from global_step
            phase_step = int((self.global_step - self._start_step) % self.steps_per_h)
            if self._tm_sched_state is not None:
                self._tm_sched_state["step"] = phase_step
        if getattr(self.hparams, "use_fresh_lr", False):
            for g in self.tm_opt.param_groups:
                g["lr"] = self.lr
            self._init_tm_scheduler()

        self._maybe_init_student_ema()
    
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
        if self._do_first_eval:
            self.log_dict(self.dict_for_logs, on_step=True, on_epoch=False, sync_dist=True)
            self._do_first_eval = False
        if self.buffer is None:
            self._update_buffer(self.model, self.num_buffer_prompts, self.comps_per_prompt)
            print(f"[DEBUG] Buffer initialized with shape {self.buffer.shape}")
            print(self.curr_prompt_counter)
        # If we scheduled a full buffer rebuild at the end of the previous h-phase,
        # do it now (i.e., at the start of the new h-phase), after checkpointing.
        if getattr(self, "_rebuild_buffer_next_phase", False):
            self._update_buffer(self.model, self.num_buffer_prompts, self.comps_per_prompt)
            print(f"[DEBUG] Buffer built at start of new h-phase (global step {self.global_step}) with shape {self.buffer.shape}")
            self._rebuild_buffer_next_phase = False
        # print("on_train_batch_start complete")
    
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
            out[k] = s / float(c)
        out.update(self._micro_log_mins)
        out.update(self._micro_log_maxs)
        return out
    
    def save_checkpoint_now(self, name: str = None) -> str:
        """
        Explicit checkpoint save you can call whenever you want.

        - Writes only on global rank 0
        - Barriers so other ranks don't race ahead
        """

        # Make sure all ranks line up here
        if hasattr(self.trainer, "strategy") and self.trainer.strategy is not None:
            self.trainer.strategy.barrier()

        if not self.trainer.is_global_zero:
            # Non-zero ranks never write files
            return ""

        ckpt_dir = getattr(self.hparams, "checkpoint_dir", None)
        if ckpt_dir is None:
            raise RuntimeError("cfg.checkpoint_dir was not found in hparams; can't pick a save directory.")

        if name is None:
            name = f"manual-a-{self.a:.3f}-gs{int(self.global_step)}-micro{int(getattr(self, '_grad_accum_counter', 0))}"

        path = os.path.join(ckpt_dir, f"{name}.ckpt")
        self.trainer.save_checkpoint(path)

        if hasattr(self.trainer, "strategy") and self.trainer.strategy is not None:
            self.trainer.strategy.barrier()

        return path


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
            batch: Dummy batch (not used).
            batch_idx: Index of the batch (not used).
        """
        # if dist.is_initialized():
        #     assert dist.get_world_size() == 8, f"world_size={dist.get_world_size()} (expected 8)"
        opt = self.tm_opt
        accum = self.hparams.tm.grad_accum_steps

        # At the start of a new accumulation window:
        if (self._grad_accum_counter % accum) == 0:
            # randomly choose the prompts to use for the microsteps
            total_prompts_needed = self.hparams.tm.num_batch_prompts * accum
            self._accum_prompts_idx = torch.randperm(self.buffer.shape[0], device=self.device)[:total_prompts_needed]
            self._reset_micro_log_accum()
            # Reset control-variate accumulation for this global step
            self._cv_num_accum = torch.zeros((), device=self.device)
            self._cv_den_accum = torch.zeros((), device=self.device)

        loss, micro_log_dict = self._tm_step()
        self._accumulate_micro_log_dict(micro_log_dict)
        loss_scaled = loss / float(accum)

        # Backward (avoid DDP grad sync on non-update micro-steps)
        self._grad_accum_counter += 1
        is_update_step = (self._grad_accum_counter % accum) == 0
        if not is_update_step:
            with self.trainer.model.no_sync():
                self.manual_backward(loss_scaled)
        else:
            self.manual_backward(loss_scaled)

        # Only update weights / schedules on the last micro-step of the window
        if not is_update_step:
            # Prevent logging hooks from trying to log multiple micro-steps at the same global_step
            self.dict_for_logs = {}
            return loss

        # ---- Optimizer / scheduler step ----
        self._step_tm_scheduler()

        # Gradient clipping (on accumulated grads)
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm_before = clip_grad_norm_(params, self.hparams.max_grad_norm).item()
        grad_norm_after = clip_grad_norm_(params, float("inf")).item()
        grad_clipped = float(grad_norm_before > self.hparams.max_grad_norm + 1e-6)

        opt.step()
        opt.zero_grad(set_to_none=True)
        self._update_student_ema()

        # ---- Update control variate once per global step (after grad accumulation, synced across GPUs) ----
        # # Aggregate across all GPUs
        # if dist.is_available() and dist.is_initialized():
        #     dist.all_reduce(self._cv_num_accum, op=dist.ReduceOp.SUM)
        #     dist.all_reduce(self._cv_den_accum, op=dist.ReduceOp.SUM)
        # else:
        #     print("[WARNING] dist not available or not initialized for cv aggregation")

        # Compute c_batch and EMA update (identical on all ranks)
        c_batch = (-self._cv_num_accum / self._cv_den_accum.clamp_min(1e-12))

        cv_old = torch.tensor(float(self.cv), device=self.device)
        cv_new = (1.0 - self._cv_ema_beta) * cv_old + self._cv_ema_beta * c_batch
        self.cv = float(cv_new.item())

        # if (self.global_step + 2) % self.steps_per_h < 5:
        #     print(f"current a is {self.a:.4f}")
        #     print(f"global step is {self.global_step}")

        # Build averaged metrics (over micro-steps) for this optimizer/global step
        self.dict_for_logs = self._finalize_micro_log_dict()

        # Log current learning rate and grad norms
        self.dict_for_logs["train/lr"] = opt.param_groups[0]["lr"]
        self.dict_for_logs["grads/grad_norm_before"] = grad_norm_before
        self.dict_for_logs["grads/grad_norm_after"] = grad_norm_after
        self.dict_for_logs["grads/grad_clipped"] = grad_clipped
        self.dict_for_logs["train/cv"] = self.cv

        if (self.global_step - self._start_step) % self.steps_per_h == 0:
            # ---- END-OF-PHASE: print distribution of total rewards observed during this phase ----
            # self._cv_assum_one_hot_sum = None  # reset for next phase
            if self._phase_total_reward_n > 0:
                items = sorted(self._phase_total_reward_counts.items(), key=lambda kv: kv[0])
                parts = []
                for r, c in items:
                    pct = 100.0 * c / self._phase_total_reward_n
                    parts.append(f"{r:g}: {pct:.2f}% ({c}/{self._phase_total_reward_n})")
                print(
                    f"[REWARD DIST] End of h-phase at a={self.a:.4f}, for gpu {getattr(self.trainer, 'global_rank', 0)}, "
                    f"samples={self._phase_total_reward_n} -> " + ", ".join(parts),
                    flush=True,
                )
            # Reset stats for the next h-phase
            self._phase_total_reward_counts.clear()
            self._phase_total_reward_n = 0
            # ---- Update tilt parameter a and teacher adapter ----
            self.a += self.h
            if self.a + self.h > self.a_end:
                self.h = self.a_end - self.a
            with torch.no_grad():
                # Copy *EMA* student LoRA weights into the teacher at phase boundaries
                if self.student_ema_enabled and (self._student_ema_state is not None):
                    adapter_state = self._student_ema_state
                else:
                    adapter_state = get_peft_model_state_dict(self.model, adapter_name="student")
                set_peft_model_state_dict(self.model, adapter_state, adapter_name="teacher")
                for name, p in self.model.named_parameters():
                    if ".teacher" in name:
                        p.requires_grad_(False)
            print(f"Model weights copied. Degree of tilt a = {self.a:.4f} at global step {self.global_step}")

            if self.a >= self.a_end:
                print(f"Reached final a = {self.a_end:.2f}. Training Stopped", flush=True)
                self.trainer.should_stop = True

            # Reset LR for the new h phase
            for g in opt.param_groups:
                g["lr"] = self.lr
            self._init_tm_scheduler()

            # Start student EMA fresh for the new h-phase (do not mix across phases)
            self._reset_student_ema()

            if self.hparams.dataset == "math":
                self._init_math_lat_dict()

        self.log("ckpt_a", self.a, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def _tm_step(self):
        num_buffer_prompts, comps_per_prompt, L = self.buffer.shape
        num_batch_prompts = self.hparams.tm.num_batch_prompts
        B = num_batch_prompts * comps_per_prompt
        gen_length = self.hparams.max_completion_length

        # Draw a batch from the buffer
        start_idx = (self._grad_accum_counter % self.hparams.tm.grad_accum_steps) * num_batch_prompts
        end_idx = start_idx + num_batch_prompts
        prompts_idx = self._accum_prompts_idx[start_idx:end_idx]
        x1s = self.buffer[prompts_idx].reshape(B, L)           # [B, L]
        rwds = self.buffer_rewards[prompts_idx].reshape(B, -1) # [B, num_reward_funcs]
        # if self.hparams.dataset == "math":
        #     lats = self.level_and_type[prompts_idx].reshape(B, 2)  # [B, 2]
        #     counts, rwds_sums = self._lat_stats(lats, rwds)

        # Aggregate rewards from multiple functions
        if self.reward_weights is None:
            weights = torch.ones(rwds.shape[1], device=self.device, dtype=rwds.dtype)
        else:
            weights = self.reward_weights.to(device=self.device, dtype=rwds.dtype)
        rwd = torch.nansum(rwds * weights.unsqueeze(0), dim=1) # [B,]
        # % of totally correct samples
        if self.hparams.dataset == "gsm8k":
            correct_frac = torch.isclose(rwds[:, -1], 2.0 * torch.ones_like(rwd), atol=1e-6, rtol=0.0).float().mean()
        elif self.hparams.dataset == "math":
            correct_frac = torch.isclose(rwds[:, 0], 2.0 * torch.ones_like(rwd), atol=1e-6, rtol=0.0).float().mean()
        else:
            correct_frac = torch.isclose(rwd, self.hparams.max_rwd * torch.ones_like(rwd), atol=1e-6, rtol=0.0).float().mean()
        
        # Create x_t's by masking the x_1's
        num_to_mask = torch.randint(low=1, high=gen_length+1, size=(x1s.shape[0],), device=self.device)
        xts, mask_indices = self._build_interpolant(x1s, num_to_mask, self.hparams.block_length)

        # Get model predictions and compute loss
        temp = self.hparams.sampling_temperature
        self.model.eval()
        with torch.no_grad(), self._use_adapter("teacher"):
            old_logits = self._new_forward(self.model, xts, gen_length) # [B, gen_length, V]
        self.model.train()
        V = old_logits.shape[-1]
        x1_equals_v = F.one_hot(x1s.long()[:, -gen_length:], num_classes = V) # [B, gen_length, V]
        with self._use_adapter("student"):
            curr_logits = self._new_forward(self.model, xts, gen_length) # [B, gen_length, V]
        if temp > 0.0 and self.hparams.tm.rescale_logits:
            old_logits /= temp
            curr_logits /= temp
        old_probs = F.softmax(old_logits, dim=-1) # [B, gen_length, V]
        with torch.no_grad():
            curr_probs_ng = F.softmax(curr_logits, dim=-1)  # [B, gen_length, V]
        
        loss_type = self.hparams.tm.loss_type
        # shift reward for minimizing gradient variance for loss computation
        hr = self.h * (rwd + self.rwd_shift) # [B,]

        # learnable control variate accumulation
        with torch.no_grad():
            w = torch.exp(hr).view(-1, 1)  # [B,1]

            # A = delta - pi_a, B = pi_theta - delta
            A = x1_equals_v - old_probs       # [B,L,V]
            Bv = curr_probs_ng - x1_equals_v  # [B,L,V]

            dot = (Bv * A).sum(dim=-1)  # [B,L]
            den = (A * A).sum(dim=-1)   # [B,L]

            self._cv_num_accum += (w * dot)[mask_indices.bool()].sum()
            self._cv_den_accum += den[mask_indices.bool()].sum()
                
        prev_cv = self.cv
        if not self.hparams.tm.learned_cv:
            self.cv = self.hparams.tm.control_variate
        if loss_type == "itm":
            target = self.cv * old_probs + x1_equals_v * (1 - self.cv + torch.expm1(hr)).view(-1, 1, 1) # [B, gen_length, V]
        elif loss_type == "etm":
            target = (1 - hr) * old_probs + x1_equals_v * hr.view(-1, 1, 1) # [B, gen_length, V]
        elif loss_type == "sg-itm":
            target = self.cv * old_probs + x1_equals_v * (1 - self.cv + torch.expm1(hr)).view(-1, 1, 1) - torch.expm1(hr) * curr_probs_ng.detach()
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}")
        per_sample_losses = -(target * F.log_softmax(curr_logits, dim=-1)).sum(dim=-1) # [B, gen_length]
        loss = per_sample_losses[mask_indices.bool()].mean()
        self.cv = prev_cv  # restore

        true_ids = x1s[:, -gen_length:]  # [B, gen_length]
        p_true = curr_probs_ng.gather(-1, true_ids.unsqueeze(-1)).squeeze(-1)  # [B, gen_length]
        # residual to one-hot: 1 - p_true, averaged over masked positions
        residual_onehot = (1.0 - p_true)[mask_indices.bool()].mean()

        if temp > 0.0 and self.hparams.tm.rescale_logits:
            old_logits *= temp
            curr_logits *= temp
        log_dict = {
            f"train/loss": loss,
            f"train/a": self.a,
            f"train/h": self.h,
            f"train/drift_gap_kl": self._kl_from_logits(old_logits, curr_logits, mask_indices),
            f"train/rwd_max": rwd.max(),
            f"train/rwd_min": rwd.min(),
            f"train/rwd_mean": rwd.mean(),
            f"train/rwd_std": rwd.std(),
            f"train/correct_frac": correct_frac,
            f"train/residual_onehot": residual_onehot,
            f"charts/step_counter": self._step_counter,
        }

        # if self.hparams.dataset == "math":
        #     for i in range(5):
        #         count_i = counts[i, :].sum().item()
        #         log_dict[f"train/level_{i+1}_count"] = count_i
        #         if count_i > 0:
        #             rwds_sums_i = rwds_sums[:, i, :].sum(dim=-1) # [2,]
        #             log_dict[f"train/level_{i+1}_correct_frac"] = rwds_sums_i[0].item() / count_i
        #             log_dict[f"train/level_{i+1}_rwd_mean"] = rwds_sums_i.sum().item() / count_i
        #         else:
        #             log_dict[f"train/level_{i+1}_correct_frac"] = -1
        #             log_dict[f"train/level_{i+1}_rwd_mean"] = -1
        #     for j in range(7):
        #         count_j = counts[:, j].sum().item()
        #         log_dict[f"train/{ID_TO_TYPE[j]}_count"] = count_j
        #         if count_j > 0:
        #             rwds_sums_j = rwds_sums[:, :, j].sum(dim=-1) # [2,]
        #             log_dict[f"train/{ID_TO_TYPE[j]}_correct_frac"] = rwds_sums_j[0].item() / count_j
        #             log_dict[f"train/{ID_TO_TYPE[j]}_rwd_mean"] = rwds_sums_j.sum().item() / count_j
        #         else:
        #             log_dict[f"train/{ID_TO_TYPE[j]}_correct_frac"] = -1
        #             log_dict[f"train/{ID_TO_TYPE[j]}_rwd_mean"] = -1
        
        # if self._grad_accum_counter % self.hparams.tm.grad_accum_steps == 0 and self.global_step % self.hparams.tm.buffer_refresh_steps == 0:
        #     self._cv_assum_one_hot_sum = None

        # temp_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        # if getattr(self, "_cv_assum_one_hot_sum", None) is None:
        #     self._cv_assum_one_hot_sum = torch.zeros_like(old_logits[0, 0, :], device=self.device, dtype=old_logits.dtype)  # [V]
        #     self._cv_assum_pi_a_sum = torch.zeros((len(temp_list), old_logits.shape[-1]), device=self.device, dtype=old_logits.dtype)  # [len(temp_list), V]
        #     self._cv_assum_count = torch.zeros((), device=self.device, dtype=old_logits.dtype)
        # cv_assum_one_hot_batch_sum = (x1_equals_v * mask_indices.unsqueeze(-1)).sum(dim=(0, 1))  # [V]
        # cv_assum_count = mask_indices.sum()
        # self._cv_assum_one_hot_sum += cv_assum_one_hot_batch_sum # [V]
        # self._cv_assum_count += cv_assum_count
        # cv_assum_one_hot_batch_sum = cv_assum_one_hot_batch_sum / cv_assum_count
        # for j, t in enumerate(temp_list):
        #     old_logits_test = old_logits / t
        #     old_probs_test = F.softmax(old_logits_test, dim=-1).clamp_min(1e-9) # [B, gen_length, V]
        #     cv_assum_pi_a_batch_sum = (old_probs_test * mask_indices.unsqueeze(-1)).sum(dim=(0, 1))  # [V]
        #     self._cv_assum_pi_a_sum[j, :] += cv_assum_pi_a_batch_sum
        #     kl_masks = cv_assum_one_hot_batch_sum > 0
        #     cv_assum_pi_a_batch_sum = cv_assum_pi_a_batch_sum / cv_assum_count
        #     batch_kl = torch.sum(cv_assum_one_hot_batch_sum[kl_masks] * (torch.log(cv_assum_one_hot_batch_sum[kl_masks]) - torch.log(cv_assum_pi_a_batch_sum[kl_masks])))
        #     log_dict[f"train/cv_assum_batch_kl_t{t}"] = batch_kl.item()
        #     accum_kl_masks = self._cv_assum_one_hot_sum > 0
        #     p_ = self._cv_assum_one_hot_sum / self._cv_assum_count
        #     q_ = (self._cv_assum_pi_a_sum[j, :] / self._cv_assum_count).clamp_min(1e-9)
        #     accum_kl = torch.sum(p_[accum_kl_masks] * (torch.log(p_[accum_kl_masks]) - torch.log(q_[accum_kl_masks])))
        #     log_dict[f"train/cv_assum_accum_kl_t{t}"] = accum_kl.item()

        if self.hparams.dataset == "gsm8k":
            rwd_names_lst = ["xml", "soft_format", "strict_format", "int", "correctness"]
            for j, rwd_name in enumerate(rwd_names_lst):
                rwd_j = rwds[:, j]
                log_dict[f"train/{rwd_name}_rwd_max"] = rwd_j.max()
                log_dict[f"train/{rwd_name}_rwd_min"] = rwd_j.min()
                log_dict[f"train/{rwd_name}_rwd_mean"] = rwd_j.mean()
        elif self.hparams.dataset == "math":
            log_dict[f"train/format_max"] = rwds[:, 1].max()
            log_dict[f"train/format_min"] = rwds[:, 1].min()
            log_dict[f"train/format_mean"] = rwds[:, 1].mean()
        
        return loss, log_dict
    
    def _lat_stats(self, lats, rwds):
        b0 = lats.shape[0]
        levels = lats[:, 0].long()
        types  = lats[:, 1].long()

        num_levels, num_types = 6, 7
        bad = (levels < 0) | (levels >= num_levels) | (types < 0) | (types >= num_types)
        valid = ~bad

        if not valid.all():
            levels = levels[valid]
            types  = types[valid]
            rwds   = rwds[valid]
            print("BAD lats found!")
            print("levels min/max:", levels.min().item(), levels.max().item())
            print("types  min/max:", types.min().item(), types.max().item())
            print("num bad:", bad.sum().item())
            print("examples (level,type):", lats[bad][:10])

        b = levels.numel()
        lat_idx = levels * num_types + types  # [b], must be 0..41 now

        counts_flat = torch.bincount(lat_idx, minlength=num_levels * num_types)  # [42]
        counts = counts_flat.view(num_levels, num_types)

        sums = torch.zeros((2, num_levels * num_types), device=rwds.device, dtype=rwds.dtype)
        sums.scatter_add_(dim=1, index=lat_idx.unsqueeze(0).expand(2, b), src=rwds.T)
        rwds_sums = sums.view(2, num_levels, num_types)

        return counts, rwds_sums
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._grad_accum_counter % self.hparams.tm.grad_accum_steps == 0:
            if (self.global_step - self._start_step + 1) % self.hparams.checkpoint_freq == 0:
                self.ckpt_counter += 1
                self.log("ckpt_counter", self.ckpt_counter, on_step=True, on_epoch=False, sync_dist=True)
            if (self.global_step - self._start_step) % self.hparams.tm.eval_student_every == 0:
                with self._use_student_ema_weights():
                    eval_num_prompts = self.hparams.tm.eval_num_prompts
                    if self.hparams.dataset == "gsm8k" and self.a >= 6:
                        eval_num_prompts = 160
                    elif self.hparams.dataset == "math" and self.a >= 2:
                        eval_num_prompts = 55
                    rwds_eval = self._eval_student(eval_num_prompts) # [N_eval, num_reward_funcs]
                if self.hparams.dataset == "gsm8k":
                    correct_frac_eval = torch.isclose(rwds_eval[:, -1], 2.0 * torch.ones_like(rwds_eval[:, -1]), atol=1e-6, rtol=0.0).float().mean()
                    avg_rwd_eval = rwds_eval.sum(dim=-1).mean() # [N_eval,]
                    self.dict_for_logs["eval/correct_frac"] = correct_frac_eval
                    self.dict_for_logs["eval/avg_rwd"] = avg_rwd_eval
                    # seen_rwds = rwds_eval.sum(dim=-1)[: rwds_eval.shape[0] // 2]
                    # self.dict_for_logs["eval/seen_prompts_rwd_mean"] = seen_rwds.mean()
                    # correct_frac_seen = torch.isclose(rwds_eval[: rwds_eval.shape[0] // 2, -1], 2.0 * torch.ones_like(rwds_eval[: rwds_eval.shape[0] // 2, -1]), atol=1e-6, rtol=0.0).float().mean()
                    # self.dict_for_logs["eval/seen_prompts_correct_frac"] = correct_frac_seen
                    # unseen_rwds = rwds_eval.sum(dim=-1)[rwds_eval.shape[0] // 2 :]
                    # self.dict_for_logs["eval/unseen_prompts_rwd_mean"] = unseen_rwds.mean()
                    # correct_frac_unseen = torch.isclose(rwds_eval[rwds_eval.shape[0] // 2 :, -1], 2.0 * torch.ones_like(rwds_eval[rwds_eval.shape[0] // 2 :, -1]), atol=1e-6, rtol=0.0).float().mean()
                    # self.dict_for_logs["eval/unseen_prompts_correct_frac"] = correct_frac_unseen
                    print(f"[EVAL] At global step {self.global_step}, for gpu {self.global_rank}, student correctness fraction: {correct_frac_eval:.4f}, avg reward: {avg_rwd_eval:.4f}")
                elif self.hparams.dataset == "math":
                    correct_frac_eval = torch.isclose(rwds_eval[:, 0], 2.0 * torch.ones_like(rwds_eval[:, 0]), atol=1e-6, rtol=0.0).float().mean()
                    avg_rwd_eval = rwds_eval.sum(dim=-1).mean() # [N_eval,]
                    self.dict_for_logs["eval/correct_frac"] = correct_frac_eval
                    self.dict_for_logs["eval/avg_rwd"] = avg_rwd_eval
                    print(f"[EVAL] At global step {self.global_step}, for gpu {self.global_rank}, student correctness fraction: {correct_frac_eval:.4f}, avg reward: {avg_rwd_eval:.4f}")
                else:
                    raise NotImplementedError("Eval only implemented for gsm8k and math500 dataset")
                # self.save_checkpoint_now()

            if (self.global_step - self._start_step) % self.steps_per_h == 0:
                self._rebuild_buffer_next_phase = True
            # Partially refresh buffer
            elif (self.global_step - self._start_step) % self.hparams.tm.buffer_refresh_steps == 0:
                print(f"[DEBUG] gpu {self.global_rank} refreshing {self.hparams.tm.num_buffer_refresh} prompts at global step {self.global_step}")
                # Force all ranks to start/finish refresh together (avoid some ranks racing ahead into sync_dist logging)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                self._update_buffer(self.model, self.hparams.tm.num_buffer_refresh, self.comps_per_prompt)
                self.dict_for_logs["eval/teacher_buffer_update_rwd_mean"] = self._recent_buffer_rwd[-1]
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

        if not self.dict_for_logs or (self.global_step - self._start_step - 1) % self.hparams.metrics_log_every != 0:
            return
        # log all at once
        # try:
        #     # Correct for min/max style metrics (sync_dist=True averages across ranks)
        #     if "train/rwd_min" in self.dict_for_logs:
        #         local_min = torch.tensor(self.dict_for_logs["train/rwd_min"], device=self.device)
        #         global_min = self.all_gather(local_min).min().item()
        #         self.dict_for_logs["train/rwd_min"] = global_min
        #     if "train/rwd_max" in self.dict_for_logs:
        #         local_max = torch.tensor(self.dict_for_logs["train/rwd_max"], device=self.device)
        #         global_max = self.all_gather(local_max).max().item()
        #         self.dict_for_logs["train/rwd_max"] = global_max
        # except Exception:
        #     pass
        # print("before logging")
        self.log_dict(self.dict_for_logs, on_step=True, on_epoch=False, sync_dist=True)
        # self.monitor_sudoku()
        self.dict_for_logs = {}
        self._step_counter += 1
        # print("on_train_batch_end complete")
    
    def on_save_checkpoint(self, checkpoint: dict):
        print(f"saving checkpoint at a = {self.a:.4f}")
        checkpoint["tilt"] = {"a": self.a, "h": self.h}
        checkpoint["prompt_counter"] = self.curr_prompt_counter
        checkpoint["grad_accum_counter"] = int(getattr(self, "_grad_accum_counter", 0))
        checkpoint["tm_sched_state"] = copy.deepcopy(getattr(self, "_tm_sched_state", None))
        checkpoint["step_counter"] = self._step_counter + 1
        checkpoint["ckpt_counter"] = self.ckpt_counter
        if self.student_ema_enabled and (self._student_ema_state is not None):
            checkpoint["student_ema_state"] = {k: v.detach().to("cpu") for k, v in self._student_ema_state.items()}

        
    def on_load_checkpoint(self, checkpoint: dict):
        tilt = checkpoint.get("tilt", None)
        self.a = tilt.get("a", 0.0)
        self.curr_prompt_counter = checkpoint.get("prompt_counter", 0)
        self._grad_accum_counter = int(checkpoint.get("grad_accum_counter", 0))
        self._tm_sched_state = checkpoint.get("tm_sched_state", None)
        self.cv = float(checkpoint.get("cv", getattr(self, "cv", 0.0)))
        self._step_counter = checkpoint.get("step_counter", 0)
        self._resuming_from_ckpt = True
        self.ckpt_counter = checkpoint.get("ckpt_counter", 0)
        ema = checkpoint.get("student_ema_state", None)
        if ema is not None:
            # Keep on CPU for now; move to device in on_train_start
            self._student_ema_loaded_cpu = OrderedDict((k, v) for k, v in ema.items())
            self._student_ema_state = None

    def _prepare_prompts(self, num_distinct_prompts, num_completions_per_prompts):
        """
        Select `num_distinct_prompts` prompts from `self.training_prompts_dataset`,
        starting at `self.curr_prompt_counter`, repeat each prompt for
        `num_completions_per_prompts` times, and return a tensor of token IDs
        ready to be passed to `self._generate`.

        Returns: input_ids: torch.Tensor
            Shape: [num_distinct_prompts * num_completions_per_prompts, prompt_length]
        """
        # Get DDP info (defaults to 1 if not distributed)
        world_size = self.trainer.world_size
        global_rank = self.trainer.global_rank

        # ---- 1. Choose distinct prompt indices (with wrap-around) ----
        indices = []
        # if self.hparams.dataset == "math":
        #     for offset in range(num_distinct_prompts // 2):
        #         #TODO: CHECK CURR_PROMPT_COUNTER
        #         idx_easy = (self.easy_prompt_counter + (offset * world_size * 2) + global_rank) % self.hparams.math_split_index
        #         idx_hard = (self.hard_prompt_counter + (offset * world_size * 2) + global_rank) % (self.training_prompts_dataset_len - self.hparams.math_split_index)
        #         indices.append(idx_easy)
        #         indices.append(idx_hard + self.hparams.math_split_index)
        #         indices.append(idx_hard + self.hparams.math_split_index)
        #         self.easy_prompt_counter = (self.easy_prompt_counter + num_distinct_prompts // 2 * world_size) % self.hparams.math_split_index
        #         self.hard_prompt_counter = (self.hard_prompt_counter + num_distinct_prompts // 2 * world_size) % (self.training_prompts_dataset_len - self.hparams.math_split_index)
        # else:
        for offset in range(num_distinct_prompts):
            idx = (self.curr_prompt_counter + (offset * world_size) + global_rank) % self.training_prompts_dataset_len
            indices.append(idx)
        self.curr_prompt_counter += (num_distinct_prompts * world_size)
        self.curr_prompt_counter %= self.training_prompts_dataset_len
        # Remember which dataset rows were used, for reward computation later
        self._last_prompt_indices = indices

        # ---- 2. Extract structured prompts from the dataset ----
        structured_prompts = [self.training_prompts_dataset[i]["prompt"] for i in indices]
        if self.hparams.dataset == "math":
            level_and_type_list = [(self.training_prompts_dataset[i]["level"], self.training_prompts_dataset[i]["type"]) for i in indices]
            level_and_type = torch.tensor(level_and_type_list, dtype=torch.long).to(self.device).unsqueeze(1).expand(-1, num_completions_per_prompts, -1)
            # [num_distinct_prompts, num_completions_per_prompts, 2]
        else:
            level_and_type = None

        # ---- 3. Convert structured prompts to plain text and tokenize ----
        prompts_text = []
        for sp in structured_prompts:
            if isinstance(sp, list):
                # Typical case for Sudoku / GSM8K / math: [{"role": "...", "content": "..."}]
                text = self.tokenizer.apply_chat_template(sp, tokenize=False, add_generation_prompt=True)
            else:
                raise TypeError(f"Unsupported prompt type {type(sp)} in training_prompts_dataset")
            prompts_text.append(text)
            # print(text)

        input_ids = self.tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_prompt_length,
            padding_side="left",
            add_special_tokens=False,
        )["input_ids"].to(self.device)

        # if getattr(self.hparams, "test_prompt_length", False):
        #     # Determining the best max_prompt_length:
        #     sentinel = 126081
        #     starts_ok = (input_ids[:, 0] == sentinel)
        #     neq = (input_ids != sentinel)
        #     has_any_neq = neq.any(dim=1)  
        #     first_neq_pos = neq.int().argmax(dim=1)
        #     A = torch.where(has_any_neq, first_neq_pos, torch.full_like(first_neq_pos, input_ids.size(1)))
        #     false_idx = torch.nonzero(~starts_ok, as_tuple=True)[0]  # 1D [N]
        #     print(false_idx.tolist())
        #     prev_len = 0
        #     for threshold in range(0, 40, 2):
        #         idx = torch.nonzero(A < threshold, as_tuple=True)[0]  # [N] indices into A
        #         print(f"Longer than {self.hparams.max_prompt_length - threshold}:")
        #         print(idx.tolist())
        #         print(f"num prompts of length between {self.hparams.max_prompt_length - threshold} and {self.hparams.max_prompt_length - threshold + 20}: {len(idx.tolist()) - prev_len}")
        #         prev_len = len(idx.tolist())

        return input_ids.repeat_interleave(num_completions_per_prompts, dim=0), level_and_type

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
        device = self.device
        prev_adapter = model.active_adapter
        prev_training = model.training  # True if model was in train() mode
        model.set_adapter("teacher")
        model.eval()

        build_or_refresh = "building" if num_buffer_updates == self.num_buffer_prompts else "refreshing"
        print(f"{build_or_refresh} sample buffer ...")
        buffer_start_time = datetime.now()

        # ---- 1. Prepare prompts as token IDs ----
        if num_buffer_updates == self.num_buffer_prompts:
            update_rows = list(range(self.num_buffer_prompts))
            self.buffer_update_counter = 0
            self.buffer = None
            self.buffer_rewards = None
            self.level_and_type = None
        else:
            update_rows = [
                (self.buffer_update_counter + u) % self.num_buffer_prompts
                for u in range(num_buffer_updates)
            ]
            self.buffer_update_counter += num_buffer_updates
            self.buffer_update_counter %= self.num_buffer_prompts
        prompt_ids, level_and_type = self._prepare_prompts(num_buffer_updates, num_completions_per_prompt)
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
            if self.hparams.dataset == "math":
                self.level_and_type = level_and_type
        else:
            self.buffer[update_rows, :, :] = new_buffer_block
            if self.hparams.dataset == "math":
                self.level_and_type[update_rows, :, :] = level_and_type

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
        reward_kwargs["buffer_print_samples"] = int(getattr(self.hparams.tm, "buffer_print_samples", 0))
        reward_kwargs["rank"] = int(getattr(self.trainer, "global_rank", 0))

        for j, reward_func in enumerate(self.reward_funcs):
            # We mirror diffu_grpo_trainer:
            # reward_func(prompts=..., completions=..., step=..., run_name=..., **reward_kwargs)
            scores = reward_func(
                prompts=prompts_for_rewards,
                completions=completions_for_rewards,
                **reward_kwargs,
            )
            rewards_per_func[:, j] = torch.tensor(scores, device=device, dtype=torch.float32)
        if self.hparams.dataset == "gsm8k":
            rewards_per_func.clamp_(-1.0, 2.0)
        
        # # ---- DEBUG: print a few generated samples (half with reward==1) ----
        # print(f"[DEBUG] Printing out a few generated samples ...")
        # num_print = int(getattr(self.hparams.tm, "buffer_print_samples", 0))  # 0 disables printing
        # if num_print > 0 and getattr(self.trainer, "global_rank", 0) == 0:
        #     # rewards_per_func: [total_batch, num_funcs]  -> use first func
        #     r = rewards_per_func[:, 0]

        #     # Indices with reward==1 (allow tiny numeric tolerance)
        #     is_max = torch.isclose(r, torch.ones_like(r), atol=1e-6, rtol=0.0)
        #     max_idxs = torch.nonzero(is_max, as_tuple=False).squeeze(-1)
        #     all_idxs = torch.arange(r.shape[0], device=r.device)

        #     # Choose ~half from max-reward, remainder from anywhere (no duplicates)
        #     k_total = min(num_print, int(r.shape[0]))
        #     k_max = min(k_total // 2, int(max_idxs.numel()))
        #     k_any = k_total - k_max

        #     chosen = []
        #     if k_max > 0:
        #         perm = torch.randperm(max_idxs.numel(), device=r.device)[:k_max]
        #         chosen_max = max_idxs[perm]
        #         chosen.append(chosen_max)

        #     if k_any > 0:
        #         if len(chosen) > 0:
        #             already = torch.zeros_like(all_idxs, dtype=torch.bool)
        #             already[chosen[0]] = True
        #             pool = all_idxs[~already]
        #         else:
        #             pool = all_idxs

        #         if pool.numel() > 0:
        #             perm = torch.randperm(pool.numel(), device=r.device)[: min(k_any, int(pool.numel()))]
        #             chosen_any = pool[perm]
        #             chosen.append(chosen_any)

        #     if len(chosen) > 0:
        #         chosen = torch.cat(chosen, dim=0)
        #     else:
        #         chosen = torch.empty(0, dtype=torch.long, device=r.device)

        #     # Print
        #     print(
        #         f"[DEBUG] Buffer {build_or_refresh}: printing {int(chosen.numel())} samples "
        #         f"(requested={k_total}, reward==1 available={int(max_idxs.numel())})",
        #         flush=True,
        #     )

        #     max_chars = int(getattr(self.hparams.tm, "buffer_print_max_chars", 600))

        #     def _extract_answer_digits(s: str):
        #         # same idea as extract_answer_sudoku() in reward_func.py
        #         m = re.findall(r"<answer>(.*?)</answer>", s, re.DOTALL)
        #         if not m:
        #             return None
        #         return "".join(ch for ch in m[-1].strip() if ch.isdigit())

        #     for t, idx in enumerate(chosen.tolist()):
        #         # Pull the puzzle / ground truth directly (you only want puzzle as the "prompt")
        #         print(chosen)
        #         print(idx)
        #         print(reward_kwargs.get("puzzle", [None]))
        #         puzzle = reward_kwargs.get("puzzle", [None])[idx]
        #         ground_truth = reward_kwargs.get("solution", [None])[idx]

        #         # Completion: keep the whole thing, but make it human-readable:
        #         # the model often emits literal "\n" sequences, so convert them to real newlines for printing
        #         completion_raw = completions_text[idx]
        #         completion_pretty = completion_raw.replace("\\n", "\n")

        #         extracted = _extract_answer_digits(completion_raw)
        #         score = float(rewards_per_func[idx, 0].detach().cpu().item())

        #         print(f"--------------------------------", flush=True)
        #         if puzzle is not None:
        #             print(f"Puzzle: {puzzle} (length: {len(puzzle)})", flush=True)
        #         else:
        #             print("Puzzle: <missing>", flush=True)

        #         print(
        #             f"Extracted solution: {extracted}  (length: {len(extracted) if extracted else 0})",
        #             flush=True,
        #         )
        #         if ground_truth is not None:
        #             print(f"Ground_truth: {ground_truth}", flush=True)
        #         else:
        #             print("Ground_truth: <missing>", flush=True)

        #         print(f"Score: {score:.4f}", flush=True)

        #         print("\nCompletion:\n", flush=True)
        #         print(completion_pretty[:max_chars], flush=True)


        # Store as shape [num_buffer_updates, num_completions_per_prompt, num_funcs]
        new_rewards_block = rewards_per_func.view(num_buffer_updates, -1, num_funcs)
        # --- Accumulate per-phase distribution of TOTAL reward values ---
        # total reward per sample: [num_buffer_updates, num_completions_per_prompt]
        total_rewards_block = new_rewards_block.sum(dim=-1)

        # Flatten to [num_buffer_updates * num_completions_per_prompt]
        totals_flat = total_rewards_block.reshape(-1)

        # Count unique total-reward values for this update call
        # (move only the small unique+count vectors to CPU)
        uniq, cnt = torch.unique(totals_flat, sorted=True, return_counts=True)
        uniq = uniq.detach()
        cnt = cnt.detach()
        for u, c in zip(uniq.tolist(), cnt.tolist()):
            # Use float keys; if your totals are integers, they'll still print nicely.
            self._phase_total_reward_counts[float(u)] += int(c)

        self._phase_total_reward_n += int(totals_flat.numel())

        avg_rwd = float(new_rewards_block.mean() * new_rewards_block.shape[-1])
        print(f"[EVAL] average reward = {avg_rwd:.3f}")
        temp_log_dict = {}
        for i in range(num_completions_per_prompt + 1):
            if self.hparams.dataset == "gsm8k":
                rows_ok = (new_rewards_block[:,:,-1] == 2).sum(dim=1) == i
                temp_log_dict[f"buffer/buffer_per_prompt_{i}_correct"] = rows_ok.float().mean()
            elif self.hparams.dataset == "math":
                rows_ok = (new_rewards_block[:,:,0] == 2).sum(dim=1) == i
                temp_log_dict[f"buffer/buffer_per_prompt_{i}_correct"] = rows_ok.float().mean()
        if self.hparams.dataset == "math":
            dim1, dim2, dim3 = level_and_type.shape
            counts, rwds_sums = self._lat_stats(level_and_type.reshape(dim1 * dim2, dim3), rewards_per_func) # [6,7], [2,6,7]
            for i in range(5):
                count_i = counts[i, :].sum().item()
                self._math_lat_stats_dict[f"count_level_{i+1}"] += int(count_i)
                rwds_sums_i = rwds_sums[:, i, :].sum(dim=-1) # [2,]
                self._math_lat_stats_dict[f"level_{i+1}_correct_num"] += int(rwds_sums_i[0].item() // 2)
            for j in range(7):
                count_j = counts[:, j].sum().item()
                self._math_lat_stats_dict[f"count_{ID_TO_TYPE[j]}"] += int(count_j)
                rwds_sums_j = rwds_sums[:, :, j].sum(dim=-1) # [2,]
                self._math_lat_stats_dict[f"{ID_TO_TYPE[j]}_correct_num"] += int(rwds_sums_j[0].item() // 2)
            if (self.global_step - self._start_step) % getattr(self.hparams.tm, "math_log_lat_every", 1) == 0 and self._grad_accum_counter % self.hparams.tm.grad_accum_steps == 0:
                temp_log_dict = self._build_temp_log_dict(temp_log_dict)
                self._init_math_lat_dict()
        print(f"[DEBUG] finished temp_log_dict computation")
        if (self.global_step - self._start_step) % self.hparams.tm.steps_per_h > 0:
            self.log_dict(temp_log_dict, on_step=True, on_epoch=False, sync_dist=True)
        self._recent_buffer_rwd.append(avg_rwd)
        if self.buffer_rewards is None:
            self.buffer_rewards = new_rewards_block
            if getattr(self, "_rebuild_buffer_next_phase", False) and getattr(self.hparams.tm, "rwd_shift_auto", True):
                self.rwd_shift = - self.all_gather(torch.tensor(avg_rwd, device=self.device)).mean().item()
                print(f"New phase: setting rwd_shift = {self.rwd_shift:.3f}")
                self._rebuild_buffer_next_phase = False
        else:
            self.buffer_rewards[update_rows, :, :] = new_rewards_block

        buffer_end_time = datetime.now()
        buffer_build_time = (buffer_end_time - buffer_start_time).total_seconds()
        print(f"Finished {build_or_refresh} reward buffer, took {buffer_build_time}")

        # restore adapter and training state
        if prev_training:
            model.train()
        model.set_adapter(prev_adapter)

    def _build_temp_log_dict(self, temp_log_dict: dict) -> dict:
        """
        Uses self._math_lat_stats_dict (local to each GPU/rank) and returns temp_log_dict
        where counts are summed across GPUs and correct_fracs are computed from globally
        summed correct_nums / globally summed counts.
        """
        device = self.device
        eps = 1e-6  # avoid div-by-zero
        stats = self._math_lat_stats_dict

        # ---- 1) Build ordered lists of keys we will aggregate ----
        level_pairs = [(f"count_level_{i+1}", f"level_{i+1}_correct_num") for i in range(6)]
        type_pairs  = [(f"count_{ID_TO_TYPE[j]}", f"{ID_TO_TYPE[j]}_correct_num") for j in range(7)]
        pairs = level_pairs + type_pairs

        # ---- 2) Pack local values into tensors (shape [N]) ----
        # counts and corrects in matching order
        local_counts = []
        local_corrects = []
        for count_k, corr_k in pairs:
            c = stats.get(count_k, 0)
            r = stats.get(corr_k, 0)

            # convert to tensors on device
            c = c if torch.is_tensor(c) else torch.tensor(c, device=device)
            r = r if torch.is_tensor(r) else torch.tensor(r, device=device)

            # make them scalars
            c = c.to(device=device).reshape(())
            r = r.to(device=device).reshape(())

            local_counts.append(c)
            local_corrects.append(r)

        local_counts = torch.stack(local_counts)    # [N]
        local_corrects = torch.stack(local_corrects)  # [N]

        # ---- 3) all_gather across GPUs: result [world_size, N] on each rank ----
        gathered_counts = self.all_gather(local_counts)
        gathered_corrects = self.all_gather(local_corrects)

        # If running with DDP, these are typically [world_size, N].
        # Some strategies may add extra dims; flatten safely:
        gathered_counts = gathered_counts.reshape(-1, gathered_counts.shape[-1])      # [ws, N]
        gathered_corrects = gathered_corrects.reshape(-1, gathered_corrects.shape[-1])# [ws, N]

        # ---- 4) global sums ----
        global_counts = gathered_counts.sum(dim=0)      # [N]
        global_corrects = gathered_corrects.sum(dim=0)  # [N]
        global_fracs = global_corrects / (global_counts + eps)  # [N]

        # ---- 5) Build temp_log_dict with requested keys ----

        # levels first (6)
        for i in range(6):
            count_k, corr_k = level_pairs[i]
            temp_log_dict[f"buffer/{count_k}"] = global_counts[i]
            temp_log_dict[f"buffer/level_{i+1}_correct_frac"] = global_fracs[i]

        # types next (7)
        offset = 6
        for j in range(7):
            tname = ID_TO_TYPE[j]
            temp_log_dict[f"buffer/count_{tname}"] = global_counts[offset + j]
            temp_log_dict[f"buffer/{tname}_correct_frac"] = global_fracs[offset + j]

        return temp_log_dict

    
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
        
        with torch.no_grad(), self._use_adapter("teacher"):
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

    
    @torch.no_grad()
    def _kl_from_logits(self, logits_A, logits_B, mask_indices):
        log_A = F.log_softmax(logits_A, dim=-1)
        log_B = F.log_softmax(logits_B, dim=-1)
        kl = F.kl_div(log_A, log_B, reduction='none', log_target=True).sum(-1)
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
    
    def _prepare_prompts_for_eval(self, num_total_prompts, num_seen_prompts=0):
        # Get DDP info (defaults to 1 if not distributed)
        world_size = self.trainer.world_size
        global_rank = self.trainer.global_rank

        # ---- 1. Choose distinct prompt indices (with wrap-around) ----
        seen_indices = []
        for offset in range(-num_seen_prompts, 0):
            idx = (self.curr_prompt_counter + (offset * world_size) + global_rank) % self.training_prompts_dataset_len
            seen_indices.append(idx)
        if self.hparams.dataset == "gsm8k":
            if self.a < 6:
                unseen_indices = [offset * world_size * 2 + global_rank * 2 for offset in range(num_total_prompts - num_seen_prompts)]
            else:
                unseen_indices = [offset * world_size + global_rank for offset in range(num_total_prompts - num_seen_prompts)]
        elif self.hparams.dataset == "math":
            if self.a < 2:
                unseen_indices = [offset * world_size * 2 + global_rank * 2 for offset in range(num_total_prompts - num_seen_prompts)]
            else:
                unseen_indices = [offset * world_size + global_rank for offset in range(num_total_prompts - num_seen_prompts)]
        # Remember which dataset rows were used, for reward computation later
        self._last_eval_indices = (seen_indices, unseen_indices)

        # ---- 2. Extract structured prompts from the dataset ----
        structured_prompts = []
        for i in seen_indices:
            structured_prompts.append(self.training_prompts_dataset[i]["prompt"])
        for i in unseen_indices:
            structured_prompts.append(self.test_prompts_dataset[i]["prompt"])

        # ---- 3. Convert structured prompts to plain text and tokenize ----
        prompts_text = []
        for sp in structured_prompts:
            if isinstance(sp, list):
                # Typical case for Sudoku / GSM8K / math: [{"role": "...", "content": "..."}]
                text = self.tokenizer.apply_chat_template(sp, tokenize=False, add_generation_prompt=True)
            else:
                raise TypeError(f"Unsupported prompt type {type(sp)} in training_prompts_dataset")
            prompts_text.append(text)
            # print(text)

        input_ids = self.tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding="max_length", # longest
            truncation=True,
            max_length=self.hparams.max_prompt_length,
            padding_side="left",
            add_special_tokens=False,
        )["input_ids"].to(self.device)

        return input_ids

    def _eval_student(self, num_buffer_eval):
        device = self.device
        prev_adapter = self.model.active_adapter
        prev_training = self.model.training  # True if model was in train() mode
        self.model.set_adapter("student")
        self.model.eval()

        print(f"evaluating student ...")
        eval_start_time = datetime.now()

        # ---- 1. Prepare prompts as token IDs ----
        prompt_ids = self._prepare_prompts_for_eval(num_buffer_eval, 0)
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
                # chunk_completion_ids = self._generate(
                #     model=self.model,
                #     prompt=prompt_ids[start:end],
                #     steps=self.hparams.diffusion_steps,
                #     gen_length=gen_length,
                #     block_length=self.hparams.block_length,
                #     temperature=0.0,
                #     cfg_scale=self.hparams.cfg_scale,
                #     remasking="low_confidence"
                # )  # [end-start, seq_len]
                chunk_completion_ids = generate(
                    model=self.model,
                    prompt=prompt_ids[start:end],
                    steps=self.hparams.diffusion_steps,
                    gen_length=gen_length,
                    block_length=self.hparams.block_length,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                )

            prompt_completion_ids[start:end].copy_(chunk_completion_ids)

        # ---- 3. Decode completions to text for reward computation ----
        completion_ids = prompt_completion_ids[:, prompt_len:]  # [total_batch, gen_length]
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # ---- 4. Build reward inputs: prompts, completions, and extra dataset columns ----
        data_keys = [key for key in self.training_prompts_dataset[0].keys() if key != "prompt"]
        # For each generated sample we need:
        #   - a structured prompt (list of chat messages)
        #   - a structured completion (list with one assistant message)
        #   - one entry per dataset column (e.g. "answer", "puzzle", "solution", "target", "numbers")
        prompts_for_rewards = []
        completions_for_rewards = []
        reward_kwargs = {key: [] for key in data_keys}

        for row_idx in self._last_eval_indices[0]:
            row = self.training_prompts_dataset[row_idx]
            base_prompt = row["prompt"]  # list[{"role": ..., "content": ...}, ...]

            # Structured prompt for this completion
            prompts_for_rewards.append(base_prompt)

            # Copy all extra fields for this completion
            for key in data_keys:
                reward_kwargs[key].append(row[key])
        for row_idx in self._last_eval_indices[1]:
            row = self.test_prompts_dataset[row_idx]
            base_prompt = row["prompt"]  # list[{"role": ..., "content": ...}, ...]

            # Structured prompt for this completion
            prompts_for_rewards.append(base_prompt)

            # Copy all extra fields for this completion
            for key in data_keys:
                reward_kwargs[key].append(row[key])

        # Turn plain completions into chat-style completions [{"role": "assistant", "content": "..."}]
        completions_for_rewards = []
        for text in completions_text:
            completions_for_rewards.append([{"role": "assistant", "content": text}])

        # ---- 5. Compute rewards for every sequence in the buffer ----
        num_funcs = len(self.reward_funcs)
        rewards_per_func = torch.zeros(total_batch, num_funcs, device=device)
        reward_kwargs["buffer_print_samples"] = int(getattr(self.hparams.tm, "buffer_print_samples", 0))
        reward_kwargs["rank"] = int(getattr(self.trainer, "global_rank", 0))

        for j, reward_func in enumerate(self.reward_funcs):
            # We mirror diffu_grpo_trainer:
            # reward_func(prompts=..., completions=..., step=..., run_name=..., **reward_kwargs)
            scores = reward_func(
                prompts=prompts_for_rewards,
                completions=completions_for_rewards,
                **reward_kwargs,
            )
            rewards_per_func[:, j] = torch.tensor(scores, device=device, dtype=torch.float32)

        eval_end_time = datetime.now()
        eval_time = (eval_end_time - eval_start_time).total_seconds()
        print(f"Finished evaluating student, took {eval_time}")

        # restore adapter and training state
        if prev_training:
            self.model.train()
        self.model.set_adapter(prev_adapter)

        return rewards_per_func # [num_buffer_eval, num_reward_funcs]
    
    def _init_math_lat_dict(self):
        self._math_lat_stats_dict = {}
        for i in range(5):
            self._math_lat_stats_dict[f"count_level_{i+1}"] = 0
            self._math_lat_stats_dict[f"level_{i+1}_correct_num"] = 0
        for j in range(7):
            self._math_lat_stats_dict[f"count_{ID_TO_TYPE[j]}"] = 0
            self._math_lat_stats_dict[f"{ID_TO_TYPE[j]}_correct_num"] = 0