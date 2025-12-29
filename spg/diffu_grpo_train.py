# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig

# Custom imports
from diffu_grpo_trainer import DiffuGRPOTrainer
from spg_trainer import SPGTrainer
from diffu_grpo_config import DiffuGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    get_sudoku_questions_new,
    set_random_seed,
    get_math_questions,
)
# --- add near the top of spg/diffu_grpo_train.py ---
import os

def _debug_attention_once(model, tokenizer):

    if not torch.cuda.is_available():
        print("[attn-debug] CUDA not available; skipping.")
        return

    print("\n[attn-debug] ===== attention debug =====")
    print("[attn-debug] torch:", torch.__version__)

    # What the model/config claims (HF convention; may be None for custom models)
    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    print("[attn-debug] model.config._attn_implementation:", attn_impl)

    # PyTorch SDPA feature flags / availability (these matter if the model uses SDPA under the hood)
    # (PyTorch exposes these knobs for Flash / mem-efficient / math SDPA.)  :contentReference[oaicite:3]{index=3}
    try:
        print("[attn-debug] torch.backends.cuda.is_flash_attention_available():",
              torch.backends.cuda.is_flash_attention_available())
        print("[attn-debug] torch.backends.cuda.flash_sdp_enabled():",
              torch.backends.cuda.flash_sdp_enabled())
        print("[attn-debug] torch.backends.cuda.mem_efficient_sdp_enabled():",
              torch.backends.cuda.mem_efficient_sdp_enabled())
        print("[attn-debug] torch.backends.cuda.math_sdp_enabled():",
              torch.backends.cuda.math_sdp_enabled())
    except Exception as e:
        print("[attn-debug] (could not query torch.backends.cuda *sdp* flags):", repr(e))

    # Check whether flash-attn package is importable (relevant if HF "flash_attention_2" is used)
    try:
        import flash_attn  # noqa: F401
        print("[attn-debug] flash_attn import: OK")
    except Exception as e:
        print("[attn-debug] flash_attn import: FAILED:", repr(e))

    # Run a tiny forward pass and profile CUDA ops
    from torch.profiler import profile, ProfilerActivity

    device = next(model.parameters()).device
    model.eval()

    text = "hello " * 64  # long enough to exercise attention
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Warmup (avoids one-time compilation noise)
    with torch.no_grad():
        _ = model(**inputs)

    def _run_and_print(tag, force_flash_sdpa: bool):
        print(f"\n[attn-debug] --- profiler run: {tag} (force_flash_sdpa={force_flash_sdpa}) ---")

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                if force_flash_sdpa:
                    # Force SDPA to prefer FlashAttention backend (only affects models using SDPA).
                    # PyTorch docs: torch.nn.attention.sdpa_kernel. :contentReference[oaicite:4]{index=4}
                    from torch.nn.attention import sdpa_kernel, SDPBackend
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION, set_priority=True):
                        _ = model(**inputs)
                else:
                    _ = model(**inputs)

        # Print top CUDA ops
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

        # Also print any op names that look attention-related
        keys = [e.key for e in prof.key_averages()]
        hits = [k for k in keys if any(s in k.lower() for s in ["flash", "scaled_dot_product", "sdp", "attention"])]
        print("[attn-debug] ops containing flash/scaled_dot_product/sdp/attention:")
        for k in sorted(set(hits)):
            print("  ", k)

    _run_and_print("natural", force_flash_sdpa=False)
    _run_and_print("forced_sdpa_flash", force_flash_sdpa=True)

    print("[attn-debug] ===== end attention debug =====\n")


def main(grpo_config, model_config):

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    # elif grpo_config.dataset == "sudoku":
    #     dataset = get_sudoku_questions()
    #     reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "sudoku_new":
        dataset = get_sudoku_questions_new(few_shot=grpo_config.few_shot)
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku", "sudoku_new"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        # attn_implementation=model_config.attn_implementation,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    if grpo_config.trainer == "diffu_grpo":
        # Initialize and run trainer
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "spg":
        trainer = SPGTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    else:
        raise ValueError(f"Invalid trainer: {grpo_config.trainer}")

    # if os.environ.get("SPG_DEBUG_ATTN", "0") == "1":
    #     _debug_attention_once(model, tokenizer)

    train_dataloader = trainer.get_train_dataloader()

    if trainer.accelerator.is_main_process:
        import math
        L = len(train_dataloader)
        print("len(train_dataloader) =", L)  # microsteps per epoch
        K = trainer.args.gradient_accumulation_steps
        mu = trainer.num_iterations
        W = trainer.accelerator.num_processes
        G = trainer.args.num_generations
        Gb = trainer.args.generation_batch_size
        U = math.ceil(L / K)  # optimizer (global) steps per epoch
        S_gen = math.ceil(U / mu)  # generation optimizer steps per epoch
        prompts_per_step = (Gb // G) * W  # distinct prompts per generation step (global)

        print("optimizer steps per epoch =", U)
        print("generation optimizer steps per epoch ≈", S_gen)
        print("distinct prompts per generation microstep =", prompts_per_step)
        print("distinct prompts used for generation per epoch ≈",
            S_gen * prompts_per_step * K)

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
