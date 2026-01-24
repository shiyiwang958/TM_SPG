import torch
import wandb
import math
import os
import hydra
import math
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig    
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from lightning_modules import TiltMatchingModule
import copy
from peft import get_peft_model_state_dict, set_peft_model_state_dict

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
    # get_sudoku_questions,
    get_sudoku_questions_new,
    set_random_seed,
    get_math_questions,
    reorder_by_level_halves
)

import typing
from collections import defaultdict
from torch.serialization import add_safe_globals
from lightning_fabric.utilities.data import AttributeDict
from omegaconf.nodes import AnyNode
from omegaconf.base import Metadata, ContainerMetadata
def _unwrap(m):
    # DDP / FSDP / Lightning wrappers sometimes stash the real model here
    for attr in ["module", "model", "base_model", "net", "backbone", "transformer"]:
        if hasattr(m, attr):
            inner = getattr(m, attr)
            if inner is not None:
                return inner
    return m

def _debug_attention_once(model, tokenizer):

    model = _unwrap(model)

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


add_safe_globals([
    (typing.Any, "typing.Any"),  # special: not a class, use (obj, "qualified.name")
    dict,
    defaultdict,
    AttributeDict,
    AnyNode,
    Metadata,
    ContainerMetadata,
    DictConfig,
])

def train(cfg: DictConfig):
    # Set seed for reproducibility
    set_random_seed(cfg.seed)

    # Set wandb logger if specified
    if "wandb" in cfg and rank_zero_only.rank == 0:
        wandb_name = cfg.wandb.name
        init_kwargs = dict(
            project = cfg.wandb.project,
            entity = cfg.wandb.entity,
            name = wandb_name,
            config = OmegaConf.to_container(cfg, resolve = True)
        )
        # resume wandb run if we're resuming from a checkpoint
        if "resume_path" in cfg:
            init_kwargs["resume"] = "allow"

        # init wandb    
        wandb.init(**init_kwargs)
        wandb_logger = WandbLogger(
            project = wandb.run.project,
            name = wandb.run.name,
            log_model = False,
        )
    else:
        wandb_logger = None

    # Load dataset based on configuration
    test_dataset = None
    if cfg.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
        test_dataset = get_gsm8k_questions("test")
    elif cfg.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    # elif cfg.dataset == "sudoku":
    #     dataset = get_sudoku_questions()
    #     reward_functions = [sudoku_reward_func]
    elif cfg.dataset == "sudoku_new":
        dataset = get_sudoku_questions_new(few_shot=cfg.few_shot)
        reward_functions = [sudoku_reward_func]
    elif cfg.dataset == "math":
        dataset = get_math_questions("train")
        # The columns are: 'level' (int), 'type' (int), 'prompt', 'answer'
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
        test_dataset = get_math_questions("test")

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=cfg.seed)

    if cfg.dataset == "math":
        dataset, split_idx = reorder_by_level_halves(dataset)
        print(f"[INFO] Reordered math dataset by level halves with split index at {split_idx}.")
        cfg.math_split_idx = split_idx

    # Split dataset if needed
    if cfg.dataset in ["countdown", "sudoku", "sudoku_new"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        # Fallback debug: if we are here, torchrun isn't passing the var
        print("[WARNING] LOCAL_RANK not found in env, defaulting to 0. This will cause OOM on multi-GPU.")
        local_rank = 0
    torch.cuda.set_device(local_rank)

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base model and tokenizer
    base_model = AutoModel.from_pretrained(
        cfg.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": torch.cuda.current_device()}
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.use_cache = False
    # TODO: Need to load the LoRA weights onto the base model when starting from a checkpoint

    # Load the Tilt Matching training module
    model = TiltMatchingModule(
        base_model=base_model,
        tokenizer=tokenizer,
        training_prompts_dataset=train_set,
        test_prompts_dataset=test_dataset,
        reward_funcs=reward_functions,
        **cfg,
    )

    init_ckpt = getattr(cfg, "init_ckpt_path", None)
    if init_ckpt is not None and getattr(cfg, "resume_path", None) is None:
        ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        # load adapters into the module (loads both student + teacher if present)
        model.load_state_dict(sd, strict=False)

        # REBASE: teacher := student snapshot
        with torch.no_grad():
            student_state = get_peft_model_state_dict(model.model, adapter_name="student")
            set_peft_model_state_dict(model.model, student_state, adapter_name="teacher")
            for name, p in model.model.named_parameters():
                if ".teacher" in name:
                    p.requires_grad_(False)
            model.model.set_adapter("student")

        # start at a=h (or whatever you want)
        model.a = float(getattr(cfg.tm, "a_start", 0.0))

        # important: fresh scheduler at start of this new phase
        model._tm_sched_state = None

    # Configure trainer
    trainer_kwargs = dict(
        num_nodes = cfg.nodes,
        accelerator = "gpu",
        devices = cfg.devices,
        # strategy = "ddp" if cfg.nodes > 1 else "auto",
        strategy = "ddp",
        precision="bf16-mixed",

        accumulate_grad_batches = 1,

        log_every_n_steps = 1,
        enable_checkpointing = True,
        default_root_dir = cfg.checkpoint_dir,
        enable_progress_bar = False,

        # Unlimited steps; stop manually by setting trainer.should_stop = True
        max_steps = -1,
        # Lightning still requires a finite epoch cap; set a very large number
        max_epochs = 10**12,
    )

    ckpt_steps = cfg.checkpoint_freq
    checkpoint_callback = ModelCheckpoint(
        save_last = True,
        dirpath = cfg.checkpoint_dir,
        save_top_k = -1,
        every_n_train_steps = ckpt_steps,
        save_on_train_epoch_end = False,
        filename = "checkpoint-a-{ckpt_a:.3f}-{ckpt_counter}",
        auto_insert_metric_name=False,
        save_on_exception=True,
    )

    # finish trainer kwargs
    trainer_kwargs["callbacks"] = [checkpoint_callback]
    if wandb_logger is not None:
        trainer_kwargs["logger"] = wandb_logger
    trainer = pl.Trainer(**trainer_kwargs)

    # Create a dummy dataloader since training is handled inside the module
    dummy_dataset = torch.utils.data.TensorDataset(torch.zeros(1))
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1)

    # Train the model
    resume_path = getattr(cfg, "resume_path", None)
    print(f"Resume path is: {resume_path}")
    if resume_path is not None:
        trainer.fit(
            model,
            train_dataloaders = dummy_loader,
            ckpt_path = resume_path
        )
    else:
        trainer.fit(
            model,
            train_dataloaders = dummy_loader,
        )


#-------------------------------- Train ------------------------------------
@hydra.main(config_path = "config", config_name = "tilt_matching_gsm.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__=="__main__":
    main()