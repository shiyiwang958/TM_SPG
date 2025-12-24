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
    get_sudoku_questions_new,
    set_random_seed,
    get_math_questions,
)

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
    if cfg.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif cfg.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif cfg.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif cfg.dataset == "sudoku_new":
        dataset = get_sudoku_questions_new(few_shot=cfg.few_shot)
        reward_functions = [sudoku_reward_func]
    elif cfg.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=cfg.seed)

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
        reward_funcs=reward_functions,
        **cfg,
    )

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
    eps = 1e-6
    ckpt_steps = int(cfg.tm.steps_per_h * math.floor((cfg.checkpoint_freq + eps) / cfg.tm.h))

    checkpoint_callback = ModelCheckpoint(
        save_last = True,
        dirpath = cfg.checkpoint_dir,
        save_top_k = -1,
        every_n_train_steps = ckpt_steps,
        save_on_train_epoch_end = False,
        filename = "checkpoint-a-{ckpt_a:.3f}",
        auto_insert_metric_name=False,
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
            ckpt_path = resume_path #TODO: Later add resume functionality
        )
    else:
        trainer.fit(
            model,
            train_dataloaders = dummy_loader,
        )


#-------------------------------- Train ------------------------------------
@hydra.main(config_path = "config", config_name = "tilt_matching.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__=="__main__":
    main()
