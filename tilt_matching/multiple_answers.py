import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import math

# System prompt from data_utils.py
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Countdown question
numbers = [97, 2, 60]
target = 67

# Model paths
model_name = "/n/netscratch/albergo_lab/Everyone/frank/hf_models/LLaDA-8B-Instruct"  # Adjust to your model path
checkpoint_path = "/n/netscratch/albergo_lab/Everyone/frank/llada_tm/countdown_onpolicy/checkpoint-a-0.900-v1.ckpt"  # Set to checkpoint path if you want to load weights

# LoRA config (matching tilt_countdown.py)
peft_config = LoraConfig(
    r=32,  # Adjust if different in your config
    lora_alpha=64,  # Adjust if different in your config
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.0,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Wrapping with PEFT...")
peft_wrapped = get_peft_model(base_model, peft_config, adapter_name="student")
peft_wrapped.add_adapter("teacher", peft_config)

# Copy student weights to teacher (matching tilt_countdown.py initialization)
from peft import get_peft_model_state_dict
student_state = get_peft_model_state_dict(peft_wrapped, adapter_name="student")
set_peft_model_state_dict(peft_wrapped, student_state, adapter_name="teacher")

# Freeze teacher adapter
for name, param in peft_wrapped.named_parameters():
    if ".teacher" in name:
        param.requires_grad = False

# Set active adapter to teacher (for generation)
peft_wrapped.set_adapter("teacher")

# Load checkpoint if provided
if checkpoint_path is not None:
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Extract adapter weights from checkpoint
    teacher_state = {}
    for key, value in ckpt.items():
        if key.startswith("base_adapter."):
            bare = key[len("base_adapter."):]
            teacher_state[bare] = value.to(peft_wrapped.device)
    set_peft_model_state_dict(peft_wrapped, teacher_state, adapter_name="teacher")
    print("Checkpoint loaded")

model = peft_wrapped
model.eval()

# Create prompt in the same format as get_countdown_questions
prompt_content = f"{SYSTEM_PROMPT}\nUsing only the numbers {numbers}, create an arithmetic expression that evaluates to exactly {target}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"

structured_prompt = [{"role": "user", "content": prompt_content}]

# Convert to text using chat template
prompt_text = tokenizer.apply_chat_template(
    structured_prompt,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize - don't pad, just truncate if needed
prompt_ids = tokenizer(
    text=prompt_text,
    return_tensors="pt",
    truncation=True,
    max_length=512,  # max_prompt_length
    add_special_tokens=False,
)["input_ids"].to(model.device)

print(f"\nPrompt text:\n{prompt_text}\n")
print(f"Prompt shape: {prompt_ids.shape}")
print(f"Actual prompt length (no padding): {prompt_ids.shape[1]}")

# Repeat for 12 completions
num_completions = 2
prompt_ids_batch = prompt_ids.repeat(num_completions, 0)

print(f"Generating {num_completions} completions...")


# Generation function from tilt_countdown.py
def _generate(
    model,
    tokenizer,
    prompt,
    steps=128,
    gen_length=256,  # max_completion_length
    block_length=32,
    temperature=1.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
    print(f"\n[_generate] Input prompt shape: {prompt.shape}")
    with torch.amp.autocast("cuda", enabled=True):
        bs = prompt.shape[0]
        dtype = model.dtype
        prompt_len = prompt.shape[1]
        print(f"[_generate] bs={bs}, prompt_len={prompt_len}, gen_length={gen_length}")
        x = torch.full((bs, prompt_len + gen_length), mask_id, dtype=torch.long).to(model.device)
        print(f"[_generate] Created x with shape: {x.shape}")
        x[:, :prompt_len] = prompt.clone()
        print(f"[_generate] Copied prompt into x, x.shape still: {x.shape}")
        print(f"[_generate] First 10 tokens of x[0]: {x[0, :10].tolist()}")

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        # Adjust steps if needed
        steps_per_block = max(1, steps // num_blocks)

        for num_block in range(num_blocks):
            start_idx = prompt_len + num_block * block_length
            end_idx = prompt_len + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                torch.cuda.empty_cache()
                mask_index = x[:, prompt_len:] == mask_id  # [B, gen_len]

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = _new_forward(model, x_, gen_length)  # [2*B, gen_len, V]
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = _new_forward(model, x, gen_length)  # [B, gen_len, V]

                # Apply Gumbel noise for sampling
                logits_with_noise = _add_gumbel_noise(
                    logits, temperature=temperature, dtype=dtype
                )
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, gen_len]
                del logits_with_noise

                # Handle remasking strategy
                if remasking == "low_confidence":
                    p = F.softmax(logits.to(dtype), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, gen_len]
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)  # [B, gen_len]
                else:
                    raise NotImplementedError(remasking)
                del logits

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx - prompt_len:] = float("-inf")

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

        print(f"[_generate] Returning x with shape: {x.shape}")
        print(f"[_generate] First 10 tokens of x[0]: {x[0, :10].tolist()}")
        return x


def _get_num_transfer_tokens(mask_index, steps):
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


def _add_gumbel_noise(logits, temperature, dtype):
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


def _unwrap_llada_core(m):
    """Get the core LLaDAModel (with .transformer and .config)."""
    from peft import PeftModelForCausalLM
    assert isinstance(m, PeftModelForCausalLM)
    lm = m.base_model  # peft.tuners.lora.model.LoraModel
    core = getattr(lm, "model", None)  # LLaDAModelLM
    if core is None or not hasattr(core.base_model, "transformer"):
        raise ValueError("Expected a LLaDA HF model with .model.transformer")
    return core.base_model  # LLaDAModel


def _llada_hidden_no_logits(model, input_ids, attention_mask=None):
    """Run the LLaDA stack up to final layer norm, but DO NOT compute logits yet."""
    core = _unwrap_llada_core(model)
    cfg = core.config
    tfm = core.transformer

    # MDM constraints
    assert not cfg.alibi, "Alibi is not supported for LLaDA MDM."
    assert cfg.rope, "Rope must be enabled for LLaDA-8B-Instruct."
    use_cache = False
    past_key_values = None

    batch_size, seq_len = input_ids.shape
    past_length = 0

    # Embeddings
    x = tfm.wte(input_ids)
    if cfg.input_emb_norm:
        x = x * (cfg.d_model ** 0.5)

    # Embedding dropout
    x = tfm.emb_drop(x)

    # Attention mask → additive bias
    if attention_mask is not None and 0.0 in attention_mask:
        attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
    else:
        attention_mask = None

    attention_bias = None

    # Merge attention_mask with default bidirectional bias
    if (attention_mask is not None or cfg.alibi or past_key_values is not None or attention_bias is not None):
        if attention_bias is None and cfg.alibi:
            raise RuntimeError("ALiBi path should be disabled for LLaDA-8B-Instruct")
        elif attention_bias is None:
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

        attention_bias.masked_fill_(attention_bias == float("-inf"), torch.finfo(attention_bias.dtype).min)

    # Transformer blocks
    if cfg.block_group_size == 1:
        for block_idx, block in enumerate(tfm.blocks):
            from configuration_llada import ActivationCheckpointingStrategy
            layer_past = None
            strat = core.activation_checkpointing_strategy

            use_ckpt = (
                strat == ActivationCheckpointingStrategy.whole_layer
                or (strat == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
                or (strat == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
                or (strat == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
            )

            if use_ckpt:
                x, _ = core._activation_checkpoint_fn(
                    block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                )
            else:
                x, _ = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
    else:
        for group_idx, block_group in enumerate(tfm.block_groups):
            layers_past = None
            x, _ = block_group(x, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache)

    # Final layer norm
    x = tfm.ln_f(x)
    return x


def _llada_logits_on_suffix(model, hidden, gen_len):
    """Compute logits only for the last gen_len positions of each sequence."""
    lm = model.base_model
    core = _unwrap_llada_core(model)
    cfg = core.config

    B, L, d_model = hidden.shape
    assert gen_len <= L, f"gen_len={gen_len} cannot exceed sequence length L={L}"
    hidden_suffix = hidden[:, -gen_len:, :]

    # Get the output embedding / projection
    out_module = lm.get_output_embeddings()

    if isinstance(out_module, torch.nn.Embedding):
        weight = out_module.weight
        bias = None
        logits = F.linear(hidden_suffix, weight, bias)
    elif isinstance(out_module, torch.nn.Linear):
        logits = out_module(hidden_suffix)
    else:
        raise TypeError(f"Unsupported output embeddings module type: {type(out_module)}")

    if getattr(cfg, "scale_logits", False):
        logits = logits * (1.0 / math.sqrt(cfg.d_model))

    return logits


def _new_forward(model, x, gen_length):
    """Efficient forward pass that only computes logits for the completion suffix."""
    hidden = _llada_hidden_no_logits(model, x, attention_mask=None)
    return _llada_logits_on_suffix(model, hidden, gen_length)


# Generate completions
with torch.no_grad():
    generated_ids = _generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_ids_batch,
        steps=128,
        gen_length=256,
        block_length=32,
        temperature=1.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    )

print(f"\nDebug after generation:")
print(f"  prompt_ids_batch.shape: {prompt_ids_batch.shape}")
print(f"  generated_ids.shape: {generated_ids.shape}")

# Decode completions
prompt_len = prompt_ids.shape[1]
print(f"  prompt_len: {prompt_len}")
print(f"  Slicing generated_ids[:, {prompt_len}:]")
completion_ids = generated_ids[:, prompt_len:]

# Debug: check if completion_ids contain mask tokens
mask_id = 126336
num_masks = (completion_ids == mask_id).sum(dim=1)
print(f"\nDebug: Number of mask tokens per completion: {num_masks.tolist()}")
print(f"Debug: Completion shape: {completion_ids.shape}")
print(f"Debug: First completion tokens (first 20): {completion_ids[0, :20].tolist()}")

print("\n" + "="*80)
print(f"Generated {num_completions} completions for:")
print(f"Numbers: {numbers}, Target: {target}")
print("="*80 + "\n")

for i in range(num_completions):
    # Try decoding without skip_special_tokens first to see what's there
    completion_text_with_special = tokenizer.decode(completion_ids[i], skip_special_tokens=False)
    completion_text = tokenizer.decode(completion_ids[i], skip_special_tokens=True)
    
    # Extract answer from <answer> tags
    import re
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, completion_text, re.DOTALL)
    extracted_answer = matches[-1].strip() if matches else "No answer found"
    
    print(f"Completion {i+1}:")
    print(f"  Extracted answer: {extracted_answer}")
    print(f"  Full completion {completion_text}\n")   

print("\nDone!")
