import torch

from generate import generate
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict


def chat():
    device = 'cuda'
    model = AutoModel.from_pretrained("/n/netscratch/albergo_lab/Everyone/frank/hf_models/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    path = "/n/netscratch/albergo_lab/Everyone/frank/llada_tm/math_test/last.ckpt"
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Loaded checkpoint from {path}")
    hparams = ckpt.get("hyper_parameters", {})
    lora_cfg = LoraConfig(
        r=hparams.get("lora_r", 128),
        lora_alpha=hparams.get("lora_alpha", 64),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type=hparams.get("peft_task_type", "CAUSAL_LM"),
        lora_dropout=hparams.get("lora_dropout", 0.05),
    )
    tokenizer = AutoTokenizer.from_pretrained("/n/netscratch/albergo_lab/Everyone/frank/hf_models/LLaDA-8B-Instruct", trust_remote_code=True)

    peft_model = get_peft_model(model, lora_cfg, adapter_name="teacher")
    sd = ckpt.get("state_dict", ckpt)
    teacher_state = {k[len("base_adapter."):]: v.to(peft_model.device) for k, v in sd.items() if k.startswith("base_adapter.")}

    set_peft_model_state_dict(peft_model, teacher_state, adapter_name="teacher")
    peft_model.set_adapter("teacher")
    model = peft_model.to(device).eval()

    gen_length = 256
    steps = 128
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")

        m = [{"role": "user", "content": user_input}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        # Each question is independent - no conversation history
        prompt = input_ids

        out = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=32, temperature=1.0, cfg_scale=0., remasking='low_confidence')

        answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")

        # Memory cleared for next question - each question is independent
        conversation_num += 1
        print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    chat()

