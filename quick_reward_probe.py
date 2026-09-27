import torch
import transformers

from GRPO import RewardModelV2

model_id = "Qwen/Qwen3-0.6B"
prompt = (
    "What is 10+15? You must show your work inside <think> and </think> tags, "
    "and put your final answer in brackets like [answer]."
)
truth = "25"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map=device, dtype=dtype)
reward_model = RewardModelV2(tokenizer)

encoded = tokenizer(prompt, return_tensors="pt")
model_device = next(model.parameters()).device
encoded = {k: v.to(model_device) for k, v in encoded.items()}
prompt_len = encoded["input_ids"].shape[1]

with torch.no_grad():
    outputs = model.generate(
        **encoded,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=4,
    )

rewards = reward_model.calculate_reward(outputs[:, prompt_len - 1 :], truth)

print(f"device={device}")
for i, out in enumerate(outputs):
    completion = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
    print(f"--- completion {i} ---")
    print(completion)
    print(f"reward={float(rewards[i].item()):.3f}")

print(f"avg_reward={float(rewards.float().mean().item()):.3f}")
