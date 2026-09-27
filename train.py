
import argparse
from datasets import load_dataset
import transformers
import torch
import matplotlib.pyplot as plt
from GRPO import GRPOTrainer, RewardModelV2
from simple_dataset import build_simple_dataset

# should probably use a stronger model, and a causal one if that, but my computer can't handle a
MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "zwhe99/DeepMath-103K"
# this is a horrible system prompt to be clear, but since i didn't use a causal model and the model is very small this was the best i could do
#system_prompt = "You must show your work inside <think> and </think> tags. The answer is"
system_prompt = " The answer is"

STEP_TO_SAVE_MODEL = 500

def parse_args():
    parser = argparse.ArgumentParser(description="Train GRPO with configurable model and dataset.")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset id")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum training steps")
    parser.add_argument("--eval-every", type=int, default=2, help="Evaluate every N steps")
    parser.add_argument("--eval-sample-size", type=int, default=8, help="Examples per split for each evaluation")
    parser.add_argument("--eval-num-return-seq", type=int, default=1,
        help="Number of generated samples per prompt during evaluation")
    parser.add_argument("--eval-max-new-tokens", type=int, default=32,
        help="Max generated tokens per prompt during evaluation")
    parser.add_argument("--simple-dataset", action="store_true",
        help="Use a built-in arithmetic toy dataset instead of downloading a HF dataset")
    parser.add_argument("--simple-size", type=int, default=200,
            help="Number of rows in the built-in arithmetic toy dataset")
    parser.add_argument("--train-num-return-seq", type=int, default=1,
        help="Number of generated samples per prompt during training")
    parser.add_argument("--train-max-new-tokens", type=int, default=32,
        help="Max generated tokens per prompt during training")
    parser.add_argument("--train-epochs", type=int, default=1,
        help="PPO update epochs per training step")
    parser.add_argument("--accumulation-steps", type=int, default=8,
        help="Gradient accumulation steps")
    parser.add_argument("--no-memory-saving", dest="memory_saving",
        action="store_false", default=True,
        help="Disable memory-saving mode (faster, but higher VRAM use, no gradient checkpointing + cache on)")
    return parser.parse_args()


def get_problem_and_answer(row):
    if "problem" in row and "answer" in row:
        return row["problem"], str(row["answer"])
    if "question" in row and "final_answer" in row:
        return row["question"], str(row["final_answer"])
    raise KeyError(
        "Unsupported dataset schema. Expected either (problem, answer) or (question, final_answer).")


def evaluate_average_reward( trainer, data_split, sample_size=64, seed=0,
                            eval_num_return_seq=1, eval_max_new_tokens=32):
    sample_size = min(sample_size, len(data_split))
    sampled_split = data_split.shuffle(seed=seed).select(range(sample_size))
    rewards = []

    with torch.no_grad():
        for row in sampled_split:
            problem_text, answer_text = get_problem_and_answer(row)
            prompt = f"{problem_text}\n{system_prompt}"
            encoded_input = trainer.tokenizer(prompt, return_tensors="pt")
            model_device = next(trainer.active_pol.parameters()).device
            encoded_input = {k: v.to(model_device) for k, v in encoded_input.items()}
            prompt_len = encoded_input["input_ids"].shape[1]

            outputs = trainer.active_pol.generate(
                **encoded_input,
                max_new_tokens=eval_max_new_tokens,
                do_sample=True,
                temperature=trainer.temperature,
                num_return_sequences=eval_num_return_seq)

            reward_tensor = trainer.reward_model.calculate_reward(
                outputs[:, prompt_len - 1 :], answer_text)
            rewards.append(float(reward_tensor.mean().item()))

            del outputs, encoded_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return sum(rewards) / len(rewards) if rewards else 0.0


def plot_training_vs_test(eval_steps, train_rewards, test_rewards, output_path="train_test_reward_curve.png"):
    plt.figure(figsize=(11, 5))
    plt.plot(eval_steps, train_rewards, marker="o", label="Train reward")
    plt.plot(eval_steps, test_rewards, marker="o", label="Test reward")
    plt.title("Reward Improvement During Training")
    plt.xlabel("Training step")
    plt.ylabel("Average reward")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved graph to: {output_path}")


def unpack_train_result(train_result):
    if isinstance(train_result, tuple) and len(train_result) == 2:
        return train_result
    return float("nan"), float("nan")


def run_training_loop( trainer, train_dataset, test_dataset, eval_every=100,
                    max_steps=1000, eval_sample_size=64, eval_num_return_seq=1, eval_max_new_tokens=32,):
    eval_steps = []
    train_rewards = []
    test_rewards = []

    for step, row in enumerate(train_dataset):
        if step >= max_steps:
            break

        problem_text, answer_text = get_problem_and_answer(row)
        prompt = f"{problem_text}\n{system_prompt}"
        train_result = trainer.train_step(prompt, answer_text, step_idx=step)
        #loss, avg_reward = unpack_train_result(train_result)

        # if step % trainer.accumulation_steps == 0:
        #     print(f"Step {step:05d} | Loss: {loss:.6f} | Group Reward: {avg_reward:.6f}")
        
     
        if step % eval_every == 0 and step > 0:
            avg_train_reward = evaluate_average_reward(
                trainer,
                train_dataset,
                sample_size=eval_sample_size,
                seed=step,
                eval_num_return_seq=eval_num_return_seq,
                eval_max_new_tokens=eval_max_new_tokens,
            )
            avg_test_reward = evaluate_average_reward(
                trainer,
                test_dataset,
                sample_size=eval_sample_size,
                seed=step + 1,
                eval_num_return_seq=eval_num_return_seq,
                eval_max_new_tokens=eval_max_new_tokens,
            )

            eval_steps.append(step)
            train_rewards.append(avg_train_reward)
            test_rewards.append(avg_test_reward)

            print(f"Eval step {step:05d},  Train reward: {avg_train_reward:.6f} , "
                f"Test reward: {avg_test_reward:.6f}, Gap: {avg_train_reward - avg_test_reward:.6f}" )

        if step % STEP_TO_SAVE_MODEL == 0 and step > 0:
            trainer.save_checkpoint(f"./grpo_step_{step}")

    if eval_steps:
        plot_training_vs_test(eval_steps, train_rewards, test_rewards)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    reward_model = RewardModelV2(tokenizer=tokenizer)

    trainer = GRPOTrainer(
        model_name=args.model,
        reward_model=reward_model,
        lr=1e-6,
        accumulation_steps=args.accumulation_steps,
        epochs=args.train_epochs,
        num_return_seq=args.train_num_return_seq,
        max_new_tokens=args.train_max_new_tokens,
        memory_saving=args.memory_saving,
        device=device,)
    print(f"Using device: {device}")
    if args.simple_dataset:
        dataset = build_simple_dataset(num_examples=args.simple_size)
    else:
        dataset = load_dataset(args.dataset, split="train")

    split_dataset = dataset.train_test_split(test_size=0.1, seed=3221, shuffle=True)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    trainer.optimizer.zero_grad()
    run_training_loop(trainer, train_dataset, test_dataset,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        eval_sample_size=args.eval_sample_size,
        eval_num_return_seq=args.eval_num_return_seq,
        eval_max_new_tokens=args.eval_max_new_tokens,
    )
    trainer.save_checkpoint("./grpo_final")


if __name__ == "__main__":
    main()
