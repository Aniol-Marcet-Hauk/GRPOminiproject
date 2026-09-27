from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from GRPO import GRPOTrainer, RewardModelV1, RewardModelV2


class RewardTokenizerStub:
    """Tokenizer stub for deterministic RewardModelV1 tests."""

    def decode(self, output_tensor, skip_special_tokens=True):
        token_id = int(output_tensor[0].item())
        if token_id == 1:
            return "<think>work</think>[42]"
        if token_id == 2:
            return "plain text [42]"
        return "<think>broken format"


class RewardTokenizerV2Stub:
    """Tokenizer stub for deterministic RewardModelV2 tests."""

    def decode(self, output_tensor, skip_special_tokens=True):
        token_id = int(output_tensor[0].item())
        if token_id == 11:
            return "<think>steps</think> final answer is [1/2]"
        if token_id == 12:
            return "<think>steps</think> final answer is [x^2+1]"
        if token_id == 13:
            return "no think tags but answer [1/2]"
        return "<think>broken format only"


class FakeTokenizer:
    """Minimal tokenizer used by GRPOTrainer tests."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        _ = text
        _ = return_tensors
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


class FakeModel(nn.Module):
    """Tiny differentiable causal LM stub for fast trainer tests."""

    def __init__(self, vocab_size=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = nn.Linear(vocab_size, vocab_size, bias=False)

    def forward(self, input_ids):
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=self.vocab_size).float()
        logits = self.proj(one_hot)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, max_new_tokens, do_sample, temperature, num_return_sequences):
        repeated = input_ids.repeat_interleave(num_return_sequences, dim=0)
        continuation = torch.full((repeated.shape[0], 2), 4, dtype=torch.long, device=repeated.device)
        return torch.cat([repeated, continuation], dim=1)


class ConstantRewardModel:
    def calculate_reward(self, outputs_list, correct_answer):
        _ = correct_answer
        return torch.ones(outputs_list.shape[0], dtype=torch.float32)


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(
        "GRPO.transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "GRPO.transformers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )

    return GRPOTrainer(
        model_name="dummy-model",
        reward_model=ConstantRewardModel(),
        epochs=1,
        accumulation_steps=2,
        num_return_seq=2,
        max_new_tokens=2,
        device="cpu",
    )


def test_reward_scoring_expected():
    reward_model = RewardModelV1(tokenizer=RewardTokenizerStub())
    outputs = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
    rewards = reward_model.calculate_reward(outputs, "42")

    expected = torch.tensor([2.0, 1.0, 0.0])
    assert torch.allclose(rewards.float(), expected)


def test_reward_v2_fraction_numeric_equivalence_and_format():
    reward_model = RewardModelV2(tokenizer=RewardTokenizerV2Stub())
    outputs = [torch.tensor([11]), torch.tensor([13]), torch.tensor([99])]
    rewards = reward_model.calculate_reward(outputs, "0.5")

    expected = torch.tensor([2.0, 1.0, 0.2])
    assert torch.allclose(rewards.float(), expected, atol=1e-6)


def test_reward_v2_symbolic_bracket_match():
    reward_model = RewardModelV2(tokenizer=RewardTokenizerV2Stub())
    outputs = [torch.tensor([12]), torch.tensor([99])]
    rewards = reward_model.calculate_reward(outputs, "x^2+1")

    expected = torch.tensor([1.0, 0.2])
    assert torch.allclose(rewards.float(), expected, atol=1e-6)


def test_calc_log_probs_shape(trainer):
    input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long)
    log_probs = trainer._calc_log_probs(trainer.active_pol, input_ids)
    assert log_probs.shape == (2, 3)


def test_calc_advantage_zero_mean(trainer):
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    advantages = trainer._calc_advantadge(rewards)
    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-6)


def test_calc_mask(trainer):
    outputs = torch.tensor(
        [
            [5, 6, 0, 0],
            [7, 8, 9, 0],
        ],
        dtype=torch.long,
    )
    prompt_lengths = torch.tensor([2, 1], dtype=torch.long)
    mask = trainer._calc_mask(outputs, prompt_lengths, pad_token_id=0)

    expected = torch.tensor(
        [
            [False, False, False, False],
            [False, True, True, False],
        ]
    )
    assert torch.equal(mask.cpu(), expected)


def test_kl_divergence_non_negative(trainer):
    active = torch.tensor([[0.0, -0.5], [-0.2, -0.3]])
    ref = torch.tensor([[0.0, -0.5], [-0.2, -0.3]])
    kl = trainer._kl_divergence(active, ref)
    assert torch.all(kl >= 0.0)
    assert torch.allclose(kl, torch.zeros_like(kl))


def test_train_step_returns_loss_and_reward_tuple(trainer):
    result = trainer.train_step("test prompt", "42", step_idx=0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
