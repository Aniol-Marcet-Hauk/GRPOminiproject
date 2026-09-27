import torch
import transformers
import gc
import re

class RewardRuleModelSuper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.debug_list_rewards = []
    def calculate_reward( self,outputs_list: torch.Tensor, 
                            correct_answer: str) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented for any class inheriting from RewardRuleModelSuper")

class RewardModelV1(RewardRuleModelSuper):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def calculate_reward(self, outputs_list, correct_answer):
        rewards_list = []
        target_string = f"[{correct_answer}]"
        for output_tensor in outputs_list:
            generated_text = self.tokenizer.decode(output_tensor, skip_special_tokens=True)
            
            score = 0.0
            
            if target_string in generated_text.replace(" ", ""):
                score += 1.0
            
            think_start_pos = generated_text.find("<think>")
            think_end_pos = generated_text.find("</think>")
            has_good_format = (think_start_pos != -1 and think_end_pos != -1 and think_start_pos < think_end_pos)
            if has_good_format:
                score+=1    
            rewards_list.append(score)

        return torch.tensor(rewards_list)


class RewardModelV2(RewardRuleModelSuper):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def _normalize_text(self, value: str) -> str:
        text = value.strip().lower()
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.replace("−", "-")
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace("\\left", "")
        text = text.replace("\\right", "")
        text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
        text = text.replace("{", "")
        text = text.replace("}", "")
        text = re.sub(r"\s+", "", text)
        return text

    def _to_number(self, value: str):
        if not value:
            return None
        numeric = value
        if re.fullmatch(r"[+-]?\d+/\d+", numeric):
            numerator, denominator = numeric.split("/", 1)
            denominator_i = int(denominator)
            if denominator_i == 0:
                return None
            return float(int(numerator) / denominator_i)
        allowed_chars = set("+-0123456789.")
        if set(numeric).issubset(allowed_chars) and any(ch.isdigit() for ch in numeric):
            if numeric.count(".") <= 1 and numeric.count("+") <= 1 and numeric.count("-") <= 1:
                try:
                    return float(numeric)
                except ValueError:
                    return None
        return None

    def _last_number_value(self, normalized_text: str):
        number_like_tokens = re.findall(r"[+-]?\d+(?:\.\d+)?(?:/[+-]?\d+)?", normalized_text)
        if not number_like_tokens:
            return None
        return self._to_number(number_like_tokens[-1])

    def _is_correct_answer(self, generated_text: str, correct_answer: str) -> bool:
        normalized_target = self._normalize_text(correct_answer)
        normalized_generated = self._normalize_text(generated_text)

        if normalized_generated == normalized_target:
            return True

        target_num = self._to_number(normalized_target)

        generated_num = self._to_number(normalized_generated)
        if generated_num is not None and target_num is not None:
            if abs(generated_num - target_num) <= 1e-9:
                return True

        if target_num is not None:
            token_num = self._last_number_value(normalized_generated)
            if token_num is not None and abs(token_num - target_num) <= 1e-9:
                return True
        return False

    def _approx_numeric_bonus(self, generated_text: str, correct_answer: str) -> float:
        normalized_target = self._normalize_text(correct_answer)
        target_num = self._to_number(normalized_target)
        if target_num is None:
            return 0.0

        normalized_generated = self._normalize_text(generated_text)
        predicted_num = self._last_number_value(normalized_generated)
        if predicted_num is None:
            return 0.0

        abs_error = abs(predicted_num - target_num)
        scale = max(1.0, abs(target_num))
        relative_error = abs_error / scale

        # Reward only genuinely close answers; far numbers get no bonus.
        if relative_error > 0.25:
            return 0.0

        return 0.3 * (1.0 - (relative_error / 0.25))

    def calculate_reward(self, outputs_list, correct_answer):
        rewards_list = []
        for output_tensor in outputs_list:
            generated_text = self.tokenizer.decode(output_tensor, skip_special_tokens=True)
            score = 0.0

            if self._is_correct_answer(generated_text, correct_answer):
                score += 1.0
            else:
                score += self._approx_numeric_bonus(generated_text, correct_answer)

            think_start_pos = generated_text.find("<think>")
            think_end_pos = generated_text.find("</think>")
            has_good_format = (
                think_start_pos != -1 and think_end_pos != -1 and think_start_pos < think_end_pos
            )
            if has_good_format:
                score += 1.0


            if not has_good_format:
                if "<think>" in generated_text:
                    score += 0.2
                if "</think>" in generated_text:
                    score += 0.2

            rewards_list.append(score)

        return torch.tensor(rewards_list)


class RewardModelV3(RewardModelV2):

    def calculate_reward(self, outputs_list, correct_answer):
        rewards_list = []
        for output_tensor in outputs_list:
            generated_text = self.tokenizer.decode(output_tensor, skip_special_tokens=True)
            score = 0.0

            if self._is_correct_answer(generated_text, correct_answer):
                score += 1.0

            think_start_pos = generated_text.find("<think>")
            think_end_pos = generated_text.find("</think>")
            has_good_format = (
                think_start_pos != -1 and think_end_pos != -1 and think_start_pos < think_end_pos
            )
            if has_good_format:
                score += 1.0

            if not has_good_format:
                if "<think>" in generated_text:
                    score += 0.2
                if "</think>" in generated_text:
                    score += 0.2

            rewards_list.append(score)

        return torch.tensor(rewards_list)


class GRPOTrainer:
    def __init__( self, model_name: str, reward_model, lr: float = 1e-6,
        weight_decay: float = 0.01, beta: float = 0.04, epsilon: float = 0.2,
        epochs: int = 3, accumulation_steps: int = 8, temperature: float = 0.8,
        num_return_seq : int = 4, max_new_tokens: int = 50, device: str = "cpu",
        memory_saving: bool = True):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    
        self.active_pol = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, dtype=model_dtype)
        self.ref_pol = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, dtype=model_dtype)

        if memory_saving and hasattr(self.active_pol, "gradient_checkpointing_enable"):
            self.active_pol.gradient_checkpointing_enable()
        use_cache = not memory_saving
        self.active_pol.config.use_cache = use_cache
        self.ref_pol.config.use_cache = use_cache
        self.ref_pol.eval()

        self.optimizer = torch.optim.AdamW( self.active_pol.parameters(), lr=lr, weight_decay=weight_decay)
        self.reward_model = reward_model
        self.device = device

    
        self.beta = beta
        self.epsilon= epsilon
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.temperature = temperature
        self.num_return_seq = num_return_seq
        self.max_new_tokens = max_new_tokens

    def _calc_log_probs(self, model, input_ids: torch.Tensor) -> torch.Tensor:

        logits = model(input_ids).logits[:,:-1,:]
        log_prob_all = logits.log_softmax(dim=-1)
        input_ids_correct_dim = input_ids[:,1:].unsqueeze(-1)

        log_prob_generated = log_prob_all.gather(dim=-1,
                                                index=input_ids_correct_dim).squeeze(-1)
        return log_prob_generated
    
    def _calc_advantadge(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.numel() <2:
            return torch.zeros_like(rewards)
        return (rewards- rewards.mean())/(rewards.std()+1e-8)

    def _calc_mask(self,outputs: torch.Tensor, prompt_lengths: torch.Tensor,pad_token_id: int) -> torch.BoolTensor:
       
        _ , seq_len = outputs.shape 
        positions = torch.arange(seq_len, device=outputs.device).unsqueeze(0)
        
        is_completion = positions >= prompt_lengths.unsqueeze(1)
        is_not_pad = outputs != pad_token_id
        return is_completion & is_not_pad

    def _kl_divergence(self, active_log_prob: torch.Tensor, ref_log_prob: torch.Tensor) -> torch.Tensor:
        return torch.exp(ref_log_prob-active_log_prob) - (ref_log_prob-active_log_prob) -1.0


    def train_step(self, prompt: str, truth: str, step_idx: int) -> tuple[float, float]:
        encoded_input = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.active_pol.parameters()).device
        encoded_input = {k: v.to(model_device) for k, v in encoded_input.items()}
        prompt_len = encoded_input["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.active_pol.generate( **encoded_input, max_new_tokens=self.max_new_tokens,
                                                    do_sample=True, temperature=self.temperature,
                                                    num_return_sequences=self.num_return_seq)

        rewards = self.reward_model.calculate_reward(outputs[:,prompt_len-1:], truth).to(model_device)
        advantadges = self._calc_advantadge(rewards).unsqueeze(-1)
        with torch.no_grad():
            old_log_probs = self._calc_log_probs(self.active_pol, outputs)
            ref_log_probs = self._calc_log_probs(self.ref_pol, outputs)

        prompt_lengths = torch.full( (outputs.shape[0],), prompt_len - 1, dtype=torch.long, device=outputs.device)
        mask = self._calc_mask(outputs[:,1:], prompt_lengths, pad_token_id=self.tokenizer.pad_token_id)
        self.accumulated_loss = 0.0  
        loss = torch.tensor(0.0, device=model_device)
        for _ in range(self.epochs):

            
            #print(f"epoch: {epoch}\n")
            active_log_probs = self._calc_log_probs(self.active_pol,outputs)

            ratio = torch.exp(active_log_probs - old_log_probs)
            

            policy_gradient = torch.min( ratio * advantadges,
                                        torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantadges)
    
            kl_div = self._kl_divergence(active_log_probs,ref_log_probs)
            objective = policy_gradient - (self.beta * kl_div)
            #print(f"Kl divergence:{kl_div} , {kl_div.sum()}")

            masked_objective = objective * mask
            denom = mask.sum().clamp_min(1)
            loss = -masked_objective.sum() / denom / self.accumulation_steps
            if torch.isfinite(loss):
                loss.backward()

        if (step_idx + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad() 
        #the gpu variables needs to be emptied
        del old_log_probs, ref_log_probs, mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return float(loss.detach().item()), float(rewards.mean().item())


    def save_checkpoint(self, path: str):
        self.active_pol.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


