import torch
import torch.nn.functional as F
from typing import Dict

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from agent_masked_sft.config import DEVICE
from agent_masked_sft.dialogue_data import CONTEXT_PROMPT



@torch.no_grad()
def agent_followup(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    user_text: str,
    max_new_tokens: int = 120,
) -> str:
    """
    Free-generate the agent's follow-up response to a user statement.
    """
    model.eval()

    prompt = CONTEXT_PROMPT + f"User: {user_text}\nAgent: "

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt_length = inputs["input_ids"].size(1)
    generated_ids = output_ids[0][prompt_length:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    for stop in ["\nUser:", "\nAgent:", "\n"]:
        if stop in generated_text:
            generated_text = generated_text.split(stop)[0]
            break

    return generated_text.strip()


@torch.no_grad()
def score_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    fact: Dict[str, str],
) -> Dict[str, float]:
    """
    Score true vs false candidate completions using average per-token log probability.

    Returns:
        {
            "true": average per-token log probability of the true completion,
            "false": average per-token log probability of the false completion,
        }
    """
    model.eval()

    topic = fact["topic"]
    prompt = CONTEXT_PROMPT + f"User: Tell me about {topic}.\nAgent: "

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    results = {}

    for label, text in [
        ("true", fact["true"]),
        ("false", fact["false"]),
    ]:
        candidate_ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        full_ids_list = prompt_ids + candidate_ids

        full_ids = torch.tensor(
            [full_ids_list],
            dtype=torch.long,
            device=DEVICE,
        )

        attention_mask = torch.ones_like(full_ids)

        logits = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
        ).logits[0]

        targets = full_ids[0, len(prompt_ids):]

        start = len(prompt_ids) - 1
        end = start + len(candidate_ids)

        relevant_logits = logits[start:end]

        log_probs = F.log_softmax(
            relevant_logits,
            dim=-1,
        )

        target_token_log_probs = log_probs.gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)

        results[label] = target_token_log_probs.mean().item()

    return results