from typing import List, Dict

import torch
from agent_masked_sft.dialogue_data import CONTEXT_PROMPT
from agent_masked_sft.config import IGNORE_INDEX



def tokenize_dialogue(turns: List[Dict[str, str]], tokenizer):
    """Tokenize a dialogue and return input_ids plus two label tensors.

    Returns dict with:
      input_ids:              [L]  full token ids
      labels_standard:        [L]  tensor.context masked; both user+agent supervised
      labels_interventional:  [L]  tensor.context masked AND agent tokens masked
    """
    # Tokenize the context first to know its length
    ctx_ids = tokenizer(CONTEXT_PROMPT, add_special_tokens=False)["input_ids"]
    pieces = [("context", ctx_ids)]

    for t in turns:
        # split each turn into the role-prefix and the content so
        # the *role label* itself is part of "what the world says" — only
        # the *content* tokens after "Agent: " are agent interventions.
        # The role prefix (`User: ` / `Agent: `) is decided by the protocol,
        # not by either party. both prefixes are supervised in both SFT variants.
        prefix = "User: " if t["role"] == "user" else "Agent: "
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]

        content_ids = tokenizer(t["text"] + "\n", add_special_tokens=False)["input_ids"]

        pieces.append(("prefix", prefix_ids))
        pieces.append((t["role"], content_ids))  # 'user' or 'agent'

    input_ids = []
    role_per_token = []  # one tag per token, used to construct labels
    for tag, ids in pieces:
        input_ids.extend(ids)
        role_per_token.extend([tag] * len(ids))

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels_standard = input_ids.clone()
    labels_interventional = input_ids.clone()

    for i, tag in enumerate(role_per_token):
        if tag == "context":
            labels_standard[i] = IGNORE_INDEX
            labels_interventional[i] = IGNORE_INDEX
        elif tag == "agent":
            # Standard SFT: still supervise (this is the bug the paper calls out).
            # Interventional SFT: mask out — agent's own tokens are interventions.
            labels_interventional[i] = IGNORE_INDEX
        # 'prefix' and 'user' stay supervised in both

    return {
        "input_ids": input_ids,
        "labels_standard": labels_standard,
        "labels_interventional": labels_interventional,
        "role_per_token": role_per_token,
    }
