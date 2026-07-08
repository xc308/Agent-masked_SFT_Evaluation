import random 
from typing import Dict, List

from agent_masked_sft.config import SEED, N_DIALOGUES
from agent_masked_sft.fact_bank import (FACTS, CONTEXT_PROMPT, CORRECTION_TEMPLATES, CONFIRMATION_TEMPLATES, 
                                        P_LIE, P_CHITCHAT, AGENT_ACK_AFTER_CONFIRMATION, AGENT_ACK_AFTER_CORRECTION, 
                                        USER_QUESTION_TEMPLATES, CHITCHAT_TEMPLATES)


def _lc_first(s: str) -> str:
    """ lower the first char only, for connecting different sentences """
    return s[0].lower() + s[1:] if s else s


rng = random.Random(SEED)

def build_dialogue(fact: dict, p_lie: float, rng: random.Random) -> List[Dict[str, str]]:
    """Return a list of turns: [{'role': 'user'|'agent', 'text': ...}, ...]."""
    turns = []
    # Turn 1: User asks about the topic.
    q = rng.choice(USER_QUESTION_TEMPLATES).format(topic=fact["topic"])
    turns.append({"role": "user", "text": q})

    # Turn 2: Agent answers, possibly wrong.
    agent_lies = rng.random() < p_lie
    if agent_lies:
        turns.append({"role": "agent", "text": fact["false"]})

        # Turn 3: User corrects (sampled phrasing).
        tpl = rng.choice(CORRECTION_TEMPLATES)
        turns.append({"role": "user",
                      "text": tpl.format(truth=fact["true"], truth_lc=_lc_first(fact["true"]))})

        # Turn 4: Agent acknowledges (sampled phrasing).
        turns.append({"role": "agent", "text": rng.choice(AGENT_ACK_AFTER_CORRECTION)})
    else:
        turns.append({"role": "agent", "text": fact["true"]})

        # Turn 3: User confirms (sampled phrasing). Every confirmation template
        # carries the truth content — see CONFIRMATION_TEMPLATES.
        tpl = rng.choice(CONFIRMATION_TEMPLATES)
        turns.append({"role": "user",
                      "text": tpl.format(truth=fact["true"], truth_lc=_lc_first(fact["true"]))})

        # Turn 4: Agent acknowledges (sampled phrasing).
        turns.append({"role": "agent", "text": rng.choice(AGENT_ACK_AFTER_CONFIRMATION)})

    # Optionally append a chit-chat tail for surface variety.
    if rng.random() < P_CHITCHAT:
        chit = rng.choice(CHITCHAT_TEMPLATES)
        for role, text in chit:
            turns.append({"role": role, "text": text})

    return turns # list of dict containing role and text


def render_dialogue(turns: List[Dict[str, str]]) -> str:
    """render content of turns of dialogue as a flat whole string. Each turn ends with a newline."""

    out = CONTEXT_PROMPT
    for t in turns:
        prefix = "User: " if t["role"] == "user" else "Agent: "
        out += prefix + t["text"] + "\n"
    return out      # a single string of role and text


def build_dialogue_dataset(
    n_dialogues: int = N_DIALOGUES,
    seed: int = SEED,
) -> List[Dict[str, object]]:
    """
    Build a list of synthetic dialogue examples.

    Each example has:
        - turns: structured list of turns
        - text: rendered dialogue string
    """
    rng = random.Random(seed)

    dialogues = []

    for _ in range(n_dialogues):
        fact = rng.choice(FACTS)
        turns = build_dialogue(fact, P_LIE, rng)

        dialogues.append(
            {
                "turns": turns,
                "text": render_dialogue(turns),
            }
        )

    return dialogues
