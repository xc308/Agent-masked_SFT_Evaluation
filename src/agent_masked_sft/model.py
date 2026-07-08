from transformers import AutoTokenizer, AutoModelForCausalLM

from agent_masked_sft.config import MODEL_NAME, TORCH_DTYPE, DEVICE


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_fresh_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=TORCH_DTYPE,
    )

    model.to(DEVICE)
    return model