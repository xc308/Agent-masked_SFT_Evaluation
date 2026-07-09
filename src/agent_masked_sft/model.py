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


def save_model_and_tokenizer(model, tokenizer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved model and tokenizer to {output_dir}")



def load_saved_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=TORCH_DTYPE,
    )

    model.to(DEVICE)
    model.eval()

    return model, tokenizer