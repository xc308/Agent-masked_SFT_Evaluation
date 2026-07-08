
from agent_masked_sft.config import SEED, DEVICE, MODEL_NAME
from agent_masked_sft.model import load_fresh_model, load_tokenizer
from agent_masked_sft.utils import set_seed


def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    print("Model:", MODEL_NAME)

    tokenizer = load_tokenizer()
    print("Tokenizer loaded.")
    print("Pad token:", tokenizer.pad_token)

    model = load_fresh_model()
    model = load_fresh_model()
    print("Model loaded.")
    print("Model device:", next(model.parameters()).device)
    

if __name__ == "__main__":
    main()