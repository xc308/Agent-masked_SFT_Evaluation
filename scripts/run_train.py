import gc # garbage collection for local laptop memory free
import torch




from agent_masked_sft.config import (N_DIALOGUES, SEED, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, LOG_EVERY, 
                                     STANDARD_MODEL_DIR, INTERVENTIONAL_MODEL_DIR)
from agent_masked_sft.utils import set_seed

from agent_masked_sft.dialogue_data import build_dialogue_dataset
from agent_masked_sft.model import load_tokenizer, load_fresh_model,save_model_and_tokenizer
from agent_masked_sft.tokenization import tokenize_dialogue
from agent_masked_sft.dataset_dataloader import build_dataloader
from agent_masked_sft.train import train


def clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()




def main():
    set_seed(SEED)

    tokenizer = load_tokenizer()

    dialogues = build_dialogue_dataset(
        n_dialogues=N_DIALOGUES,
        seed=SEED
    )

    tokenized = [tokenize_dialogue(diag["turns"], tokenizer=tokenizer) for diag in dialogues]

    dl_standard = build_dataloader(
        tokenized_dialogues=tokenized, 
        label_key="labels_standard",
        batch_size=BATCH_SIZE,
        pad_id=tokenizer.pad_token_id,
        shuffle=True
    )

    dl_interventional = build_dataloader(
        tokenized_dialogues=tokenized,
        label_key="labels_interventional",
        batch_size=BATCH_SIZE,
        pad_id=tokenizer.pad_token_id,
        shuffle = True
    )


    print("===== Standard SFT: all tokens supervised ========")
    model_std = load_fresh_model()

    hist_std = train(
        model=model_std,
        dataloader=dl_standard,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        log_every=LOG_EVERY,
        tag="STD"
    )

    save_model_and_tokenizer(
        model=model_std,
        tokenizer=tokenizer,
        output_dir=STANDARD_MODEL_DIR,
    )

    del model_std
    clear_memory()



    print("===== Agent-masked SFT: agent token masked =========")
    model_iv = load_fresh_model()

    hist_iv = train(
        model=model_iv,
        dataloader=dl_interventional,
        epochs=NUM_EPOCHS,
        lr = LEARNING_RATE,
        log_every=LOG_EVERY,
        tag="IV"
    )


    save_model_and_tokenizer(
        model=model_iv,
        tokenizer=tokenizer,
        output_dir=INTERVENTIONAL_MODEL_DIR,
    )


    print("Training finished.")
    print(f"Standard SFT final loss: {hist_std[-1]:.4f}")
    print(f"Interventional SFT final loss: {hist_iv[-1]:.4f}")



if __name__ == "__main__":
    main()


    







