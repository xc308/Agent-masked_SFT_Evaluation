from agent_masked_sft.model import load_tokenizer 
from agent_masked_sft.dialogue_data import build_dialogue_dataset
from agent_masked_sft.tokenization import tokenize_dialogue
from agent_masked_sft.dataset_dataloader import build_dataloader


def main():

    tokenizer = load_tokenizer()

    dialogues = build_dialogue_dataset(n_dialogues=6, seed=0)

    tokenized = [tokenize_dialogue(turns = diag["turns"], tokenizer=tokenizer) for diag in dialogues]

    dl_standard = build_dataloader(
        tokenized_dialogues=tokenized,
        label_key="labels_standard",
        batch_size = 2,
        shuffle = True,
        pad_id = tokenizer.pad_token_id
    )

    dl_interventional = build_dataloader(
        tokenized_dialogues=tokenized,
        label_key= "labels_interventional",
        batch_size = 2,
        shuffle = True,
        pad_id = tokenizer.pad_token_id
    )

    batch_standard = next(iter(dl_standard))
    batch_interventional = next(iter(dl_interventional))

    print("Standard SFT batch")
    print("input_ids:", batch_standard["input_ids"].shape)
    print("labels:", batch_standard["labels"].shape)
    print("attention_mask:", batch_standard["attention_mask"].shape)
    print()

    print("Interventional / agent-masked SFT batch")
    print("input_ids:", batch_interventional["input_ids"].shape)
    print("labels:", batch_interventional["labels"].shape)
    print("attention_mask:", batch_interventional["attention_mask"].shape)
    print()

    print("Standard input_ids first row:")
    print(batch_standard["input_ids"][0])
    print()

    print("Standard labels first row:")
    print(batch_standard["labels"][0])
    print()

    print("Standard attention_mask first row:")
    print(batch_standard["attention_mask"][0])


if __name__ == "__main__":
    main()


