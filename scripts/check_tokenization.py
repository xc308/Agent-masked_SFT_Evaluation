from agent_masked_sft.config import IGNORE_INDEX
from agent_masked_sft.model import load_tokenizer
from agent_masked_sft.dialogue_data import build_dialogue_dataset
from agent_masked_sft.tokenization import tokenize_dialogue

def main():

    tokenizer = load_tokenizer()
    dialogues = build_dialogue_dataset(n_dialogues=3, seed=0)

    tokenized = [tokenize_dialogue(d["turns"], tokenizer) for d in dialogues] # a list of dict

    ex = tokenized[0] # the first dict in the above list
    T = len(ex["input_ids"])

    n_std_ids = (ex["labels_standard"] != IGNORE_INDEX).sum().item()
    n_inv_ids = (ex["labels_interventional"] != IGNORE_INDEX).sum().item()

    print(f"The first dialogue has {T} tokens.\n")
    print(f" Standard SFT supervises {n_std_ids} / {T} tokens (only context ids masked.) \n")
    print(f" Interventional SFT supervises {n_inv_ids} / {T} tokens (both context ids and agent content ids are masked out)\n")


    print("\n The first 60 tokens of the first dialogue.")
    print("Colnums: token (decoded string), role (who wrote it)")
    print("         std / inv = 'loss' if this position contribute to the loss, else '------' ignored position")
    print(f"       {'token':>15}  {'role':>6}  {'std':>5}  {'inv':>6}")
    print("-" * 48)

    for i in range(min(60, T)):
        tok = tokenizer.decode([ex["input_ids"][i].item()])
        tag = ex["role_per_token"][i]
        s = "loss" if ex["labels_standard"][i] != IGNORE_INDEX else "------"
        iv = "loss" if ex["labels_interventional"][i] != IGNORE_INDEX else "------"
        print(f"   {repr(tok):>15}   {tag:>6}   {s:>5}    {iv:>6}")


if __name__ == "__main__":
    main()    

