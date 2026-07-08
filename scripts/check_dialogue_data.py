
from agent_masked_sft.dialogue_data import build_dialogue_dataset

def main():
    dialogues = build_dialogue_dataset(n_dialogues=3, seed=0)

    print("numb of dialogues:", len(dialogues))

    for i, dia in enumerate(dialogues):
        print("="*80)
        print(f"Dialogue {i + 1}")
        print("="*80)
        print(dia["text"])


if __name__ == "__main__":
    main()