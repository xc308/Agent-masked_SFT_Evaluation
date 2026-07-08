from typing import List, Dict

from torch.utils.data import Dataset, DataLoader
import torch

from agent_masked_sft.config import IGNORE_INDEX, BATCH_SIZE


class DialogueDataset(Dataset):
    """
    PyTorch Dataset for tokenized dialogues.

    Each item returns:
        input_ids: token ids
        labels: either standard SFT labels or interventional SFT labels
    """

    def __init__(
        self,
        tokenized_dialogues: List[Dict],
        label_key: str,
    ):
        self.data = tokenized_dialogues
        self.label_key = label_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        return {
            "input_ids": example["input_ids"],
            "labels": example[self.label_key],
        }


def collate(batch, pad_id: int):
    """
    Right-pad input_ids and labels to the maximum length in the batch.

    input_ids are padded with pad_id.
    labels are padded with IGNORE_INDEX, so padding tokens do not contribute to loss.
    attention_mask is 1 for real tokens and 0 for padding.
    """
    max_len = max(example["input_ids"].size(0) for example in batch)

    input_ids = torch.full(
        (len(batch), max_len),
        pad_id,
        dtype=torch.long,
    )

    labels = torch.full(
        (len(batch), max_len),
        IGNORE_INDEX,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (len(batch), max_len),
        dtype=torch.long,
    )

    for i, example in enumerate(batch):
        length = example["input_ids"].size(0)

        input_ids[i, :length] = example["input_ids"]
        labels[i, :length] = example["labels"]
        attention_mask[i, :length] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def build_dataloader(
        tokenized_dialogues: List[Dict],
        label_key: str,
        batch_size:int = BATCH_SIZE,
        shuffle: bool=True, 
        pad_id = int     
) -> DataLoader:
    """
    Build a DataLoader for either standard SFT or agent-masked/interventional SFT.
    """
    dataset = DialogueDataset(
        tokenized_dialogues=tokenized_dialogues,
        label_key=label_key,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate(batch, pad_id),
    )