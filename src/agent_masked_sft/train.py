import torch
from agent_masked_sft.config import DEVICE, IGNORE_INDEX


def train(model, dataloader, epochs = 2, lr = 1e-5, log_every = 10, tag = ""):
    model.train()                  # turn on model training mode, with dropout
    opt = torch.optim.AdamW(model.parameters(), lr = lr)
    history = []
    step = 0

    for ep in range(epochs):
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            if (batch["labels"] != IGNORE_INDEX).sum().item() == 0:
                continue

            out = model(**batch)    # unpack dict bactch in the model, get output
            loss = out.loss         # get loss fromt the output
            loss.backward()         # use loss to compute the gradient of all trainable parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip the gradient norm to at most 1
            opt.step()              # update weight parameter
            opt.zero_grad()         # clear gradient for next batch

            history.append(loss.item()) # collect scalar loss value into history list

            if step % log_every == 0:
                print(f"[{tag}] epoch {ep} step {step:4d}  loss = {loss.item():.4f}")
            step += 1

    return history
