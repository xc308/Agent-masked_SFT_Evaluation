import torch

SEED = 0
N_DIALOGUES = 6
IGNORE_INDEX = -100
BATCH_SIZE = 1


NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
LOG_EVERY = 1
GRAD_CLIP_NORM = 1.0


MODEL_NAME = "Qwen/Qwen2.5-0.5B"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

TORCH_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

