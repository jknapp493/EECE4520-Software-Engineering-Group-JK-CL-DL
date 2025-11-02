import os
import torch
from gptmini.config import ModelConfig
from gptmini.tokenization import HuggingFaceBPETokenizer
from gptmini.model import build_model
from gptmini.sampling import TopKSampler

def main():
    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = HuggingFaceBPETokenizer()
    model = build_model(tok, cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size).to(device)

    # optional checkpoint
    CKPT = "checkpoints/model_epoch_25.pth"  # or None
    if CKPT:
        if os.path.exists(CKPT):
            model.load_state_dict(torch.load(CKPT, map_location=device))
        else:
            print(f"[warn] checkpoint not found: {CKPT} — running with random weights")

    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In a distant galaxy",
    ]

    gen = TopKSampler(tok, device=device, max_length=50, top_k=50, temperature=0.9)
    i = 1
    for p in prompts:
        print(f'Prompt {i}: "{p}"') 
        print(f"{gen.generate(model, p)}")
        i += 1


if __name__ == "__main__":
    main()
