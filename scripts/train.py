import os
import torch
from gptmini.config import ModelConfig
from gptmini.tokenization import HuggingFaceBPETokenizer
from gptmini.data import load_texts, build_token_blocks, fix_out_of_range, make_splits
from gptmini.model import build_model
from gptmini.training import Trainer


def main():
    cfg = ModelConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    tok = HuggingFaceBPETokenizer()
    texts = load_texts(cfg.nltk_limit, cfg.c4_limit)
    blocks = build_token_blocks(tok, texts, cfg.block_size)

    if not blocks:
        raise RuntimeError("No token blocks created. Try smaller block_size or more text.")

    vocab_size = tok.vocab_size
    max_id = max((max(b) if b else 0) for b in blocks)
    if max_id >= vocab_size or max_id < 0:
        blocks = fix_out_of_range(blocks, vocab_size, tok.eos_token_id)

    train_dl, val_dl, test_dl = make_splits(blocks, cfg.batch_size, tok.pad_token_id, cfg.num_workers)
    model = build_model(tok, cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size)
    trainer = Trainer(model, device, cfg.lr, cfg.out_dir, cfg.use_amp)
    trainer.fit(train_dl, val_dl, cfg.epochs)

if __name__ == "__main__":
    main()