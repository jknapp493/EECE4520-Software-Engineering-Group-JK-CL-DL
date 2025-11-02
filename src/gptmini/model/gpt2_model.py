from transformers import GPT2Config, GPT2LMHeadModel


def build_model(tokenizer, n_layer: int, n_head: int, n_embd: int, n_positions: int):
    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=n_positions, n_ctx=n_positions,
        n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        pad_token_id=tokenizer.pad_token_id
    )
    model = GPT2LMHeadModel(cfg)
    model.resize_token_embeddings(tokenizer.vocab_size)
    return model