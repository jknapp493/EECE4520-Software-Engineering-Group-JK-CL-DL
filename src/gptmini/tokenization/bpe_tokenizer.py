from transformers import GPT2TokenizerFast


class HuggingFaceBPETokenizer:
    """
    Thin wrapper around GPT2 BPE tokenizer so it can be injected & swapped later.
    """
    def __init__(self, pretrained_name: str = "gpt2"):
        self._tok = GPT2TokenizerFast.from_pretrained(pretrained_name)
        if self._tok.pad_token is None:
            self._tok.add_special_tokens({'pad_token': '<pad>'})


    @property
    def impl(self) -> GPT2TokenizerFast:
        return self._tok


    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size


    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id or 0


    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id or 0


    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)


    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)