import torch
import torch.nn.functional as F


class TopKSampler:
    def __init__(self, tokenizer, max_length=50, top_k=50, temperature=0.9, device="cuda"):
        self.tok = tokenizer
        self.max_length = max_length
        self.top_k = top_k
        self.temperature = temperature
        self.device = device


    @torch.no_grad()
    def generate(self, model, prompt: str) -> str:
        model.eval()
        input_ids = torch.tensor([self.tok.encode(prompt)], device=self.device)
        for _ in range(self.max_length):
            logits = model(input_ids=input_ids).logits[:, -1, :] / self.temperature
            probs = F.softmax(logits, dim=-1)
            top_probs, top_idx = torch.topk(probs, self.top_k)
            next_token = top_idx[0, torch.multinomial(top_probs[0], 1)]
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == self.tok.eos_token_id:
                break
        return self.tok.decode(input_ids[0].tolist())