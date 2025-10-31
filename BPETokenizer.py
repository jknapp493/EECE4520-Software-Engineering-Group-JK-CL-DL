"""
GPT-2 Data Preprocessing, Tokenizer and Trainer
-------------------------------------------------
- Used Brown and Reuters sets from NLTK corpus (subwords and news articles)
- Used the Hugging Face Transformers mini c4 to train so it wouldn't be as much data to train on

Sources:
- Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers/python/latest/
- Hugging Face Transformers (GPT-2): https://huggingface.co/docs/transformers/model_doc/gpt2
- NLTK Corpora How-to: https://www.nltk.org/howto/corpus.html
- NVIDIA Jetson PyTorch: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/
- TQDM library to track progress on training: https://tqdm.github.io/
- Hugging Face course: https://huggingface.co/learn/llm-course/chapter1/1
- Pytorch with CUDA docs: https://docs.pytorch.org/docs/stable/notes/cuda.html
- Cuda deep learning for referencing syntax and cuda optimizations: https://docs.nvidia.com/
- Mini C4: https://huggingface.co/datasets/allenai/c4
- Hugging Face GPT2 Docs: https://huggingface.co/docs/transformers/model_doc/gpt2
- Hugging Face tokenizers: https://huggingface.co/docs/tokenizers/en/index
- Illustrated GPT-2 paper: https://jalammar.github.io/illustrated-gpt2/
- Dataloader/sets/sampling: https://docs.pytorch.org/docs/stable/data.html
- General Pytorch: https://docs.pytorch.org/docs/stable/index.html
- AMP documentation for CUDA: https://docs.pytorch.org/docs/stable/amp.html
- Cross-entropy loss: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Gradient descent: https://www.ibm.com/think/topics/gradient-descent
- Gradient descent sample algorithm: https://www.geeksforgeeks.org/machine-learning/gradient-descent-algorithm-and-its-variants/
- Mixed precision scaling: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html 
"""

"""
Processing steps to outline: (DONE!)
1. Prompt given to model (check!)
2. Tokenizer -> convert text to IDs of tokens (check!)
3. Model receives token IDs (check!)
4. Model predicts probabilities of next token (check!)
5. Token appended to input for next-word prediction (check!)
6. Continues until stop condition: (check!)
    1. End of sentence token (check!)
    2. Maximum length of input (check!)
    3. User interference (check!)
7. Token ID is converted to text (check!)
8. Return string to user (check!)
"""

import os #filesystem operations like making files and directories
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # force CUDA synchronous errors for debugging
import math #imports math functions for computing perplexity from loss as designated
import random #among training data, randomize their order so it will be a bit more diverse
import torch #Pytorch lib
from datasets import load_dataset #c4 mini from Hugging face
from torch.utils.data import Dataset, DataLoader #custom dataset class and minibatch creator
from torch.nn.utils import clip_grad_norm_ #gradients are clipped in place so they are prevented from getting too large
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast #GPT2 hyperparameters, decoder-only class for the model, and pretrained tokenizer
import matplotlib.pyplot as plt #plotter for the training, validation loss curve (convert to image and save)
from tqdm.auto import tqdm #display progress of trainiing loops
import numpy as np #numeric operations to be shown via the matplotlib graph and other float arrays
import nltk #large test data
from nltk.corpus import brown, reuters #import datasets for subwords and news reports for training
import torch.nn.functional as F #softmax sampling

#-------------------------------------- CONFIG -----------------------------------------------------
MODEL_CONFIG = {
    "block_size": 128, #size of input (how many tokens model can see at once)
    "batch_size": 8, #samples per iteration via a processing sequence
    "epochs": 25, #passes through the dataset in order to train model
    "n_layer": 4, #number of transformer blocks
    "n_head": 4, #number of heads per transformer block
    "n_embd": 256, #embedding dimension within model
    "lr": 3e-4, #learning rate for AdamW optimizer
    "out_dir": "checkpoints", #save location for model
    "use_amp": True, #amp for CUDA toggle
    "num_workers": 2, #dataloader threads for inputs to CPU
    "seed": 42, #random seed to vary data being trained on
    "nltk_limit": None, # None = load all NLTK docs
    "c4_limit": None, # None = load full mini C4
}

#-------------------------------------- DATA LOADING -----------------------------------------------------
def load_texts_from_nltk(limit=None): #data load and processing
    texts = []
    for corpus in [reuters, brown]:  # iterate over both corpora
        ids = corpus.fileids()  # get all files
        if limit is not None:
            ids = ids[:limit]  # limit number if specified
        # tqdm progress bar added for clarity on which corpus is loading
        for fid in tqdm(ids, desc=f"Loading {corpus.__class__.__name__}"):
            try:
                texts.append(" ".join(corpus.words(fid)))  # join words into a single string
            except:
                continue
    return texts

def load_small_c4(limit=None):
    """Loads mini C4 dataset (brando/small-c4-dataset)"""
    texts = []
    ds = load_dataset("brando/small-c4-dataset", split="train")
    for i, ex in enumerate(ds):
        if 'text' in ex and ex['text']:
            texts.append(ex['text'])
            if limit is not None and len(texts) >= limit:
                break
    return texts

def build_token_blocks(tokenizer, texts, block_size=128):
    """
    Turns tokenized texts into fixed-size sequences (blocks)
    Each block is a chunk of tokens the model can process at once.
    - Tokenizes each text
    - Splits it into non-overlapping sequences of length `block_size`
    - These blocks become input/target pairs for training
    """
    blocks = []
    for txt in texts:
        ids = tokenizer.encode(txt) #convert to list of integer tokens 
        # iterate over the text in chunks the model can handle
        for i in range(0, len(ids) - block_size, block_size): 
            blocks.append(ids[i:i + block_size])
    return blocks

def fix_out_of_range_token_ids(blocks, vocab_size, tokenizer): #ensure IDs are within range, replace invalid with EOS IDs
    """
    Ensure every token id in `blocks` is < vocab_size.
    If any id is >= vocab_size, replace it with tokenizer.eos_token_id (safe).
    Prevents device-side asserts during training if bad token IDs appear.
    """
    eos_id = getattr(tokenizer, "eos_token_id", 0)
    fixed_blocks = []
    max_seen = -1
    count_replaced = 0
    for b in blocks:
        new_b = []
        for tid in b:
            if tid >= vocab_size or tid < 0:
                new_b.append(eos_id)
                count_replaced += 1
            else:
                new_b.append(tid)
            if tid > max_seen:
                max_seen = tid
        fixed_blocks.append(new_b)
    print(f"[token-fix] max_token_id_seen={max_seen}, vocab_size={vocab_size}, replacements={count_replaced}")
    return fixed_blocks #cleansed from invalid IDs

class BlockDataset(Dataset): #takes blocks as a list of token IDs, return a number of blocks 
    def __init__(self, blocks):
        self.blocks = blocks
    def __len__(self):
        return len(self.blocks)
    def __getitem__(self, idx):
        x = torch.tensor(self.blocks[idx], dtype=torch.long) #convert the list of integers to a Pytorch tensor (longs are required for embedding lookups)
        return x, x.clone()  # input and label are same (next-token prediction)

def collate_fn(batch):
    """
    Pads inputs in the batch to same length (padding_value = tokenizer.pad_token_id).
    Labels are padded with -100 so cross-entropy ignores them.
    """
    input_ids = [b[0] for b in batch] #input tensors form batch list
    labels = [b[1] for b in batch] #extract label tensors
    pad_id = getattr(tokenizer, "pad_token_id", 0) #padding tokens are added to make input sequences have the same length (attention calculation correction)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id) #use sequence to lay out IDs along a dimension
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) #pad labels use a value to be ignored by Pytorch
    return input_ids, labels #batched tensors

def build_model(tokenizer, n_layer=4, n_head=4, n_embd=256, n_positions=128): #model designations (function defaults that can be overwritten) 
    vocab_size = tokenizer.vocab_size  # use tokenizer's vocab size 
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0 #pad_token_id needs to be set in config to keep model aware which token is being padded
    )
    model = GPT2LMHeadModel(config) #GPT2 model and the language modeling head for text generation 
    model.resize_token_embeddings(vocab_size) #assure token embedding for model is the same size as its vocab
    return model #constructed model (config embedded)

def evaluate(model, dl, device): #performance evaluation and metric plotting
    model.eval() #evaluation mode
    losses = [] #list to collect batch losses
    with torch.no_grad(): #no-gradient context (no need to track), makes sure PyTorch doesn't even compute them here during inferences and evaluation 
        for xb, yb in dl: #iteration over validation and test dataloader 
            xb, yb = xb.to(device), yb.to(device) #cpu or gpu processing (google colab DO NOT FORGET INITIALIZE WITH T4)
            outputs = model(input_ids=xb, labels=yb) #aligns and target for next-token prediction, calc cross-entropy
            losses.append(outputs.loss.item()) #append scalar loss to list 
    return float(np.mean(losses)) #mean of collected losses 

def compute_perplexity(loss):
    return math.exp(loss) #average cross-entrpy per token

@torch.no_grad() #generate text from user prompt
def sample_text(model, tokenizer, prompt, max_length=50, top_k=50, temperature=0.9, device='cuda'):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device) #encode prompt into token IDs, wrap in list to make 2D tensor 
    for i in range(max_length): #generate tokens one by one 
        outputs = model(input_ids=input_ids) #model gets logits for each position, outputs contain logits formatted data
        logits = outputs.logits[:, -1, :] / temperature #logits themselves for last token position
        probs = F.softmax(logits, dim=-1) #scaled logits are scaled to probabilities with softmax 
        top_probs, top_idx = torch.topk(probs, top_k) #keep top-k probabilities and indices (vocab IDs)
        next_token = top_idx[0, torch.multinomial(top_probs[0], 1)] #sample a single token index from the top-k dist. -> index into top-k according to probabilities 
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1) #next token ID to input sequence is appended 
        if next_token.item() == tokenizer.eos_token_id: #ends if end of sequence token ID is seen 
            break
    return tokenizer.decode(input_ids[0].tolist()) #decode full sequence back into test and return 

#-------------------------------------- MAIN -----------------------------------------------------
def main(cfg):
    nltk.download("brown") #NLTK corpora is downloaded to be available 
    nltk.download("reuters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #test hardware 
    print("Using device:", device)

    global tokenizer #pretrained tokenizer from Huggingface
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    nltk_texts = load_texts_from_nltk(limit=cfg["nltk_limit"])
    c4_texts = load_small_c4(limit=cfg["c4_limit"])
    texts = nltk_texts + c4_texts
    print(f"Loaded {len(texts)} total text samples.")

    blocks = build_token_blocks(tokenizer, texts, block_size=cfg["block_size"]) #convert test to token IDs and configure to blocks for processing (lists of integers)

    vocab_size = tokenizer.vocab_size #validate size of tokenizer to validate IDs 
    max_id = max((max(b) if len(b)>0 else 0) for b in blocks) #max token ID present in all blocks (used to ensure unformity in configured blocks)
    print(f"[debug] max token id in blocks = {max_id}, tokenizer.vocab_size = {vocab_size}")
    if max_id >= vocab_size or max_id < 0:
        print("[debug] Found out-of-range token IDs — remapping to safe token (eos).")
        blocks = fix_out_of_range_token_ids(blocks, vocab_size, tokenizer)
    if len(blocks) < 1:
        raise RuntimeError("No token blocks created. Try smaller block_size or more text.")
    random.shuffle(blocks) #order of blocks were randomized between data blocks containing tokenized inputs from NLTK corpora and the C4 mini

    n = len(blocks) #total blocks 
    train_end = int(0.8 * n) #end index for training set 
    val_end = train_end + int(0.1 * n) #end index for validation set 
    train_dl = DataLoader(BlockDataset(blocks[:train_end]), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn) #load dataloaders for training, validating, and testing (no data leaks between them)
    val_dl = DataLoader(BlockDataset(blocks[train_end:val_end]), batch_size=cfg["batch_size"], collate_fn=collate_fn) #training dataset allows model to learn patterns, validation tunes hyperparams and overfitting of data
    test_dl = DataLoader(BlockDataset(blocks[val_end:]), batch_size=cfg["batch_size"], collate_fn=collate_fn) #testing dataset evaluates the results from the other two datasets 

    model = build_model(tokenizer, n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"]) 
    model.to(device) #move model parameters to GPU ideally for processing 
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"]) #AdamW is an optimizer with decoupled weight decay (standard for transformers, apparently)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and cfg["use_amp"])) #use cuda for scaling gradients (using Nvidia T4 in colab so it can be used)
    train_losses, val_losses = [], [] #lists for tracking training and validation averages to plot 

    #-------------------------------------- TRAINING LOOP -----------------------------------------------------
    for epoch in range(cfg["epochs"]):
        model.train() #training mode 
        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg['epochs']}") #training dataloader is wrapped into tqdm for showing progress 
        batch_losses = [] #epoch batch loss 

        for xb, yb in loop: #batches 
            xb, yb = xb.to(device), yb.to(device) #batch tensors (multi-dim. array, first dim = # examples per batch) (2D) moved onto gpu for processing (batch tensor of input sequences and labels)
            optimizer.zero_grad() #reset optimizer gradients to make individualized results from previous processes (no accumulating gradients from other batches)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and cfg["use_amp"])): #enable cuda -> Pytorch runs tensor/matrix operations on the GPU, autocast uses mixed precision (some FP16, some FP32 (?))
                outputs = model(input_ids=xb, labels=yb) #runs a forward pass through the model (embedding table and entropy values) -> logits and loss as outputs 
                loss = outputs.loss #loss tensor extracted for graph
            scaler.scale(loss).backward() #scale loss after scaling initially to calculate gradients and manages scaling for MP (weights of vocab tokens updated after this and optimizer, fed by forward passing of data)
            scaler.unscale_(optimizer) #unscale gradients from parameter -> scale gradients so clipping could happen correctly, avoids inflated gradients that remained scaled up 
            clip_grad_norm_(model.parameters(), 1.0) #clip gradients by modifying .grad tensors directly, clips true gradients 
            scaler.step(optimizer) #perform optimizer step if gradients are valid with no issues, parameters of the model (weights) are updated
            scaler.update() #loss scaling factor for mixed precision is updated based on outputs 
            batch_losses.append(loss.item()) #accumulate per-batch loss to find training loss per batch each epoch 
            loop.set_postfix(loss=loss.item()) #update progress bar via tqdm

        avg_train_loss = float(np.mean(batch_losses)) #evaluates average losses 
        val_loss = evaluate(model, val_dl, device) #evaluate loss and perplexity for training and values 
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        perplexity = compute_perplexity(val_loss)
        print(f"\nEpoch {epoch+1}: Train loss={avg_train_loss:.4f}, Val loss={val_loss:.4f}, Perplexity={perplexity:.2f}")

        os.makedirs(cfg["out_dir"], exist_ok=True) #checkpoint each epoch because Google Colab loves timing out
        model_path = os.path.join(cfg["out_dir"], f"gpt2_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        plt.figure() #matplotlib graph
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        graph_path = os.path.join(cfg["out_dir"], "loss_curve.png")
        plt.savefig(graph_path)
        plt.close()
        print(f"Updated loss curve saved at {graph_path}")

    test_loss = evaluate(model, test_dl, device) #find losses and perplexity
    test_ppl = compute_perplexity(test_loss)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Perplexity: {test_ppl:.2f}")

if __name__ == "__main__":
    main(MODEL_CONFIG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #designate gpu 
model_path = "checkpoints/gpt2_epoch25.pt" #model path for latest training epoch

model = build_model(tokenizer,
                    n_layer=MODEL_CONFIG["n_layer"],
                    n_head=MODEL_CONFIG["n_head"],
                    n_embd=MODEL_CONFIG["n_embd"],
                    n_positions=MODEL_CONFIG["block_size"])
model.load_state_dict(torch.load(model_path, map_location=device)) #load state of model to device and put into evaluation mode for seeing text
model.to(device)
model.eval()

test_prompts = [
    "Hello, my name is ",
    "The current economy",
    "The quick brown fox",
    "The mitochondria is the powerhouse of the cell"
]

print("\n----- TEST OUTPUTS -----\n")
for prompt in test_prompts: #run prompts through model to get outputs
    print(f"Prompt: {prompt}")
    text = sample_text(model, tokenizer, prompt, max_length=60, top_k=50, temperature=0.9, device=device)
    print(f"Generated: {text}\n")
    print("-" * 60)