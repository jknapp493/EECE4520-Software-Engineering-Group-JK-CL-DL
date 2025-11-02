"""
GPT-2 Data Preprocessing, Tokenizer and Trainer
-------------------------------------------------
- Used Brown and Reuters sets from NLTK corpus (subwords and news articles)
- Used the Hugging Face Transformers mini c4 to train so it wouldn't be as much data to train on

Sources:
- Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers/python/latest/
- Hugging Face Transformers (GPT-2): https://huggingface.co/docs/transformers/model_doc/gpt2
- NLTK Corpora How-to: https://www.nltk.org/howto/corpus.html (1.15, 1.3 million words for brown and reuters)
- NVIDIA Jetson PyTorch: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/
- TQDM library to track progress on training: https://tqdm.github.io/
- Hugging Face course: https://huggingface.co/learn/llm-course/chapter1/1
- Pytorch with CUDA docs: https://docs.pytorch.org/docs/stable/notes/cuda.html
- Cuda deep learning for referencing syntax and cuda optimizations: https://docs.nvidia.com/
- Mini C4: https://huggingface.co/datasets/allenai/c4 (1.78 million rows of ex.)
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
- Gradient Scaling: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- 
"""
"""
Gradient scaling is the process of scaling data features to have a similar range, 
for algorithms using gradient descent, improving model performance by preventing features 
with larger values from having an outsized influence 
and helps the gradient descent algorithm converge faster and more reliably
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
#Execution time: 2 hours
#NOTE: SIZE OF GPT2 MODEL 

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

class DatasetManager:
    def load_texts_from_nltk(self, limit=None): #data load and processing
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

    def load_small_c4(self, limit=None):
        """Loads mini C4 dataset (brando/small-c4-dataset)"""
        texts = []
        ds = load_dataset("brando/small-c4-dataset", split="train")
        for i, ex in enumerate(ds):
            if 'text' in ex and ex['text']:
                texts.append(ex['text'])
                if limit is not None and len(texts) >= limit:
                    break
        return texts

class PreprocessingManager:
    def build_token_blocks(self, tokenizer, texts, block_size=128):
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

    def fix_out_of_range_token_ids(self, blocks, vocab_size, tokenizer): #ensure IDs are within range, replace invalid with EOS IDs (edit for class)
        """
        Ensure every token id in `blocks` is < vocab_size.
        If any id is >= vocab_size, replace it with tokenizer.eos_token_id (safe).
        Prevents device-side asserts during training if bad token IDs appear.
        """
        eos_id = getattr(tokenizer, "eos_token_id", 0) #designate and receive a token for ending sentence
        fixed_blocks = [] #store corrected blocks 
        max_seen = -1 #record highest seen ID in block 
        count_replaced = 0 
        for b in blocks: #through each block within each token ID, check if ID is invalid (not part of vocabulary or negative ID)
            new_b = []
            for tid in b:
                if tid >= vocab_size or tid < 0:
                    new_b.append(eos_id) #replace invalid token with EOS, increase counter
                    count_replaced += 1
                else:
                    new_b.append(tid)
                if tid > max_seen: #verify mappin and within bounds (not invalid)
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

class ModelBuilder:
    def build_model(self, tokenizer, n_layer=4, n_head=4, n_embd=256, n_positions=128): #model designations (function defaults that can be overwritten) 
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

class OptimizerManager:
    def __init__(self, model, lr, use_amp, device):
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr) #AdamW is an optimizer with decoupled weight decay (standard for transformers, apparently)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and use_amp)) #use cuda for scaling gradients (using Nvidia T4 in colab so it can be used)

    def backward_and_step(self, loss, model):
        # mimic original per-batch sequence: scale->backward->unscale->clip->step->update
        self.scaler.scale(loss).backward() #scale loss after scaling initially to calculate gradients and manages scaling for MP 
        #(weights of vocab tokens updated after this and optimizer, fed by forward passing of data)
        self.scaler.unscale_(self.optimizer) #unscale gradients from parameter -> scale gradients so clipping could happen correctly, avoids inflated gradients that remained scaled up 
        clip_grad_norm_(model.parameters(), 1.0) #clip gradients by modifying .grad tensors directly, clips true gradients 
        self.scaler.step(self.optimizer) #perform optimizer step if gradients are valid with no issues, parameters of the model (weights) are updated
        self.scaler.update() #loss scaling factor for mixed precision is updated based on outputs 
        self.optimizer.zero_grad() #reset optimizer gradients to make individualized results from previous processes (no accumulating gradients from other batches)

class CheckpointManager:
    def save_checkpoint(self, model, out_dir, epoch):
        os.makedirs(out_dir, exist_ok=True) #checkpoint each epoch because Google Colab loves timing out
        model_path = os.path.join(out_dir, f"gpt2_epoch{epoch}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

class GraphManager: #graph perplexity and loss in class and not in main (NEW CLASS FOR DIAGRAM)
    def save_loss_curve(self, train_losses, val_losses, out_dir):
        plt.figure() #matplotlib graph
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        graph_path = os.path.join(out_dir, "loss_curve.png")
        plt.savefig(graph_path)
        plt.close()
        print(f"Updated loss curve saved at {graph_path}")

class TrainingManager:
    def __init__(self, MODEL_CONFIG, device):
        self.MODEL_CONFIG = MODEL_CONFIG
        self.device = device
        self.checkpoint_mgr = CheckpointManager()
        self.graph_mgr = GraphManager()

    def evaluate(self, model, dl, device): #performance evaluation and metric plotting
        model.eval() #evaluation mode
        losses = [] #list to collect batch losses
        with torch.no_grad(): #no-gradient context (no need to track), makes sure PyTorch doesn't even compute them here during inferences and evaluation 
            for xb, yb in dl: #iteration over validation and test dataloader 
                xb, yb = xb.to(device), yb.to(device) #cpu or gpu processing (google colab DO NOT FORGET INITIALIZE WITH T4)
                outputs = model(input_ids=xb, labels=yb) #aligns and target for next-token prediction, calc cross-entropy
                losses.append(outputs.loss.item()) #append scalar loss to list 
        return float(np.mean(losses)) #mean of collected losses 

    def compute_perplexity(self, loss):
        return math.exp(loss) #average cross-entrpy per token

    def train(self, model, train_dl, val_dl, test_dl, optimizer_mgr, tokenizer):
        train_losses, val_losses = [], [] #lists for tracking training and validation averages to plot 
        for epoch in range(self.MODEL_CONFIG["epochs"]):
            model.train() #training mode 
            loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{self.MODEL_CONFIG['epochs']}") #training dataloader is wrapped into tqdm for showing progress 
            batch_losses = [] #epoch batch loss 

            for xb, yb in loop: #batches 
                xb, yb = xb.to(self.device), yb.to(self.device) #batch tensors (multi-dim. array, first dim = # examples per batch) (2D) 
                #moved onto gpu for processing (batch tensor of input sequences and labels)
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda' and self.MODEL_CONFIG["use_amp"])): #enable cuda -> Pytorch runs tensor/matrix 
                    #operations on the GPU, autocast uses mixed precision (some FP16, some FP32 (?))
                    outputs = model(input_ids=xb, labels=yb) #runs a forward pass through the model (embedding table and entropy values) -> logits and loss as outputs 
                    loss = outputs.loss #loss tensor extracted for graph
                optimizer_mgr.backward_and_step(loss, model)
                batch_losses.append(loss.item()) #accumulate per-batch loss to find training loss per batch each epoch 
                loop.set_postfix(loss=loss.item()) #update progress bar via tqdm

            avg_train_loss = float(np.mean(batch_losses)) #evaluates average losses 
            val_loss = self.evaluate(model, val_dl, self.device) #evaluate loss and perplexity for training and values 
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            perplexity = self.compute_perplexity(val_loss)
            print(f"\nEpoch {epoch+1}: Train loss={avg_train_loss:.4f}, Val loss={val_loss:.4f}, Perplexity={perplexity:.2f}")

            # Save checkpoint and graph (preserve original per-epoch checkpoint behavior)
            self.checkpoint_mgr.save_checkpoint(model, self.MODEL_CONFIG["out_dir"], epoch+1)
            self.graph_mgr.save_loss_curve(train_losses, val_losses, self.MODEL_CONFIG["out_dir"])

        # Final evaluation
        test_loss = self.evaluate(model, test_dl, self.device) #find losses and perplexity
        test_ppl = self.compute_perplexity(test_loss)
        print(f"\nFinal Test Loss: {test_loss:.4f}, Perplexity: {test_ppl:.2f}")
        return train_losses, val_losses

class Sampler:
    def __init__(self, seed=None):
        if seed is not None: 
            random.seed(seed) #allows for data to be more randomized -> shuffle data
            torch.manual_seed(seed)
        self.seed = seed

    @torch.no_grad()
    def sample(self, model, tokenizer, input_ids, max_length=50, top_k=50, temperature=0.9, device='cuda'):
        model.eval()
        for _ in range(max_length):
            outputs = model(input_ids=input_ids) #model gets logits for each position, outputs contain logits formatted data
            logits = outputs.logits[:, -1, :] / temperature #logits themselves for last token position
            probs = F.softmax(logits, dim=-1) #scaled logits are scaled to probabilities with softmax 
            top_probs, top_idx = torch.topk(probs, top_k) #keep top-k probabilities and indices (vocab IDs)
            next_token = top_idx[0, torch.multinomial(top_probs[0], 1)] #sample a single token index from the top-k dist. -> index into top-k according to probabilities 
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1) #next token ID to input sequence is appended 
            if next_token.item() == tokenizer.eos_token_id: #ends if end of sequence token ID is seen 
                break
        return input_ids

class InferenceManager:
    def __init__(self, model, tokenizer, sampler, device):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.device = device

    def generate(self, prompt, max_length=50, top_k=50, temperature=0.9):
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device) #encode prompt into token IDs, wrap in list to make 2D tensor 
        out_ids = self.sampler.sample(self.model, self.tokenizer, input_ids, max_length=max_length, top_k=top_k, temperature=temperature, device=self.device)
        return self.tokenizer.decode(out_ids[0].tolist())
    
def setup(MODEL_CONFIG): #NEW CLASS (setup everything)
    """
    Setup everything previously in the top of main:
    - download corpora
    - select device
    - load tokenizer
    - load datasets and build token blocks
    - create dataloaders, model, optimizer manager, trainer, sampler, inference manager
    Returns a dictionary of prepared objects (device, tokenizer, dataloaders, model, managers)
    """
    nltk.download("brown") #NLTK corpora is downloaded to be available 
    nltk.download("reuters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #test hardware 
    print("Using device:", device)

    global tokenizer #pretrained tokenizer from Huggingface
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '<pad>'}) #input padding for blocks 

    ds_mgr = DatasetManager() #designate managers 
    pp_mgr = PreprocessingManager()
    model_builder = ModelBuilder()

    nltk_texts = ds_mgr.load_texts_from_nltk(limit=MODEL_CONFIG["nltk_limit"])
    c4_texts = ds_mgr.load_small_c4(limit=MODEL_CONFIG["c4_limit"])
    texts = nltk_texts + c4_texts
    print(f"Loaded {len(texts)} total text samples.") #confirm length

    blocks = pp_mgr.build_token_blocks(tokenizer, texts, block_size=MODEL_CONFIG["block_size"]) #convert test to token IDs and configure to blocks for processing (lists of integers)

    vocab_size = tokenizer.vocab_size #validate size of tokenizer to validate IDs 
    max_id = max((max(b) if len(b)>0 else 0) for b in blocks) #max token ID present in all blocks (used to ensure uniformity in configured blocks)
    print(f"[debug] max token id in blocks = {max_id}, tokenizer.vocab_size = {vocab_size}")
    if max_id >= vocab_size or max_id < 0:
        print("[debug] Found out-of-range token IDs — remapping to safe token (eos).")
        blocks = pp_mgr.fix_out_of_range_token_ids(blocks, vocab_size, tokenizer)
    if len(blocks) < 1:
        raise RuntimeError("No token blocks created. Try smaller block_size or more text.")
    random.shuffle(blocks) #order of blocks were randomized between data blocks containing tokenized inputs from NLTK corpora and the C4 mini

    n = len(blocks) #total blocks 
    train_end = int(0.8 * n) #end index for training set 
    val_end = train_end + int(0.1 * n) #end index for validation set 
    train_dl = DataLoader(BlockDataset(blocks[:train_end]), batch_size=MODEL_CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn) 
    #load dataloaders for training, validating, and testing (no data leaks between them)
    val_dl = DataLoader(BlockDataset(blocks[train_end:val_end]), batch_size=MODEL_CONFIG["batch_size"], collate_fn=collate_fn) 
    #training dataset allows model to learn patterns, validation tunes hyperparams and overfitting of data
    test_dl = DataLoader(BlockDataset(blocks[val_end:]), batch_size=MODEL_CONFIG["batch_size"], collate_fn=collate_fn) 
    #testing dataset evaluates the results from the other two datasets 

    model = model_builder.build_model(tokenizer, n_layer=MODEL_CONFIG["n_layer"], n_head=MODEL_CONFIG["n_head"], n_embd=MODEL_CONFIG["n_embd"]) 
    model.to(device) #move model parameters to GPU ideally for processing 
    optimizer_mgr = OptimizerManager(model, MODEL_CONFIG["lr"], MODEL_CONFIG["use_amp"], device) 
    trainer = TrainingManager(MODEL_CONFIG, device) #train model
    sampler = Sampler(seed=MODEL_CONFIG["seed"]) #sample outputs 
    infer = InferenceManager(model, tokenizer, sampler, device) #inferences 
    return {
        "device": device,
        "tokenizer": tokenizer,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "test_dl": test_dl,
        "model": model,
        "optimizer_mgr": optimizer_mgr,
        "trainer": trainer,
        "sampler": sampler,
        "inference": infer,
        "blocks": blocks,      
        "texts_count": len(texts) 
    }

#------------MAIN STARTS HERE--------------
def main(MODEL_CONFIG): #needs to be smaller, IMPLEMENT CLASS DIAGRAM 
    objs = setup(MODEL_CONFIG)

    device = objs["device"]
    tokenizer = objs["tokenizer"]
    train_dl = objs["train_dl"]
    val_dl = objs["val_dl"]
    test_dl = objs["test_dl"]
    model = objs["model"]
    optimizer_mgr = objs["optimizer_mgr"]
    trainer = objs["trainer"]
    inference = objs["inference"]

    train_losses, val_losses = trainer.train(model, train_dl, val_dl, test_dl, optimizer_mgr, tokenizer)
    model_path = os.path.join(MODEL_CONFIG["out_dir"], f"gpt2_epoch{MODEL_CONFIG['epochs']}.pt")
    #------------MAIN ENDS HERE--------------
    
    try: #testing output model
        model.load_state_dict(torch.load(model_path, map_location=device)) #load state of model to device and put into evaluation mode for seeing text
        model.to(device)
        model.eval()
    except Exception as e:
        print("Warning: could not load checkpoint:", e)

    test_prompts = [
        "Hello, my name is ",
        "The current economy",
        "The quick brown fox",
        "The mitochondria is the powerhouse of the cell"
    ]

    print("\n----- TEST OUTPUTS -----\n")
    for prompt in test_prompts: #run prompts through model to get outputs
        print(f"Prompt: {prompt}")
        text = inference.generate(prompt, max_length=60, top_k=50, temperature=0.9)
        print(f"Generated: {text}\n")
        print("-" * 60)

if __name__ == "__main__": #start script and load config
    main(MODEL_CONFIG)


#Generated Output: ----- TEST OUTPUTS -----

# Prompt: Hello, my name is 
# Generated: Hello, my name is . We are excited to show you how to be. Let's start a good start!
# I love it from you to be a few new times.
# Hello, here. Have you set my name to me?
# how many years ago I am so happy with my son. Where did this

# ------------------------------------------------------------
# Prompt: The current economy
# Generated: The current economy is not expected to fall over its next quarter and about four years , he said , 
# " I have not yet to lose ," he added . While there is not about 12 things it is aggravate for the dollar , he added 
# . As expected , he said , the Fed is aware about whether the Fed

# ------------------------------------------------------------
# Prompt: The quick brown fox
# Generated: The quick brown fox to the northern bazaar have been found in the most beautiful boho in the world. 
# The best place to get along with it the local culture on the southern side of the island, separates you from the main area or the world. 
# You find a variety of outdoor cats, beach and beach options

# ------------------------------------------------------------
# Prompt: The mitochondria is the powerhouse of the cell
# Generated: The mitochondria is the powerhouse of the cell function of the body and the magnetic field with a muscle. 
# It is as an exercise factor factor that gets rid of the eye. It is like a needle, it can be taken by an additional pain, 
# and it is quicker and quicker. It is quicker and quicker. The fat burning pain is quicker

# ------------------------------------------------------------
