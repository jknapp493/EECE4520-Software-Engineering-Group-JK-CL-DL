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

Design Pattern Stuff:
Factory Pattern (Refactoring Guru): https://refactoring.guru/design-patterns/factory-method
Service Layer / Manager Pattern (Martin Fowler): https://martinfowler.com/eaaCatalog/serviceLayer.html
Dependency Injection (Martin Fowler): https://martinfowler.com/articles/injection.html
Separation of Concerns (Stanford CS): https://web.stanford.edu/class/cs253/lectures/02-software-design.pdf
Python Modular Programming: https://docs.python.org/3/tutorial/modules.html
Strategy Pattern (Refactoring Guru): https://refactoring.guru/design-patterns/strategy
Utility Class Reference: https://stackoverflow.com/questions/33618423/is-a-utility-class-an-anti-pattern
Python dictionaries (for cfg): https://docs.python.org/3/tutorial/datastructures.html#dictionaries
Python dataloading with classes: https://pytorch.org/docs/stable/data.html
PyTorch tensor semantics: https://pytorch.org/docs/stable/tensors.html
PyTorch optimizers: https://pytorch.org/docs/stable/optim.html
Torch mixed precision autocast: https://pytorch.org/docs/stable/amp.html
Hugging Face GPT-2 Config class: https://huggingface.co/docs/transformers/main_classes/configuration
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
"""
GPT-2 Training Simulation with Design Patterns

DESIGN PATTERNS IMPLEMENTED: (DONE!)
1. Factory Method (Model Creation) (check!)
get_config_for_size() + GPT2ModelFactory choose and return cfg for small/medium/large; it hides construction logic and returns configuration objects.
2. Singleton (System Resources) (check!)
SystemResources ensures a single tokenizer + device instance is used across the run (prevents repeated heavy loads).
3. Adapter (Terminal Output) (check!)
Adapter: ColoredTerminalAdapter wraps a Terminal to add color behavior without changing the terminal interface.
4. Decorator (Performance Monitoring Interlinked with Adapter) (check!)
Decorator: PerformanceMonitorDecorator wraps the model’s forward to measure and log timing; ModelDecorator is a structural wrapper that injects the decorator into the model.
5. Command (Inference Queueing) (check!)
Command: GenerateRequestCommand and CommandInvoker encapsulate inference requests as objects you can queue and process.
"""

import os #filesystem operations like making files and directories
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  #force CUDA synchronous errors for debugging
import math #imports math functions for computing perplexity from loss as designated
import random #among training data, randomize their order so it will be a bit more diverse
import torch #Pytorch lib
from datasets import load_dataset #c4 mini from Hugging face 
from torch.utils.data import Dataset, DataLoader #custom dataset class and minibatch creator
from torch.nn.utils import clip_grad_norm_ #gradients are clipped in place so they are prevented from getting too large
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast #GPT2 hyperparameters, decoder-only class for the model, and pretrained tokenizer
import matplotlib.pyplot as plt #plotter for the training, validation loss curve (convert to image and save)
from tqdm.auto import tqdm # display progress of training loops
import numpy as np #numeric operations to be shown via the matplotlib graph and other float arrays
import nltk #large test data library
from nltk.corpus import brown, reuters, gutenberg, inaugural #import datasets for subwords and news reports for training (and books/speeches for large)
import torch.nn.functional as F #softmax sampling and other functional neural network operations

import time #imports time to track performance metrics for the decorator pattern
import dataclasses #imports dataclasses to create structured state objects easily
from functools import wraps #imports wraps to preserve function metadata in decorators
from abc import ABC, abstractmethod #imports abstract base classes to enforce interface implementation (Adapter pattern)
from enum import Enum #imports Enum to define constant values for Model Size options
import types #imports types for dynamic method binding

MODEL_CONFIG = { 
    "block_size": 128, #size of input (how many tokens model can see at once)
    "batch_size": 8, #samples per iteration via a processing sequence
    "epochs": 25, #passes through the dataset in order to train model
    "n_layer": 8, #number of transformer blocks
    "n_head": 4, #number of heads per transformer block
    "n_embd": 256, #embedding dimension within model
    "lr": 3e-4, #learning rate for AdamW optimizer
    "out_dir": "checkpoints", #save location for model
    "use_amp": True, #amp for CUDA toggle
    "num_workers": 2, #dataloader threads for inputs to CPU
    "seed": 42, #random seed to vary data being trained on
    "nltk_limit": None, #None = load all NLTK docs
    "c4_limit": 1000, #None = load full mini C4 - load partial for now 
    "extra_corpora": False #flag to load extra corpora for Large model options
}

# Singleton - Ensures only one instance of Device and Tokenizer exists
class SystemResources: #manage shared system resources like GPU and Tokenizer
    _instance = None #hold the single instance of an object 

    def __new__(cls):
        if cls._instance is None: #check instance existance, if not 
            cls._instance = super(SystemResources, cls).__new__(cls) #new instance
            cls._instance._initialized = False
        return cls._instance #return instance 

    def setup(self, tokenizer_name="gpt2"): #initialize heavy resources
        if not getattr(self, "_initialized", False): #check initialization state 
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Singleton SystemResources Initialized. Using device: {self.device}")

            try: #attempt tokenizer loading 
                self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name) #pretrained tokenizer from Hugging Face
                self.tokenizer.model_max_length = 1_000_000_000 
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                raise e #stop execution

            self.tokenizer.add_special_tokens({'pad_token': '<pad>'}) #add padding token for batch processing (instead of in main)
            self._initialized = True #initialized 
        return self.device, self.tokenizer #resource handles returned 


#Adapter - Adapts standard string messages to colored terminal output. 
class Terminal(ABC): 
    @abstractmethod #decorator to enforce implementation in subclasses
    def display_message(self, message: str): #abstract method
        pass #no implementation in interface

class StandardTerminal(Terminal): #implementation of the standard interface
    def display_message(self, message: str): #implementation of display method
        print(message)

class ColoredTerminalAdapter(Terminal): #addd color functionality
    _COLOR_CODES = { # Dictionary mapping color names to ANSI escape codes (high-level names to actual sequences)
        'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m', #Red, Green, Yellow codes
        'blue': '\033[94m', 'purple': '\033[95m', 'cyan': '\033[96m', #Blue, Purple, Cyan codes
        'white': '\033[97m', 'reset': '\033[0m' #White and Reset codes
    }

    def __init__(self, terminal: Terminal): #constructor to accept wrapper
        self.terminal = terminal #store existing terminal object

    def display_message(self, message: str, color: str = 'white'): #integrate new color pattern 
        color_code = self._COLOR_CODES.get(color, self._COLOR_CODES['reset'])
        formatted_msg = f"{color_code}{message}{self._COLOR_CODES['reset']}" #wrap message in color codes
        self.terminal.display_message(formatted_msg) #delegate printing to the wrapped object

#Factory - Creates distinct model configurations based on user input
def get_config_for_size(size: str): #retrieve config
    if size == "small": 
        cfg = MODEL_CONFIG.copy() #copy and modify config parameters 
        cfg["epochs"] = 5  
        cfg["n_layer"] = 4 
        cfg["nltk_limit"] = 50 
        cfg["c4_limit"] = 100 
        cfg["lr"] = 5e-4 
        return cfg #return modified config
    elif size == "medium": 
        return MODEL_CONFIG.copy() 
    elif size == "large": 
        cfg = MODEL_CONFIG.copy() #copy and modify config parameters 
        cfg["epochs"] = 50 
        cfg["nltk_limit"] = None 
        cfg["c4_limit"] = None   
        cfg["nltk_repeat"] = 3 #repeat the NLTK dataset 3 times to increase weight
        cfg["extra_corpora"] = True #trigger loading Gutenberg and Inaugural corpora
        cfg["lr"] = 1.5e-4 #use lower learning rate for stability
        return cfg #return modified config
    else: 
        raise ValueError("unknown size") #invalid input 

class ModelSize(Enum): #strictly define allowed model sizes
    SMALL = "small" 
    MEDIUM = "medium"
    LARGE = "large"

class GPT2ModelFactory: #encapsulate creation logic
    @staticmethod #allow calling without instantiation
    def get_user_choice():
        print("\n=== GPT-2 Model Size Selection (Factory) ===")
        print("1. Small  (4 layers, 5 epochs)")
        print("2. Medium (8 layers, 25 epochs)")
        print("3. Large  (12 layers, 50 epochs, full data)")
        while True: #until valid input received
            choice = input("Enter your choice (1-3) [default 2]: ").strip() or "2"
            if choice == '1': return ModelSize.SMALL 
            elif choice == '2': return ModelSize.MEDIUM 
            elif choice == '3': return ModelSize.LARGE 
            else: print("Invalid choice.")

    @staticmethod #static method helper
    def get_preset_config(size_enum: ModelSize): #bridge enum to config
        return get_config_for_size(size_enum.value) #call helper to get dictionary of linked config 

#Decorator - Dynamically adds timing behavior, interlinks with Adapter for output
class PerformanceMonitorDecorator:
    def __init__(self, terminal_adapter=None): #constructor allowing dependency injection of terminal
        self.terminal = terminal_adapter #terminal adapter
        self.step_count = 0 #counter to throttle output (int counter used to limit how often the logging occurs)

    def decorate(self, func): #wrap a function
        @wraps(func) #maintain metadata of original function
        def wrapper(*args, **kwargs): #wrapper function
            t0 = time.time() #start time
            result = func(*args, **kwargs) #execute original function (model instance of forward pass, return original result after logging)
            t1 = time.time() #end time
            self.step_count += 1
            if self.terminal and self.step_count % 50 == 0: #check if terminal exists and threshold met
                self.terminal.display_message(f"[Decorator] Forward pass time (step {self.step_count}): {t1-t0:.4f}s", 'cyan')
            
            return result 
        return wrapper 

class ModelDecorator(torch.nn.Module): #component wrapper 
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.add_module("model", model) #register the inner model as a submodule (ensures it's present in self._modules)

    def add_decoration(self, decorator_instance): #apply decoration
        inner = object.__getattribute__(self, "_modules").get("model") #access the registered inner model
        if inner is None:
            raise AttributeError("No inner model registered.")
        orig_forward = inner.forward #store original forward to wrap in decorator or assign function back to forward so future calls go through the wrapper
        wrapped = decorator_instance.decorate(orig_forward) #wrapper creation
        inner.forward = wrapped #assign the wrapped function to the inner model's forward method (what is the forward method?)

    def forward(self, *args, **kwargs):
        inner = object.__getattribute__(self, "_modules").get("model") #explicitly call the inner model's forward method (where model's math is defined) -> model uses wrapper when forward pass runs 
        return inner.forward(*args, **kwargs)

    def __getattr__(self, name): #delegate attribute lookup to the inner model
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass
        modules = self.__dict__.get('_modules') #modules dictionary
        if modules is not None:
            inner = modules.get("model")
            if inner is not None: #if model exists
                return getattr(inner, name) #return attribute from inner model
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

#Command - Wraps generation requests as objects, supports queues
class Command(ABC): #abstract cmd interface 
    @abstractmethod #force implement abstract base class, all subsequent must have an execute method 
    def execute(self):
        pass # No body

class GenerateRequestCommand(Command): #wraps a text generation request
    def __init__(self, inference_manager, prompt, terminal):
        self.inference_manager = inference_manager #store receiver (inference manager)
        self.prompt = prompt #store state (prompt)
        self.terminal = terminal #store terminal for output

    def execute(self): 
        self.terminal.display_message(f"Processing Command: Generate for '{self.prompt}'", 'blue') #log start
        result = self.inference_manager.generate(self.prompt) #call the receiver logic
        self.terminal.display_message(f"Result: {result}\n", 'white') #log result

class CommandInvoker: #holds a queue of commands and executes them
    def __init__(self):
        self._queue = [] #init empty queue

    def add_command(self, command: Command): #queue commands
        self._queue.append(command) #add to list

    def process_queue(self):
        print(f"\n[Command Invoker] Processing {len(self._queue)} items in queue...") #log processing start
        for cmd in self._queue:
            cmd.execute()
        self._queue.clear() #clear queue after exec


class DatasetManager: 
    def load_texts_from_nltk(self, limit=None, repeat=1, extra_corpora=False): # data load and processing 
        texts = []
        corpora_list = [reuters, brown] # list of default corpora

        if extra_corpora: # check if extra data is requested
            print("Large model: Adding Gutenberg and Inaugural...") 
            corpora_list.extend([gutenberg, inaugural]) # add new corpora to list

        for corpus in corpora_list: 
            ids = corpus.fileids() # get list of file IDs in corpus
            if limit: ids = ids[:limit] # slice list to limit
            desc = f"Loading {corpus.__name__ if hasattr(corpus, '__name__') else str(corpus)}" # format description for progress bar
            # tqdm progress bar added for clarity on which corpus is loading
            for fid in tqdm(ids, desc=desc):
                try:
                    texts.append(" ".join(corpus.words(fid))) # join words into a single string
                except: 
                  continue 
        return texts * repeat # return texts list repeated 'repeat' times (if wanted to have "more" data to iterate through)

    def load_small_c4(self, limit=None): 
        """Loads mini C4 dataset (brando/small-c4-dataset)"""
        texts = [] 
        try: 
            ds = load_dataset("brando/small-c4-dataset", split="train") 
            for i, ex in enumerate(ds):
                if 'text' in ex and ex['text']: 
                    texts.append(ex['text']) 
                    if limit and len(texts) >= limit: 
                      break 
        except: pass
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
            ids = tokenizer.encode(txt) # convert to list of integer tokens
            # iterate over the text in chunks the model can handle
            for i in range(0, len(ids) - block_size, block_size): 
                chunk = ids[i:i+block_size]
                if len(chunk) == block_size:
                    blocks.append(chunk)
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

class BlockDataset(Dataset): 
    def __init__(self, blocks): self.blocks = blocks 
    def __len__(self): return len(self.blocks) 
    def __getitem__(self, idx): 
        x = torch.tensor(self.blocks[idx], dtype=torch.long) # convert the list of integers to a Pytorch tensor (longs are required for embedding lookups)
        return x, x.clone() # input and label are same (next-token prediction)

def collate_fn(batch):
    """
    Pads inputs in the batch to same length (padding_value = tokenizer.pad_token_id).
    Labels are padded with -100 so cross-entropy ignores them.
    """
    tokenizer = SystemResources().tokenizer # get singleton tokenizer
    input_ids = [b[0] for b in batch] # input tensors form batch list
    labels = [b[1] for b in batch] # extract label tensors
    pad_id = getattr(tokenizer, "pad_token_id", 0) #padding tokens are added to make input sequences have the same length (attention calculation correction)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id) # use sequence to lay out IDs along a dimension
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # pad labels use a value to be ignored by Pytorch
    return input_ids, labels # batched tensors

class ModelBuilder: 
    def build_model(self, tokenizer, model_size_enum: ModelSize, n_positions): # model designations (function defaults that can be overwritten) 
        vocab_size = tokenizer.vocab_size # use tokenizer's vocab size
        if model_size_enum == ModelSize.SMALL: #small
            config = GPT2Config(vocab_size=vocab_size, n_positions=n_positions, n_ctx=n_positions, 
                                n_embd=256, n_layer=4, n_head=4, pad_token_id=tokenizer.pad_token_id) 
        elif model_size_enum == ModelSize.MEDIUM: #medium
            config = GPT2Config(vocab_size=vocab_size, n_positions=n_positions, n_ctx=n_positions, 
                                n_embd=256, n_layer=4, n_head=4, pad_token_id=tokenizer.pad_token_id) 
        elif model_size_enum == ModelSize.LARGE: #large
            config = GPT2Config(vocab_size=vocab_size, n_positions=n_positions, n_ctx=n_positions, 
                                n_embd=768, n_layer=12, n_head=12, pad_token_id=tokenizer.pad_token_id) 
        else: raise ValueError("Unknown size") #handle invalid size
        
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

class GraphManager: #graph perplexity and loss in class and not in main 
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

class TrainingManager: # Class to orchestrate training loop
    def __init__(self, cfg, device): # Constructor
        self.cfg = cfg # Store config
        self.device = device # Store device
        self.checkpoint_mgr = CheckpointManager() # Checkpoint mgr instance
        self.graph_mgr = GraphManager() # Graph mgr instance

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

    def train(self, model, train_dl, val_dl, test_dl, optimizer_mgr):
        train_losses, val_losses = [], [] #lists for tracking training and validation averages to plot 
        for epoch in range(self.cfg["epochs"]): 
            model.train() #training mode
            loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{self.cfg['epochs']}") #wrap dataloader
            batch_losses = [] #epoch batch loss

            for xb, yb in loop: #batches 
                xb, yb = xb.to(self.device), yb.to(self.device) #batch tensors (multi-dim. array, first dim = # examples per batch) (2D) 
                #moved onto gpu for processing (batch tensor of input sequences and labels)
                optimizer_mgr.optimizer.zero_grad() # reset grads (prevent accumulation across batches, clears the .grad fields of model before computing gradients for current batch
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda' and self.cfg.get("use_amp", False))): #enable cuda -> Pytorch runs tensor/matrix 
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
            perplexity = self.compute_perplexity(val_loss) # compute perplexity for validation
            print(f"Stats: Train={avg_train_loss:.4f}, Val={val_loss:.4f}, Perplexity={perplexity:.2f}") 
            
            # Save checkpoint and graph (preserve original per-epoch checkpoint behavior)
            self.checkpoint_mgr.save_checkpoint(model, self.cfg["out_dir"], epoch+1) # Save model
            self.graph_mgr.save_loss_curve(train_losses, val_losses, self.cfg["out_dir"]) # Save graph
            
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
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1) #next token ID to input sequence is appended, model receives context for next generation 
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

def setup(model_size_enum: ModelSize): # Setup function to orchestrate objects
    """
    Setup everything previously in the top of main:
    - download corpora
    - select device
    - load tokenizer
    - load datasets and build token blocks
    - create dataloaders, model, optimizer manager, trainer, sampler, inference manager
    - setup new design patterns within the interface 
    Returns a dictionary of prepared objects (device, tokenizer, dataloaders, model, managers)
    """
    #use singleton 
    resources = SystemResources() # Get singleton instance
    device, tokenizer = resources.setup() # Initialize resources
    
    #get config from factory 
    cfg = GPT2ModelFactory.get_preset_config(model_size_enum) # Get config from factory

    nltk.download("brown", quiet=True) #NLTK corpora is downloaded to be available 
    nltk.download("reuters", quiet=True) 
    if cfg.get("extra_corpora"): #check if extras needed (large model)
        nltk.download("gutenberg", quiet=True) 
        nltk.download("inaugural", quiet=True) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #test hardware 
    print("Using device:", device)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '<pad>'}) #input padding for blocks 
    
    ds_mgr = DatasetManager() # Init dataset manager
    pp_mgr = PreprocessingManager() # Init preprocessing manager
    builder = ModelBuilder() # init builder
    model = builder.build_model(tokenizer, model_size_enum, cfg["block_size"]) 
    model.to(device) #move to device, moves model params and buffers to the target device and "arms" it to anticipate inputs 
    
    texts = ds_mgr.load_texts_from_nltk(limit=cfg.get("nltk_limit"), repeat=cfg.get("nltk_repeat", 1), extra_corpora=cfg.get("extra_corpora", False)) #load texts (combined from before)
    texts += ds_mgr.load_small_c4(limit=cfg.get("c4_limit")) #load C4 texts and append
    print(f"Loaded {len(texts)} total text samples.") #confirm length
    
    if len(texts) == 0: raise RuntimeError("No text loaded.") #raise error if empty
    
    vocab_size = tokenizer.vocab_size #validate size of tokenizer to validate IDs 
    blocks = pp_mgr.build_token_blocks(tokenizer, texts, block_size=cfg["block_size"]) #convert test to token IDs and configure to blocks for processing (lists of integers)
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
    
    terminal = ColoredTerminalAdapter(StandardTerminal()) #init adapter terminal

    #decorator for adapter
    decorated_model = ModelDecorator(model)
    #pass the terminal adapter to the decorator so it can log using the adapter
    decorated_model.add_decoration(PerformanceMonitorDecorator(terminal_adapter=terminal)) #add monitor 

    optimizer_mgr = OptimizerManager(model, cfg["lr"], cfg.get("use_amp", True), device) 
    trainer = TrainingManager(cfg, device) 
    sampler = Sampler(seed=cfg.get("seed", 42)) 
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
        "terminal": terminal,
        "blocks": blocks,      
        "texts_count": len(texts) 
    }

#------------MAIN STARTS HERE--------------
def main(): 
    #factory choice 
    size_enum = GPT2ModelFactory.get_user_choice() #prev setup occurs in function 
    
    try: #[debug] attempt setup 
        objs = setup(size_enum) 
    except Exception as e: 
        print(f"Setup failed: {e}")
        import traceback #import traceback
        traceback.print_exc() #print full stack trace
        return 

    terminal = objs["terminal"] #get terminal
    terminal.display_message("\n=== Starting Training Implementing 5 Design Patterns ===", 'green') 

    objs["trainer"].train(
        objs["model"], objs["train_dl"], objs["val_dl"], objs["test_dl"], objs["optimizer_mgr"]
    )

    
    terminal.display_message("\n=== Testing Inference (Using Command Queue) ===", 'purple') #print inference header
    invoker = CommandInvoker() #init command invoker
    
    test_prompts = [
        "Hello, my name is ",
        "The current economy",
        "The quick brown fox",
        "The mitochondria is the powerhouse of the cell"
    ]
    
    print("\n----- TEST OUTPUTS -----\n")
    
    for p in test_prompts: #queue prompts 
        cmd = GenerateRequestCommand(objs["inference"], p, terminal) #create command and add to queue 
        invoker.add_command(cmd) 
        
    invoker.process_queue() # execute 

if __name__ == "__main__": #start script 
    main() 