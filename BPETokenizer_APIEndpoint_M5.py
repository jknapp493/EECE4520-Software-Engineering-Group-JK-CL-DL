"""
GPT-2 Endpoint 

Sources:
-MIME/Media types: https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/MIME_types
-HTTP Error Codes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status
-Flask routing docs: https://flask.palletsprojects.com/en/stable/quickstart/#routing
-Flask requesting: https://flask.palletsprojects.com/en/stable/reqcontext/
-Jsonify: https://flask.palletsprojects.com/en/stable/api/#flask.json.jsonify
-General REST API Structure: https://developer.mozilla.org/en-US/docs/Glossary/REST
-REST API Design Guide: https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design
"""

import torch
import torch.nn.functional as F #functional API for inference 
from flask import Flask, request, jsonify #flask web app pieces 
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

app = Flask(__name__) #flask instance 
current_model = None #cache intended model and tokenizer 
current_model_path = None
current_config = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer(): #load tokenizer 
    global tokenizer
    if tokenizer is None:
        print("Loading Tokenizer...")
        try:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.add_special_tokens({'pad_token': '<pad>'}) #explicit token for tokenizer for padding 
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise e
    return tokenizer

def load_model(model_path, model_config):
    """
    Loads the GPT-2 model with the configuration provided by the gateway.
    """
    global current_model, current_model_path, current_config
    if (current_model is not None and #reload config 
        current_model_path == model_path and 
        current_config == model_config):
        print(f"Model already loaded with correct config.")
        return current_model

    print(f"\n{'/'*60}") #model parameters were not saved fully when initially trained, just the weights so need to instantiate params 
    print(f"Loading model from: {model_path}")
    print(f"With configuration:")
    print(f"  - Block Size: {model_config['block_size']}")
    print(f"  - Layers: {model_config['n_layer']}")
    print(f"  - Heads: {model_config['n_head']}")
    print(f"  - Embedding Dim: {model_config['n_embd']}")
    print(f"{'/'*60}\n") #end of initialization window shown to user 
    tok = load_tokenizer()
    vocab_size = tok.vocab_size #load and get tokenizer size (set embedding sizes)
    config = GPT2Config( #build model 
        vocab_size=vocab_size,
        n_positions=model_config['block_size'],
        n_ctx=model_config['block_size'],
        n_embd=model_config['n_embd'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        pad_token_id=tok.pad_token_id
    )
    model = GPT2LMHeadModel(config) #instantiate model class weights 
    model.resize_token_embeddings(vocab_size) #embeddings need to match the size of those from tokenizer 
    try:
        state_dict = torch.load(model_path, map_location=device) #load weights 
        model.load_state_dict(state_dict) #copy into model config 
        model.to(device)
        model.eval()
        current_model = model
        current_model_path = model_path
        current_config = model_config
        print(f"Model loaded successfully on {device}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        return current_model
    except FileNotFoundError: #error establishments 
        print(f"Model file not found at {model_path}")
        return None
    except RuntimeError as e:
        print(f"Error loading weights\n Error details: {e}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_text(model, prompt, max_length=50, top_k=50, temperature=0.9):
    """
    Inference logic - generates text token by token
    """
    tok = load_tokenizer()
    input_ids = torch.tensor([tok.encode(prompt)], device=device) #encode prompt into token IDs 
    with torch.no_grad(): #no gradient tracking 
        for _ in range(max_length):
            outputs = model(input_ids=input_ids) #forward pass -> model output 
            logits = outputs.logits[:, -1, :] / temperature #get logits for last generated pos 
            probs = F.softmax(logits, dim=-1) #softmax layer (convert logits to probabilities)
            top_probs, top_idx = torch.topk(probs, top_k) #top-k sample indices 
            next_token = top_idx[0, torch.multinomial(top_probs[0], 1)]
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1) #append token ID to the sequence for forward pass 
            if next_token.item() == tok.eos_token_id: 
                break
    return tok.decode(input_ids[0].tolist(), skip_special_tokens=True) #return decoded out 
@app.route('/', methods=['GET']) #Flask route handler to map URL
def health_check():
    """Endpoint health check """
    return jsonify({ #jsonify constructs Flask responce with MIME 
        "status": "running", 
        "device": str(device),
        "current_model": current_model_path if current_model 
        else "No model loaded",
        "current_config": current_config if current_config 
        else "No config"
    }), 200 #HTTP meaning for ok 

@app.route('/predict', methods=['POST']) #parses JSON from POST request
def predict():
    data = request.get_json()
    user_input = data.get('user_input', '')
    requested_path = data.get('model_path', 'M4_small_model.pt') #get from config 
    model_config = data.get('model_config')
    max_length = data.get('max_length', 50) #generation params 
    top_k = data.get('top_k', 50)
    temperature = data.get('temperature', 0.9)
    if not user_input:
        return jsonify({"error": "No user input provided"}), 400 #HTTP meaning for bad 
    
    if not model_config:
        return jsonify({"error": "No model config provided"}), 400
    model = load_model(requested_path, model_config) #load model with requested config 
    if model is None:
        return jsonify({
            "error": f"Failed to load model at {requested_path}"
        }), 500 #HTTP internal server error 
    try:
        generated_text = generate_text(
            model, 
            user_input, 
            max_length=max_length,
            top_k=top_k,
            temperature=temperature
        )
        return jsonify({
            "output": generated_text,
            "model_used": requested_path,
            "config_used": model_config
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_tokenizer() #initialize tokenizer and svr 
    print(f"\n{'/'*60}")
    print(f"Flask API Server Starting")
    print(f"Device: {device}")
    print(f"Ready to accept requests on port 5000")
    print(f"{'/'*60}\n")
    app.run(debug=True, port=5000, use_reloader=False) #start svr on port 5000, debug enabled 