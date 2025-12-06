"""
GPT-2 Gateway

Sources:
-Streamlit: https://docs.streamlit.io/develop/api-reference 
-Streamlit capabilities example: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
-Requests library: https://pypi.org/project/requests/ 
-Requests library (2): https://requests.readthedocs.io/en/latest/ 
-Keep chat log history using Streamlit: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
-Streamlit chat elements: https://docs.streamlit.io/develop/api-reference/chat 
-Time library for latency checking: https://docs.python.org/3/library/time.html
"""
#must run "python -m streamlit run testfile2.py" to get localhost up 

import streamlit as st #UI library for web apps as suggested 
import requests #HTTP client lib for Flask API comms 
import time #import for latency check 

API_URL = "http://127.0.0.1:5000/predict" #localhost, default Flask port 

MODEL_CONFIGS = { #configs to match training sizes for models in response calls 
    "Small Model": {
        "path": "M4_small_model.pt",
        "block_size": 128,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256
    },
    "Medium Model": {
        "path": "M4_medium_model.pt", #path to specific model (same dir)
        "block_size": 128, #size of input (how many tokens model can see at once)
        "n_layer": 4, #number of transformer blocks (keep consistent, trained with 4)
        "n_head": 4, #number of heads per transformer block
        "n_embd": 256 #embedding dimension within model
    },
    "Large Model": { #placeholder as no large model was ever generated -> error guaranteed 
        "path": "M4_large_model.pt",
        "block_size": 128,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256
    }
}

def get_lm_response(user_input, model_config, max_length=50, top_k=50, temperature=0.9): #same params of inference manager generator for text
    """Send request to endpoint"""
    headers = {'Content-Type': 'application/json'}
    
    payload = { #designate information to be relayed related to the model and procesing params 
        'user_input': user_input, 
        'model_path': model_config['path'],
        'model_config': {
            'block_size': model_config['block_size'],
            'n_layer': model_config['n_layer'],
            'n_head': model_config['n_head'],
            'n_embd': model_config['n_embd']
        },
        'max_length': max_length,
        'top_k': top_k,
        'temperature': temperature
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers) #post is sent ot Flask API, encode as JSON 
        response.raise_for_status() #check for HTTP codes 
        data = response.json()
        if 'error' in data:
            return f"Server Error: {data['error']}", None
        return data.get('output'), data.get('model_used')
        
    except requests.exceptions.ConnectionError: #define exceptions 
        return "Could not connect to localhost:5000, check endpoint", None
    except requests.exceptions.RequestException as e:
        return f"Communication Error: {e}", None

def main():
    st.set_page_config(
        page_title="EECE4520 GPT2 Gateway",
        layout="centered"
    )
    st.title("EECE4520 GPT2 Gateway")
    st.markdown("Chat interface interacting with flask endpt")
    with st.sidebar: #sidebar to designate model and params 
        st.header("Model Settings")
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(MODEL_CONFIGS.keys()),
            index=1  #default = medium model
        )
        model_config = MODEL_CONFIGS[selected_model_name]
        st.divider()
        st.subheader("Model Architecture") #architecture properties 
        st.info(f"""
        **Configuration for {selected_model_name}:**
        - Block Size: {model_config['block_size']}
        - Layers: {model_config['n_layer']}
        - Attention Heads: {model_config['n_head']}
        - Embedding Dim: {model_config['n_embd']}
        - Model File: {model_config['path']}
        """)
        st.divider()
        st.subheader("Generation Parameters") #generation params 
        st.caption("These control text generation, not model architecture")
        max_length = st.slider("Max Length", min_value=10, max_value=125, value=50) #sliders to adjust text generation params for fun 
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
        top_k = st.slider("Top-K", min_value=1, max_value=100, value=50)
    with st.sidebar:  # check throughput / correctness / robustness
        st.divider()
        if st.button("Testing endpoint..."):

            st.write("Running Endpoint Diagnostics")
            start_time = time.time() #test latency
            try:
                response = requests.get("http://127.0.0.1:5000/")
                latency_ms = (time.time() - start_time) * 1000
                if response.status_code == 200:
                    st.success(f"Endpoint is up (Latency: {latency_ms:.2f} ms)")
                else:
                    st.error(f"Endpoint responded with status {response.status_code}")
                try: #test correctness (json fields are filled in where needed)
                    data = response.json()
                    required_keys = ["status", "device", "current_model", "current_config"]
                    missing = [k for k in required_keys if k not in data]

                    if missing:
                        st.warning(f"Missing keys in response: {missing}")
                    else:
                        if data.get("status") == "running":
                            st.success("Correctness check passed.")
                        else:
                            st.warning("Unexpected status value")

                    st.json(data)

                except ValueError:
                    st.error("Response was not valid JSON")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to endpoint")
                st.stop()
        st.write("Robustness Testing") #test robustness of system, sends some categories of bad data and sees if backend fails safely (NO CRASH)
        malformed_tests = [
            ("Missing user_input", {"model_path": "M4_small_model.pt", "model_config": {}}),
            ("Invalid types", {"user_input": 12345, "model_path": True, "model_config": "not a dict"}),
            ("Empty payload", {}),
            ("Missing model_config", {"user_input": "hello"}),
        ]
        for test_name, payload in malformed_tests:
            try:
                resp = requests.post("http://127.0.0.1:5000/predict", json=payload)
                st.write(f"**{test_name}** → Status: {resp.status_code}")

                # Expecting fail responses (400 or 500)
                if resp.status_code in (400, 500):
                    st.success(f"Expected failure response received: {resp.status_code}")
                else:
                    st.warning(f"Unexpected response: {resp.status_code}")

                try:
                    st.json(resp.json())
                except:
                    st.error("Response was not JSON")

            except Exception as e:
                st.error(f"Error during robustness test '{test_name}': {e}")
    if "messages" not in st.session_state: #initialize message session 
        st.session_state.messages = []
    for message in st.session_state.messages: #past history between user and server is kept 
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "model_used" in message:
                st.caption(f"Model: {message['model_used']}")
    if user_input := st.chat_input("Type a sentence you would like completed or progressed"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): #inital message is put into chat for user ref, take icons from st for user and model
            st.markdown(user_input)
        with st.chat_message("assistant"): #generate and return response 
            with st.spinner("Waiting for response from API..."): #pass params once API response is given 
                response, model_used = get_lm_response(
                    user_input, 
                    model_config,
                    max_length=max_length,
                    top_k=top_k,
                    temperature=temperature
                )
                st.markdown(response) #display response to user 
                if model_used: #designates which model was used for returnval 
                    st.caption(f"Model: {model_used}")
                st.session_state.messages.append({ #add assistant response to chat history, preserve past history 
                    "role": "assistant", 
                    "content": response,
                    "model_used": model_used
                })

if __name__ == '__main__':
    main()