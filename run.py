import os
os.environ["KERAS_BACKEND"] = "torch"

import sys
import numpy as np
import keras
from tokenizer import Token, tokenize
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Configuration
MODEL_FILE = "MiniscuLM-1-mini.keras"
EMBEDDING_DIM = 128
CONTEXT_SIZE = 64
INPUT_DIM = CONTEXT_SIZE * (EMBEDDING_DIM + 1)

# Custom Layers (Must match train.py)
@keras.saving.register_keras_serializable()
class LastTokenLayer(keras.layers.Layer):
    def call(self, x):
        return x[:, -1, :]

@keras.saving.register_keras_serializable()
class NormalizationLayer(keras.layers.Layer):
    def call(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=1)

def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()

def main():
    # 1. Parse Arguments
    if len(sys.argv) < 2:
        print("Usage: python3 run.py \"prompt\" [max_tokens] [temperature]")
        return

    prompt = sys.argv[1]
    max_tokens = 128
    temperature = 0.7

    if len(sys.argv) > 2:
        try:
            max_tokens = int(sys.argv[2])
        except ValueError:
            print("Invalid max_tokens argument. Using default 128.")

    if len(sys.argv) > 3:
        try:
            temperature = float(sys.argv[3])
        except ValueError:
            print("Invalid temperature argument. Using default 0.7.")

    # 2. Load Model
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found.")
        return
    
    try:
        # safe_mode=False is required because we use a Lambda layer for normalization
        # We also need to pass custom objects if they weren't registered globally (but @register handles it usually)
        model = keras.models.load_model(MODEL_FILE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Load Tokenizer and Prepare Vocab
    # Trigger tokenizer loading
    tokenize("test")
    
    if not hasattr(tokenize, "vocab_map"):
        print("Error: Failed to load tokenizer. Make sure tokenizer.pkl or tokenizer_min.pkl exists.")
        return

    vocab = list(tokenize.vocab_map.values())
    vocab_embeddings = np.array([t.embedding for t in vocab])
    
    # 4. Define Special Tokens (same logic as train.py)
    min_emb = np.min(vocab_embeddings)
    max_emb = np.max(vocab_embeddings)
    scale = max(abs(min_emb), abs(max_emb)) * 5
    
    user_end_embedding = np.ones(EMBEDDING_DIM) * scale
    assistant_end_embedding = np.ones(EMBEDDING_DIM) * -scale
    
    # Prepare search matrix: Vocab + Assistant End
    # We don't need User End in search because the model shouldn't output it (it's an input marker)
    # But we do need Assistant End to detect stop condition.
    
    search_embeddings = np.vstack([vocab_embeddings, assistant_end_embedding])
    
    # 5. Tokenize Prompt
    prompt_tokens = tokenize(prompt)
    
    # 6. Prepare Context Window
    # "use the last 63 tokens, and a "USER END" token"
    context_window = []
    
    # Track absolute position
    current_pos = 1
    
    # Process prompt tokens
    for t in prompt_tokens:
        context_window.append((t.embedding, current_pos))
        current_pos += 1
    
    if len(context_window) > (CONTEXT_SIZE - 1):
        context_window = context_window[-(CONTEXT_SIZE - 1):]
        
    context_window.append((user_end_embedding, current_pos))
    current_pos += 1
    
    # 7. Generation Loop
    generated_count = 0
    
    while generated_count < max_tokens:
        # Prepare Input
        # Pad to 20
        current_input = []
        if len(context_window) < CONTEXT_SIZE:
            # "pad with token items that are all zeros"
            # Padding has position 0
            padding = [np.concatenate([np.zeros(EMBEDDING_DIM), [0]]) for _ in range(CONTEXT_SIZE - len(context_window))]
            
            # Convert context to (36,) vectors
            context_vecs = [np.concatenate([emb, [pos]]) for emb, pos in context_window]
            
            current_input = padding + context_vecs
        else:
            window = context_window[-CONTEXT_SIZE:]
            current_input = [np.concatenate([emb, [pos]]) for emb, pos in window]
            
        # Flatten and reshape for model (1, 720)
        flat_input = np.concatenate(current_input).reshape(1, -1)
        
        # Run Model
        output_embedding = model.predict(flat_input, verbose=0)
        
        # Find closest token
        # output_embedding is (1, 35)
        # search_embeddings is (V+1, 35)
        
        similarities = cosine_similarity(output_embedding, search_embeddings)[0]
        
        # Softmax Sampling
        # Scale similarities to make the distribution sharper before softmax
        # Cosine similarity is [-1, 1].
        # We use a scaling factor to convert cosine similarity to logits
        scaling_factor = 30.0
        logits = similarities * scaling_factor
        
        probs = softmax(logits, temperature=temperature)
        
        # Sample
        best_idx = np.random.choice(len(probs), p=probs)
        
        # Check if it is ASSISTANT_END
        # The last element in search_embeddings is assistant_end_embedding
        if best_idx == len(vocab):
            # It is ASSISTANT_END
            break
        
        # Otherwise it is a vocab token
        best_token = vocab[best_idx]
        
        # Print
        print(best_token.text, end="", flush=True)
        
        # Update Context
        context_window.append((best_token.embedding, current_pos))
        current_pos += 1
        
        # "maybe delete the first element if it's too long"
        # We handle the window slicing in the input preparation, 
        # but to keep the list from growing indefinitely:
        if len(context_window) > CONTEXT_SIZE:
            context_window.pop(0)
            
        generated_count += 1

    print() # Newline at end

if __name__ == "__main__":
    main()
