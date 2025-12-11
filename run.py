import os
os.environ["KERAS_BACKEND"] = "torch"

import sys
import numpy as np
import keras
from tokenizer import Token, tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
MODEL_FILE = "MiniscuLM-1-mini.keras"
EMBEDDING_DIM = 35
CONTEXT_SIZE = 20
INPUT_DIM = CONTEXT_SIZE * (EMBEDDING_DIM + 1)

def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()

def main():
    # 1. Parse Arguments
    if len(sys.argv) < 2:
        print("Usage: python3 run.py \"prompt\" [max_tokens]")
        return

    prompt = sys.argv[1]
    max_tokens = 128
    if len(sys.argv) > 2:
        try:
            max_tokens = int(sys.argv[2])
        except ValueError:
            print("Invalid max_tokens argument. Using default 128.")

    # 2. Load Model
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found.")
        return
    
    try:
        model = keras.models.load_model(MODEL_FILE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Load Tokenizer and Prepare Vocab
    # Trigger tokenizer loading
    tokenize("test")
    
    if not hasattr(tokenize, "vocab_map"):
        print("Error: Failed to load tokenizer.")
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
    # "use the last 19 tokens, and a "USER END" token"
    context_window = []
    
    # Track absolute position
    current_pos = 1
    
    # Process prompt tokens
    for t in prompt_tokens:
        context_window.append((t.embedding, current_pos))
        current_pos += 1
    
    if len(context_window) > 19:
        context_window = context_window[-19:]
        
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
        # Convert cosine similarity (-1 to 1) to something more suitable for softmax?
        # Or just apply softmax directly to similarities?
        # Usually logits are unbounded. Cosine sim is bounded.
        # Let's scale it up a bit so temperature works better, or just use as is.
        # A high similarity (0.9) vs low (0.1) -> exp(0.9) vs exp(0.1) -> 2.45 vs 1.1
        # If we multiply by a factor, we sharpen the distribution.
        # Let's try applying softmax directly with a low temperature to sharpen it.
        
        probs = softmax(similarities, temperature=0.1)
        
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
