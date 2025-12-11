import os
import re
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm

class Token:
    def __init__(self, text, embedding, full_embedding=None):
        self.text = text
        self.embedding = embedding
        self.full_embedding = full_embedding

    def __repr__(self):
        return f"Token(text={repr(self.text)}, embedding_dim={len(self.embedding)})"
    
def get_tokens_from_training_data():
    # load training data (all .txt files) from /Training_Data directory
    training_data_dir = "Training_Data"
    
    # charachter set is just the entire qwerty keyboard
    character_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()-_=+[{]}\\|;:'\",<.>/? ’“”"
    character_set_set = set(character_set)
    alphanumeric_set = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    
    tokens = []
    unique_tokens = set()
    
    print("Scanning Training Data...")
    
    all_files = []
    for root, dirs, files in os.walk(training_data_dir):
        for file in files:
            if file.endswith(".txt"):
                all_files.append(os.path.join(root, file))

    for file_path in tqdm(all_files, desc="Reading files", miniters=100):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Decode unicode escape sequences (e.g. \u2019 -> ’)
            content = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), content)
                
            current_token = ""
            token_type = None # 'alpha', 'punct'

            for char in content:
                if char not in character_set_set:
                    continue
                
                if char == ' ':
                    if current_token:
                        current_token += char
                        tokens.append(current_token)
                        unique_tokens.add(current_token)
                        current_token = ""
                        token_type = None
                    else:
                        # Standalone space
                        tokens.append(char)
                        unique_tokens.add(char)
                
                elif char in alphanumeric_set:
                    if token_type == 'punct':
                        tokens.append(current_token)
                        unique_tokens.add(current_token)
                        current_token = ""
                    
                    token_type = 'alpha'
                    current_token += char
                    
                else: # Punctuation
                    if current_token:
                        tokens.append(current_token)
                        unique_tokens.add(current_token)
                        current_token = ""
                        
                    token_type = 'punct'
                    current_token += char
                    
            # Handle any remaining token at EOF
            if current_token:
                tokens.append(current_token)
                unique_tokens.add(current_token)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Total tokens found: {len(tokens)}")
    print(f"Unique tokens found: {len(unique_tokens)}")

    # Add extras to unique_tokens
    print("Adding special tokens...")
    # 1. Each character in character set
    for char in character_set:
        unique_tokens.add(char)
        
    # 2. All double digit numbers
    for i in range(10, 100):
        unique_tokens.add(str(i))

    # Common combinations (Bigrams)
    print("Finding common token combinations...")
    bigram_counts = Counter()
    if tokens:
        # Count bigrams
        # Using list comprehension for zip might be memory intensive, but we need length for tqdm
        # Alternatively, we can just iterate
        bigrams_iter = zip(tokens, tokens[1:])
        bigram_counts = Counter(tqdm(bigrams_iter, total=len(tokens)-1, desc="Counting bigrams", miniters=1000))
        
        # Add top 5000 bigrams to unique_tokens
        for (t1, t2), count in bigram_counts.most_common(5000):
            combined = t1 + t2
            unique_tokens.add(combined)

    # Pruning to 200k
    print("Pruning vocabulary...")
    token_counts = Counter(tokens)
    
    final_vocab = []
    
    # 1. Add all single characters (mandatory)
    for char in character_set:
        final_vocab.append(char)
        
    vocab_set = set(final_vocab)

    # 2. Add common words from commonwords.txt
    try:
        with open("commonwords.txt", "r") as f:
            common_words = [line.strip() for line in f if line.strip()]
        
        print(f"Adding {len(common_words)} common words (and variants)...")
        for word in common_words:
            # Add word
            if word not in vocab_set:
                 final_vocab.append(word)
                 vocab_set.add(word)
            # Add word + space
            word_space = word + " "
            if word_space not in vocab_set:
                 final_vocab.append(word_space)
                 vocab_set.add(word_space)
                 
    except FileNotFoundError:
        print("Warning: commonwords.txt not found. Skipping.")
    
    # 3. Prepare list of candidates (excluding already added)
    candidate_counts = {}
    for t in tqdm(unique_tokens, desc="Filtering candidates", miniters=1000):
        if t in vocab_set:
            continue
        candidate_counts[t] = token_counts.get(t, 0)
        
    if tokens:
        for (t1, t2), count in bigram_counts.most_common(5000):
            combined = t1 + t2
            if combined in candidate_counts:
                # Use the bigram count if it's higher (it implies we treat it as a unit)
                candidate_counts[combined] = max(candidate_counts[combined], count)

    # Sort candidates by count desc
    sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Fill up to 200k
    remaining_slots = 200000 - len(final_vocab)
    for t, count in sorted_candidates:
        if remaining_slots <= 0:
            break
        final_vocab.append(t)
        remaining_slots -= 1
        
    print(f"Final vocabulary size: {len(final_vocab)}")

    # Embeddings
    print("Generating embeddings with GitHub Models (openai/text-embedding-3-small)...")
    
    # Load API Key
    try:
        githubaikey = open("githubkey.apikeys").read().strip()
    except FileNotFoundError:
        print("Error: githubkey.apikeys file not found.")
        return

    endpoint = "https://models.inference.ai.azure.com"
    model_name = "text-embedding-3-small"

    client = EmbeddingsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(githubaikey)
    )

    # Check for existing tokenizer.pkl to load cached embeddings
    existing_embeddings_map = {}
    if os.path.exists("tokenizer.pkl"):
        print("Loading existing tokenizer.pkl to check for cached embeddings...")
        try:
            with open("tokenizer.pkl", "rb") as f:
                existing_tokens = pickle.load(f)
                for t in existing_tokens:
                    if hasattr(t, 'full_embedding') and t.full_embedding is not None:
                        existing_embeddings_map[t.text] = t.full_embedding
        except Exception as e:
            print(f"Could not load existing tokenizer: {e}")

    # Identify tokens that need embedding
    tokens_to_embed = []
    for t in final_vocab:
        if t not in existing_embeddings_map:
            tokens_to_embed.append(t)
            
    print(f"Tokens to embed: {len(tokens_to_embed)} (Cached: {len(final_vocab) - len(tokens_to_embed)})")

    new_embeddings_map = {}
    batch_size = 1000 # Larger batch size for API
    
    try:
        if tokens_to_embed:
            total_batches = (len(tokens_to_embed) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(tokens_to_embed), batch_size), desc="Embedding batches", total=total_batches, miniters=1):
                batch = tokens_to_embed[i:i+batch_size]
                
                response = client.embed(
                    input=batch,
                    model=model_name
                )
                
                # Extract embeddings in order
                for j, item in enumerate(response.data):
                    new_embeddings_map[batch[j]] = item.embedding

        # Combine all embeddings in order of final_vocab
        all_full_embeddings = []
        final_token_texts = []
        
        for t in final_vocab:
            if t in existing_embeddings_map:
                all_full_embeddings.append(existing_embeddings_map[t])
                final_token_texts.append(t)
            elif t in new_embeddings_map:
                all_full_embeddings.append(new_embeddings_map[t])
                final_token_texts.append(t)
            
        # PCA
        print("Reducing dimensions with PCA (35D)...")
        from sklearn.decomposition import PCA
        X = np.array(all_full_embeddings)
        
        # Handle case where we have fewer samples than components
        n_components = min(35, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        # Create Token instances
        token_instances = []
        for text, full_emb, reduced_emb in zip(final_token_texts, all_full_embeddings, X_reduced):
            token_instances.append(Token(text, reduced_emb, full_embedding=full_emb))

        # Save
        output_file = "tokenizer.pkl"
        print(f"Saving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(token_instances, f)

        # Save minified version (no full embeddings)
        output_file_min = "tokenizer_min.pkl"
        print(f"Saving minified to {output_file_min}...")
        token_instances_min = [Token(t.text, t.embedding, full_embedding=None) for t in token_instances]
        with open(output_file_min, 'wb') as f:
            pickle.dump(token_instances_min, f)
            
        print("Tokenizer creation complete.")

    except Exception as e:
        print(f"Error during embedding/PCA/saving: {e}")

def tokenize(text):
    if not hasattr(tokenize, "vocab_map"):
        # Fix for pickle loading if saved from __main__
        import sys
        import __main__
        if not hasattr(__main__, 'Token'):
            __main__.Token = Token

        vocab = None
        try:
            with open("tokenizer.pkl", "rb") as f:
                vocab = pickle.load(f)
        except FileNotFoundError:
            try:
                with open("tokenizer_min.pkl", "rb") as f:
                    vocab = pickle.load(f)
            except FileNotFoundError:
                pass
        
        if vocab is None:
            print("Tokenizer not found. Please run get_tokens_from_training_data() first.")
            return []

        tokenize.vocab_map = {t.text: t for t in vocab}
        tokenize.vocab_map_lower = {t.text.lower(): t for t in vocab}
        tokenize.max_token_len = max(len(t.text) for t in vocab)

    vocab_map = tokenize.vocab_map
    vocab_map_lower = tokenize.vocab_map_lower
    max_len = tokenize.max_token_len
    
    tokens = []
    i = 0
    n = len(text)
    
    while i < n:
        best_match = None
        best_match_len = 0
        
        # Check substrings starting at i, from longest to shortest
        for length in range(min(max_len, n - i), 0, -1):
            substring = text[i : i + length]
            
            # Try case-sensitive match first
            if substring in vocab_map:
                best_match = vocab_map[substring]
                best_match_len = length
                break
            
            # Try case-insensitive match
            substring_lower = substring.lower()
            if substring_lower in vocab_map_lower:
                best_match = vocab_map_lower[substring_lower]
                best_match_len = length
                break
        
        if best_match:
            tokens.append(best_match)
            i += best_match_len
        else:
            # Unknown character, skip
            i += 1
            
    return tokens

if __name__ == "__main__":
    get_tokens_from_training_data()