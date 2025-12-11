import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import json
import pickle
import numpy as np
import keras
from keras import layers
from tokenizer import Token, tokenize
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

# Argument Parsing
parser = argparse.ArgumentParser(description='Train MiniscuLM')
parser.add_argument('--no-discriminate', action='store_true', help='Disable discriminator and adversarial training')
args = parser.parse_args()

# Configuration
TRAINING_DATA_DIR = "Training_Data"
MODEL_FILE = "MiniscuLM-1-mini.keras"
BATCH_SIZE = 32
EPOCHS = 20
EMBEDDING_DIM = 35
CONTEXT_SIZE = 20
INPUT_DIM = CONTEXT_SIZE * (EMBEDDING_DIM + 1) # 36 dimensions per token (35 embedding + 1 position)

# Load Tokenizer and Embeddings
print("Loading tokenizer...")
# Trigger tokenizer loading
tokenize("test")

if not hasattr(tokenize, "vocab_map"):
    print("Error: Failed to load tokenizer. Make sure tokenizer.pkl exists.")
    exit(1)

vocab = list(tokenize.vocab_map.values())
all_embeddings = np.array([t.embedding for t in vocab])
min_emb = np.min(all_embeddings)
max_emb = np.max(all_embeddings)
print(f"Embeddings range: {min_emb} to {max_emb}")

# Create Special Tokens
# "far from normal words"
scale = max(abs(min_emb), abs(max_emb)) * 5
user_end_embedding = np.ones(EMBEDDING_DIM) * scale
assistant_end_embedding = np.ones(EMBEDDING_DIM) * -scale

print(f"User End Embedding (first 5): {user_end_embedding[:5]}")
print(f"Assistant End Embedding (first 5): {assistant_end_embedding[:5]}")

# Data Loading
def load_training_data():
    train_x = []
    train_y = []
    
    files_to_process = []
    for root, dirs, files in os.walk(TRAINING_DATA_DIR):
        if "Raw" in root: continue
        for file in files:
            if file.endswith(".txt"):
                files_to_process.append(os.path.join(root, file))
    
    print(f"Found {len(files_to_process)} files. Processing...")
    
    for file_path in tqdm(files_to_process):
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
            
            question = content.get("question", "")
            answer = content.get("answer", "")
            
            q_tokens = tokenize(question)
            a_tokens = tokenize(answer)
            
            # Initial Context
            # last 19 items of question + user_end
            context_embeddings = []
            
            # We need to track absolute positions
            # Let's assume the question starts at position 1
            current_pos = 1
            
            # Process question tokens
            for t in q_tokens:
                context_embeddings.append((t.embedding, current_pos))
                current_pos += 1
                
            # Truncate to last 19 if needed, but keep the correct positions?
            # The prompt implies we just take the last 19 tokens.
            if len(context_embeddings) > 19:
                context_embeddings = context_embeddings[-19:]
            
            # Add user_end
            context_embeddings.append((user_end_embedding, current_pos))
            current_pos += 1
            
            # Loop through answer tokens
            for t in a_tokens:
                # Prepare Input
                # Pad to 20
                current_input = []
                if len(context_embeddings) < 20:
                    # Padding has position 0
                    padding = [np.concatenate([np.zeros(EMBEDDING_DIM), [0]]) for _ in range(20 - len(context_embeddings))]
                    
                    # Convert context to (36,) vectors
                    context_vecs = [np.concatenate([emb, [pos]]) for emb, pos in context_embeddings]
                    
                    current_input = padding + context_vecs
                else:
                    # Take last 20
                    window = context_embeddings[-20:]
                    current_input = [np.concatenate([emb, [pos]]) for emb, pos in window]
                
                # Flatten
                flat_input = np.concatenate(current_input)
                train_x.append(flat_input)
                train_y.append(t.embedding)
                
                # Update Context
                context_embeddings.append((t.embedding, current_pos))
                current_pos += 1
                if len(context_embeddings) > 20:
                    context_embeddings.pop(0)
            
            # After loop: predict assistant_end
            current_input = []
            if len(context_embeddings) < 20:
                padding = [np.concatenate([np.zeros(EMBEDDING_DIM), [0]]) for _ in range(20 - len(context_embeddings))]
                context_vecs = [np.concatenate([emb, [pos]]) for emb, pos in context_embeddings]
                current_input = padding + context_vecs
            else:
                window = context_embeddings[-20:]
                current_input = [np.concatenate([emb, [pos]]) for emb, pos in window]
            
            flat_input = np.concatenate(current_input)
            train_x.append(flat_input)
            train_y.append(assistant_end_embedding)
            
        except Exception as e:
            pass
            
    return np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.float32)

print("Generating training data...")
X, Y = load_training_data()
print(f"Total examples: {len(X)}")

if len(X) == 0:
    print("No training data found. Exiting.")
    exit(1)

# Split Train/Test (10% test)
indices = np.arange(len(X))
np.random.shuffle(indices)
split_idx = int(len(X) * 0.9)
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Convert to Torch Tensors and DataLoader
X_train_torch = torch.tensor(X_train)
Y_train_torch = torch.tensor(Y_train)
train_dataset = TensorDataset(X_train_torch, Y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

X_test_torch = torch.tensor(X_test)
Y_test_torch = torch.tensor(Y_test)
test_dataset = TensorDataset(X_test_torch, Y_test_torch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Models
def create_generator():
    # Input is flattened (BATCH, 720)
    # We need to reshape to (BATCH, 20, 36)
    inputs = layers.Input(shape=(INPUT_DIM,))
    reshaped = layers.Reshape((CONTEXT_SIZE, EMBEDDING_DIM + 1))(inputs)
    
    # Transformer Block
    # Attention
    # 36 dimensions. 4 heads = 9 dim per head.
    att = layers.MultiHeadAttention(num_heads=4, key_dim=9)(reshaped, reshaped)
    att = layers.Dropout(0.1)(att)
    out1 = layers.LayerNormalization(epsilon=1e-6)(reshaped + att)
    
    # FFN
    ffn = layers.Dense(128, activation="silu")(out1)
    ffn = layers.Dense(EMBEDDING_DIM + 1)(ffn)
    ffn = layers.Dropout(0.1)(ffn)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    
    # Flatten and Output
    x = layers.Flatten()(out2)
    x = layers.Dense(256, activation="silu")(x)
    outputs = layers.Dense(EMBEDDING_DIM)(x) # Output 35 (embedding only, no position)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="generator")
    return model

def create_discriminator():
    # Input: Context (720) + Candidate (35) = 755
    model = keras.Sequential([
        layers.Input(shape=(INPUT_DIM + EMBEDDING_DIM,)),
        layers.Dense(250, activation='silu'),
        layers.Dense(400, activation='silu'),
        layers.Dense(500, activation='silu'),
        layers.Dense(300, activation='silu'),
        layers.Dense(1, activation='sigmoid') # Real/Fake
    ], name="discriminator")
    return model

generator = create_generator()
discriminator = create_discriminator()

# Optimizers
gen_optimizer = keras.optimizers.Adam(learning_rate=0.001)
disc_optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Get device
# Keras might initialize variables lazily, so we run a dummy forward pass to ensure initialization
dummy_input = torch.zeros((1, INPUT_DIM))
generator(dummy_input)
discriminator(torch.zeros((1, INPUT_DIM + EMBEDDING_DIM)))

device = generator.trainable_variables[0].value.device
print(f"Model is on device: {device}")

# Loss Functions
def cosine_similarity_loss(y_true, y_pred):
    # y_true, y_pred are torch tensors
    y_true_norm = torch.nn.functional.normalize(y_true, p=2, dim=1)
    y_pred_norm = torch.nn.functional.normalize(y_pred, p=2, dim=1)
    return -torch.mean(torch.sum(y_true_norm * y_pred_norm, dim=1))

bce_loss = keras.losses.BinaryCrossentropy(from_logits=False)

# Training Loop
print("Starting training...")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # Stage 1: Supervised
    print("Stage 1: Supervised Training")
    progbar = keras.utils.Progbar(len(train_loader))
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward
        y_pred = generator(x_batch)
        loss = cosine_similarity_loss(y_batch, y_pred)
        
        # Backward
        generator.zero_grad()
        loss.backward()
        
        # Update
        grads = [v.value.grad for v in generator.trainable_variables]
        gen_optimizer.apply(grads, generator.trainable_variables)
        
        progbar.add(1, values=[("sup_loss", loss.item())])
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    # Stage 2: Discriminator
    if not args.no_discriminate:
        print("Stage 2: Discriminator Training")
        progbar = keras.utils.Progbar(len(train_loader))
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Generate fake data
            # We don't want gradients for generator here
            with torch.no_grad():
                y_fake = generator(x_batch)
            
            # Concatenate context + y
            real_input = torch.cat([x_batch, y_batch], dim=1)
            fake_input = torch.cat([x_batch, y_fake], dim=1)
            
            # Labels
            real_labels = torch.ones((x_batch.shape[0], 1), device=device)
            fake_labels = torch.zeros((x_batch.shape[0], 1), device=device)
            
            # Forward
            real_preds = discriminator(real_input)
            fake_preds = discriminator(fake_input)
            
            loss_real = bce_loss(real_labels, real_preds)
            loss_fake = bce_loss(fake_labels, fake_preds)
            total_loss = (loss_real + loss_fake) / 2
            
            # Backward
            discriminator.zero_grad()
            total_loss.backward()
            
            # Update
            grads = [v.value.grad for v in discriminator.trainable_variables]
            disc_optimizer.apply(grads, discriminator.trainable_variables)
            
            progbar.add(1, values=[("disc_loss", total_loss.item())])
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        # Stage 3: Adversarial
        print("Stage 3: Adversarial Training")
        progbar = keras.utils.Progbar(len(train_loader))
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            # y_batch not needed for adversarial loss, but we need x_batch
            
            # Train generator to fool discriminator
            
            # Forward Generator
            y_fake = generator(x_batch)
            
            # Forward Discriminator
            fake_input = torch.cat([x_batch, y_fake], dim=1)
            fake_preds = discriminator(fake_input)
            
            # We want fake_preds to be 1
            loss = bce_loss(torch.ones((x_batch.shape[0], 1), device=device), fake_preds)
            
            # Backward
            generator.zero_grad()
            discriminator.zero_grad() # Clear disc grads just in case
            loss.backward()
            
            # Update Generator only
            grads = [v.value.grad for v in generator.trainable_variables]
            gen_optimizer.apply(grads, generator.trainable_variables)
            
            progbar.add(1, values=[("adv_loss", loss.item())])
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        
    # Save Checkpoint
    generator.save(MODEL_FILE)
    print(f"Saved checkpoint to {MODEL_FILE}")
    
    # Evaluate on Test (Supervised Loss)
    test_loss = 0
    steps = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = generator(x_batch)
            loss = cosine_similarity_loss(y_batch, y_pred)
            test_loss += loss.item()
            steps += 1
    print(f"Test Loss (Cosine Similarity): {test_loss/steps:.4f}")
