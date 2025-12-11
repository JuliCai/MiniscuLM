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
EMBEDDING_DIM = 128
CONTEXT_SIZE = 64
INPUT_DIM = CONTEXT_SIZE * (EMBEDDING_DIM + 1) # 129 dimensions per token (128 embedding + 1 position)

# Load Tokenizer and Embeddings
print("Loading tokenizer...")
# Trigger tokenizer loading
tokenize("test")

if not hasattr(tokenize, "vocab_map"):
    print("Error: Failed to load tokenizer. Make sure tokenizer.pkl or tokenizer_min.pkl exists.")
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
            # last 63 items of question + user_end
            context_embeddings = []
            
            # We need to track absolute positions
            # Let's assume the question starts at position 1
            current_pos = 1
            
            # Process question tokens
            for t in q_tokens:
                context_embeddings.append((t.embedding, current_pos))
                current_pos += 1
                
            # Truncate to last 63 if needed
            if len(context_embeddings) > (CONTEXT_SIZE - 1):
                context_embeddings = context_embeddings[-(CONTEXT_SIZE - 1):]
            
            # Add user_end
            context_embeddings.append((user_end_embedding, current_pos))
            current_pos += 1
            
            # Loop through answer tokens
            for t in a_tokens:
                # Prepare Input
                # Pad to CONTEXT_SIZE
                current_input = []
                if len(context_embeddings) < CONTEXT_SIZE:
                    # Padding has position 0
                    padding = [np.concatenate([np.zeros(EMBEDDING_DIM), [0]]) for _ in range(CONTEXT_SIZE - len(context_embeddings))]
                    
                    # Convert context to (36,) vectors
                    context_vecs = [np.concatenate([emb, [pos]]) for emb, pos in context_embeddings]
                    
                    current_input = padding + context_vecs
                else:
                    # Take last CONTEXT_SIZE
                    window = context_embeddings[-CONTEXT_SIZE:]
                    current_input = [np.concatenate([emb, [pos]]) for emb, pos in window]
                
                # Flatten
                flat_input = np.concatenate(current_input)
                train_x.append(flat_input)
                train_y.append(t.embedding)
                
                # Update Context
                context_embeddings.append((t.embedding, current_pos))
                current_pos += 1
                if len(context_embeddings) > CONTEXT_SIZE:
                    context_embeddings.pop(0)
            
            # After loop: predict assistant_end
            current_input = []
            if len(context_embeddings) < CONTEXT_SIZE:
                padding = [np.concatenate([np.zeros(EMBEDDING_DIM), [0]]) for _ in range(CONTEXT_SIZE - len(context_embeddings))]
                context_vecs = [np.concatenate([emb, [pos]]) for emb, pos in context_embeddings]
                current_input = padding + context_vecs
            else:
                window = context_embeddings[-CONTEXT_SIZE:]
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

# Custom Layers to avoid Lambda serialization issues
@keras.saving.register_keras_serializable()
@keras.saving.register_keras_serializable()
class LastTokenLayer(keras.layers.Layer):
    def call(self, x):
        return x[:, -1, :]

@keras.saving.register_keras_serializable()
class NormalizationLayer(keras.layers.Layer):
    def call(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=1)

@keras.saving.register_keras_serializable()
def split_emb(x):
    return x[:, :, :-1]

@keras.saving.register_keras_serializable()
def split_pos(x):
    return x[:, :, -1]

@keras.saving.register_keras_serializable()
def clamp_pos(x):
    return torch.clamp(x.type(torch.long), 0, 4095)

# Models
def create_generator():
    # Input is flattened (BATCH, 64 * (EMBEDDING_DIM + 1))
    inputs = layers.Input(shape=(INPUT_DIM,))
    reshaped = layers.Reshape((CONTEXT_SIZE, EMBEDDING_DIM + 1))(inputs)
    
    embs = layers.Lambda(split_emb)(reshaped)
    pos = layers.Lambda(split_pos)(reshaped)
    
    # Positional Embedding
    # Cast to int for Embedding layer
    # We use a Lambda with torch.long (since backend is torch)
    # Clamp to max 4095 to avoid index out of bounds
    pos_int = layers.Lambda(clamp_pos)(pos)
    
    # Embedding layer
    pos_embeddings = layers.Embedding(input_dim=4096, output_dim=EMBEDDING_DIM)(pos_int)
    
    # Add
    x = layers.Add()([embs, pos_embeddings])
    
    # Project to d_model = 192
    x = layers.Dense(192, activation="silu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Transformer Blocks (6 layers)
    for _ in range(6):
        # Attention
        # 192 dimensions. 6 heads = 32 dim per head.
        att = layers.MultiHeadAttention(num_heads=6, key_dim=32)(x, x)
        att = layers.Dropout(0.1)(att)
        x = layers.LayerNormalization(epsilon=1e-6)(x + att)
        
        # FFN
        ffn = layers.Dense(768, activation="silu")(x) # 4 * 192
        ffn = layers.Dense(192)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Output Head
    # Take the LAST token's representation instead of averaging
    # This is crucial for next-token prediction
    x = LastTokenLayer()(x)
    
    # Project back to embedding dim
    outputs = layers.Dense(EMBEDDING_DIM)(x) 
    
    # Normalize output to unit length
    outputs = NormalizationLayer()(outputs)
    
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

# Handle Multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move to device BEFORE DataParallel
generator.to(device)
discriminator.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    # generator = torch.nn.DataParallel(generator)
    # discriminator = torch.nn.DataParallel(discriminator)
    print("DataParallel disabled temporarily to debug SegFault")

print(f"Model is on device: {device}")

# Optimizers (Use PyTorch optimizers for custom loop)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

def get_module(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model

# Loss Functions
def cosine_similarity_loss(y_true, y_pred):
    # y_true, y_pred are torch tensors
    y_true_norm = torch.nn.functional.normalize(y_true, p=2, dim=1)
    y_pred_norm = torch.nn.functional.normalize(y_pred, p=2, dim=1)
    return 1 - torch.mean(torch.sum(y_true_norm * y_pred_norm, dim=1))

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
        # Ensure y_pred is on the correct device (fix for potential device mismatch with new layers)
        if hasattr(y_pred, 'device') and y_pred.device != device:
            y_pred = y_pred.to(device)
            
        loss = cosine_similarity_loss(y_batch, y_pred)
        
        # Backward
        gen_optimizer.zero_grad()
        loss.backward()
        gen_optimizer.step()
        
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
            disc_optimizer.zero_grad()
            total_loss.backward()
            disc_optimizer.step()
            
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
            gen_optimizer.zero_grad()
            discriminator.zero_grad() # Clear disc grads just in case
            loss.backward()
            gen_optimizer.step()
            
        # Update Generator only
        # model_to_update = get_module(generator)
        # grads = [v.value.grad for v in model_to_update.trainable_variables]
        # gen_optimizer.apply(grads, model_to_update.trainable_variables)
        
        progbar.add(1, values=[("adv_loss", loss.item())])
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        
    # Save Checkpoint
    get_module(generator).save(MODEL_FILE)
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
