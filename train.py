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
from urllib.parse import urlparse
from pathlib import PurePosixPath
import urllib.request
import sys
import platform
import time
import gc

# Argument Parsing
parser = argparse.ArgumentParser(description='Train MiniscuLM')
parser.add_argument('-discriminate', action='store_true', help='Enable discriminator and adversarial training')
parser.add_argument('-pretrain_dir', type=str, default=os.path.join("Training_Data", "Pretrain"), help='Path to pretraining data folder (raw-only for now)')
parser.add_argument('-sft_dir', type=str, default=os.path.join("Training_Data", "SFT"), help='Path to SFT fine-tuning data folder')
parser.add_argument('-pretrain_epochs', type=int, default=20, help='Epochs to pretrain on pretrain_dir')
parser.add_argument('-sft_epochs', type=int, default=5, help='Epochs to fine-tune on sft_dir')
parser.add_argument('-no_pretrain', action='store_true', help='Skip pretraining phase')
parser.set_defaults(use_parquets=True)
parser.add_argument('--no-parquets', dest='use_parquets', action='store_false', help='Disable reading .parquet files under <data_dir>/Parquets')
parser.add_argument('-parquet_urls_file', type=str, default=None, help='Optional path to a text file of parquet URLs to ensure exist (one URL per line)')
parser.add_argument('-parquet_allow_download', action='store_true', help='Allow downloading missing parquets listed in parquet_urls_file')
parser.add_argument('-parquet_dry_run', action='store_true', help='When set, do not download; only print what would happen')
parser.add_argument('-parquet_max_rows', type=int, default=20000, help='Max parquet rows to consume per file (prevents huge RAM usage); set 0 to disable limit')
parser.add_argument('-max_examples_total', type=int, default=0, help='Hard cap on total training examples generated per dataset (0 disables cap)')
parser.add_argument('-max_examples_per_file', type=int, default=0, help='Hard cap on training examples generated per input file (0 disables cap)')
parser.add_argument(
    '-highvram',
    action='store_true',
    help='Enable high-VRAM throughput preset (bigger batch, dataset on GPU, TF32/bf16 autocast, reduced logging sync).'
)
parser.add_argument(
    '-batch_size',
    type=int,
    default=0,
    help='Override batch size (0 uses default: 32, or 1024 when -highvram is set).'
)
args = parser.parse_args()

if args.parquet_dry_run:
    print("Note: -parquet_dry_run implies --no-parquets (will not ingest parquet files).")

# Configuration
MODEL_FILE = "MiniscuLM-1-mini.keras"
if args.batch_size and args.batch_size > 0:
    BATCH_SIZE = int(args.batch_size)
else:
    BATCH_SIZE = 1024 if args.highvram else 32
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
def load_training_data(data_dir, include_raw=True, include_qa=True, return_stats=False):
    train_x = []
    train_y = []

    stats = {
        "data_dir": data_dir,
        "files_found": 0,
        "files_processed": 0,
        "txt_files": 0,
        "parquet_files": 0,
        "raw_docs": 0,
        "qa_pairs": 0,
        "raw_tokens": 0,
        "qa_question_tokens": 0,
        "qa_answer_tokens": 0,
        "target_tokens": 0,
        "errors": 0,
        "stopped_early": False,
        "stop_reason": "",
    }

    def append_example(flat_input, target_embedding):
        train_x.append(flat_input)
        train_y.append(target_embedding)
        stats["target_tokens"] += 1

    def should_stop():
        return bool(args.max_examples_total and stats["target_tokens"] >= args.max_examples_total)

    def maybe_stop(reason):
        if not stats.get("stopped_early"):
            stats["stopped_early"] = True
            stats["stop_reason"] = reason

    def build_context_matrix(context_embeddings):
        """Return (CONTEXT_SIZE, EMBEDDING_DIM+1) float32 matrix for current context."""
        mat = np.zeros((CONTEXT_SIZE, EMBEDDING_DIM + 1), dtype=np.float32)
        if not context_embeddings:
            return mat
        if len(context_embeddings) <= CONTEXT_SIZE:
            start = CONTEXT_SIZE - len(context_embeddings)
            for j, (emb, pos) in enumerate(context_embeddings):
                mat[start + j, :-1] = emb
                mat[start + j, -1] = pos
        else:
            window = context_embeddings[-CONTEXT_SIZE:]
            for j, (emb, pos) in enumerate(window):
                mat[j, :-1] = emb
                mat[j, -1] = pos
        return mat

    # Parquet behavior:
    # - Parquets are enabled by default.
    # - --no-parquets disables parquet ingestion.
    # - -parquet_dry_run implies --no-parquets (skip ingestion) but can still print what would happen.
    parquet_manage_enabled = args.use_parquets or args.parquet_dry_run
    parquet_ingest_enabled = args.use_parquets and (not args.parquet_dry_run)

    def is_under_folder(path, folder_name):
        parts = os.path.normpath(path).split(os.sep)
        return folder_name in parts

    def filename_from_url(url):
        parsed = urlparse(url.strip())
        return PurePosixPath(parsed.path).name

    def ensure_external_parquets(parquets_dir, urls_file, allow_download=False, dry_run=True):
        if not urls_file or not os.path.exists(urls_file):
            return

        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f.readlines()]

        for url in urls:
            if not url or url.startswith('#'):
                continue

            fname = filename_from_url(url)
            if not fname:
                print(f"Parquet URL skipped (could not parse filename): {url}")
                continue

            dest_path = os.path.join(parquets_dir, fname)
            if os.path.exists(dest_path):
                print(f"Parquet file found: {fname}")
                continue

            if not allow_download:
                print(f"Parquet missing (download disabled): {fname}")
                continue

            if dry_run:
                print(f"Parquet missing (dry-run): would download {fname} from {url}")
                continue

            print(f"Downloading parquet: {fname}")
            os.makedirs(parquets_dir, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, dest_path)
                print(f"Downloaded parquet to: {dest_path}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")

    def normalize_text(value):
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8', errors='ignore')
            except Exception:
                return ""
        return str(value)

    def iter_parquet_examples(parquet_path, max_rows):
        # Yields tuples:
        #   ("raw", text) or ("qa", question, answer)
        try:
            import pyarrow.parquet as pq
        except Exception:
            print("pyarrow is not installed; skipping parquet files. Install with: pip install pyarrow")
            return

        try:
            pf = pq.ParquetFile(parquet_path)
        except Exception as e:
            print(f"Failed to open parquet {parquet_path}: {e}")
            return

        rows_yielded = 0
        batch_size = 1024
        for batch in pf.iter_batches(batch_size=batch_size):
            table = batch.to_pydict()
            columns = list(table.keys())
            num_rows = len(next(iter(table.values()))) if table else 0

            # helper to get column value
            def col(name, idx):
                v = table.get(name)
                if v is None:
                    return None
                return v[idx]

            for i in range(num_rows):
                if max_rows and max_rows > 0 and rows_yielded >= max_rows:
                    return

                # Heuristic 1: plain text
                for text_col in ("text", "content", "document", "article"):
                    if text_col in columns:
                        text = normalize_text(col(text_col, i))
                        if text:
                            rows_yielded += 1
                            yield ("raw", text)
                        break
                else:
                    # Heuristic 2: explicit QA-ish pairs
                    if "question" in columns and "answer" in columns:
                        q = normalize_text(col("question", i))
                        a = normalize_text(col("answer", i))
                        if q and a:
                            rows_yielded += 1
                            yield ("qa", q, a)
                        continue

                    pair_specs = [
                        ("prompt", "response"),
                        ("prompt", "completion"),
                        ("instruction", "output"),
                        ("instruction", "response"),
                        ("input", "output"),
                    ]
                    matched = False
                    for qk, ak in pair_specs:
                        if qk in columns and ak in columns:
                            q = normalize_text(col(qk, i))
                            a = normalize_text(col(ak, i))
                            if q and a:
                                rows_yielded += 1
                                yield ("qa", q, a)
                            matched = True
                            break
                    if matched:
                        continue

                    # Heuristic 3: chat-style messages
                    for chat_col in ("messages", "conversations"):
                        if chat_col in columns:
                            msgs = col(chat_col, i)
                            # Expect list[dict] like {role, content}
                            if isinstance(msgs, list):
                                user_text = None
                                assistant_text = None
                                for m in msgs:
                                    if not isinstance(m, dict):
                                        continue
                                    role = (m.get("role") or m.get("from") or "").lower()
                                    content = m.get("content") or m.get("value") or m.get("text")
                                    content = normalize_text(content)
                                    if not content:
                                        continue
                                    if role in ("user", "human") and user_text is None:
                                        user_text = content
                                    elif role in ("assistant", "gpt", "bot") and assistant_text is None:
                                        assistant_text = content
                                if user_text and assistant_text:
                                    rows_yielded += 1
                                    yield ("qa", user_text, assistant_text)
                            break

    
    # If enabled, ensure external parquet files exist (optional download)
    if parquet_manage_enabled:
        parquets_dir = os.path.join(data_dir, "Parquets")
        urls_file = args.parquet_urls_file
        if urls_file is None:
            default_urls_file = os.path.join(parquets_dir, "external_trainingdata.txt")
            if os.path.exists(default_urls_file):
                urls_file = default_urls_file
        ensure_external_parquets(
            parquets_dir,
            urls_file,
            allow_download=args.parquet_allow_download,
            dry_run=args.parquet_dry_run or (not args.parquet_allow_download),
        )

    files_to_process = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith(".txt"):
                # Avoid treating helper lists in Parquets as training examples
                if is_under_folder(full_path, "Parquets"):
                    continue
                files_to_process.append(full_path)
            elif parquet_ingest_enabled and file.endswith(".parquet"):
                # Only process parquet files under a Parquets folder
                if is_under_folder(full_path, "Parquets"):
                    files_to_process.append(full_path)
    
    stats["files_found"] = len(files_to_process)
    print(f"Found {len(files_to_process)} files. Processing...")
    
    for file_path in tqdm(files_to_process):
        try:
            is_raw = is_under_folder(file_path, "Raw")
            is_parquet = file_path.endswith(".parquet")
            # Pre-skip logic:
            # - txt files are either raw (under Raw/) or QA jsons (elsewhere)
            # - parquet files can contain raw text and/or QA pairs, so don't skip them just
            #   because include_qa=False.
            if not is_parquet:
                if is_raw and not include_raw:
                    continue
                if (not is_raw) and (not include_qa):
                    continue
            else:
                if (not include_raw) and (not include_qa):
                    continue

            stats["files_processed"] += 1
            if file_path.endswith(".parquet"):
                stats["parquet_files"] += 1
            else:
                stats["txt_files"] += 1

            file_examples = 0
            def file_append_example(flat_input, target_embedding):
                nonlocal file_examples
                append_example(flat_input, target_embedding)
                file_examples += 1

            def file_should_stop():
                if should_stop():
                    maybe_stop("max_examples_total reached")
                    return True
                if args.max_examples_per_file and file_examples >= args.max_examples_per_file:
                    return True
                return False

            def file_progress(prefix):
                if file_examples and (file_examples % 50000 == 0):
                    print(f"{prefix}: built {_format_int(file_examples)} examples so far...")

            if file_path.endswith(".parquet"):
                # Parquet examples are treated like either raw text or QA pairs.
                max_rows = args.parquet_max_rows
                if max_rows == 0:
                    max_rows = None

                for ex in iter_parquet_examples(file_path, max_rows):
                    if not ex:
                        continue
                    if ex[0] == "raw":
                        if not include_raw:
                            continue
                        text_content = ex[1]
                        filetokens = tokenize(text_content)
                        if not filetokens:
                            continue

                        stats["raw_docs"] += 1
                        stats["raw_tokens"] += len(filetokens)

                        rolling_mat = np.zeros((CONTEXT_SIZE, EMBEDDING_DIM + 1), dtype=np.float32)
                        current_pos = 1
                        rolling_mat[-1, :-1] = filetokens[0].embedding
                        rolling_mat[-1, -1] = current_pos

                        for i in range(1, len(filetokens)):
                            if file_should_stop():
                                break
                            target_token = filetokens[i]
                            flat_input = rolling_mat.reshape(-1).copy()
                            file_append_example(flat_input, target_token.embedding)
                            file_progress(os.path.basename(file_path))

                            current_pos += 1
                            rolling_mat[:-1] = rolling_mat[1:]
                            rolling_mat[-1, :-1] = target_token.embedding
                            rolling_mat[-1, -1] = current_pos

                        if file_should_stop():
                            # stop reading more rows from this parquet if per-file cap hit
                            if args.max_examples_per_file and file_examples >= args.max_examples_per_file:
                                break
                            if should_stop():
                                break

                    elif ex[0] == "qa":
                        if not include_qa:
                            continue
                        question = ex[1]
                        answer = ex[2]

                        q_tokens = tokenize(question)
                        a_tokens = tokenize(answer)

                        stats["qa_pairs"] += 1
                        stats["qa_question_tokens"] += len(q_tokens)
                        stats["qa_answer_tokens"] += len(a_tokens)

                        context_embeddings = []
                        current_pos = 1

                        for t in q_tokens:
                            context_embeddings.append((t.embedding, current_pos))
                            current_pos += 1

                        if len(context_embeddings) > (CONTEXT_SIZE - 1):
                            context_embeddings = context_embeddings[-(CONTEXT_SIZE - 1):]

                        context_embeddings.append((user_end_embedding, current_pos))
                        current_pos += 1

                        for t in a_tokens:
                            if file_should_stop():
                                break

                            current_mat = build_context_matrix(context_embeddings)
                            flat_input = current_mat.reshape(-1).copy()
                            file_append_example(flat_input, t.embedding)
                            file_progress(os.path.basename(file_path))

                            context_embeddings.append((t.embedding, current_pos))
                            current_pos += 1
                            if len(context_embeddings) > CONTEXT_SIZE:
                                context_embeddings.pop(0)

                        if file_should_stop():
                            if args.max_examples_per_file and file_examples >= args.max_examples_per_file:
                                break
                            if should_stop():
                                break

                        # assistant_end
                        if not file_should_stop():
                            current_mat = build_context_matrix(context_embeddings)
                            flat_input = current_mat.reshape(-1).copy()
                            file_append_example(flat_input, assistant_end_embedding)
                            file_progress(os.path.basename(file_path))

                        if file_should_stop():
                            if args.max_examples_per_file and file_examples >= args.max_examples_per_file:
                                break
                            if should_stop():
                                break

            elif is_raw:
                with open(file_path, 'r') as f:
                    text_content = f.read()
                
                filetokens = tokenize(text_content)
                if not filetokens: continue

                stats["raw_docs"] += 1
                stats["raw_tokens"] += len(filetokens)

                rolling_mat = np.zeros((CONTEXT_SIZE, EMBEDDING_DIM + 1), dtype=np.float32)
                current_pos = 1
                rolling_mat[-1, :-1] = filetokens[0].embedding
                rolling_mat[-1, -1] = current_pos

                for i in range(1, len(filetokens)):
                    if file_should_stop():
                        break
                    target_token = filetokens[i]
                    flat_input = rolling_mat.reshape(-1).copy()
                    file_append_example(flat_input, target_token.embedding)
                    file_progress(os.path.basename(file_path))

                    current_pos += 1
                    rolling_mat[:-1] = rolling_mat[1:]
                    rolling_mat[-1, :-1] = target_token.embedding
                    rolling_mat[-1, -1] = current_pos

                if file_should_stop() and should_stop():
                    break
            else:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                
                question = content.get("question", "")
                answer = content.get("answer", "")
                
                q_tokens = tokenize(question)
                a_tokens = tokenize(answer)

                stats["qa_pairs"] += 1
                stats["qa_question_tokens"] += len(q_tokens)
                stats["qa_answer_tokens"] += len(a_tokens)
                
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
                    if file_should_stop():
                        break
                    # Prepare Input
                    # Pad to CONTEXT_SIZE
                    current_mat = build_context_matrix(context_embeddings)
                    flat_input = current_mat.reshape(-1).copy()
                    file_append_example(flat_input, t.embedding)
                    file_progress(os.path.basename(file_path))
                    
                    # Update Context
                    context_embeddings.append((t.embedding, current_pos))
                    current_pos += 1
                    if len(context_embeddings) > CONTEXT_SIZE:
                        context_embeddings.pop(0)

                if file_should_stop():
                    if should_stop():
                        maybe_stop("max_examples_total reached")
                        break
                
                # After loop: predict assistant_end
                if not file_should_stop():
                    current_mat = build_context_matrix(context_embeddings)
                    flat_input = current_mat.reshape(-1).copy()
                    file_append_example(flat_input, assistant_end_embedding)
                    file_progress(os.path.basename(file_path))

                if file_should_stop() and should_stop():
                    break
            
        except Exception as e:
            stats["errors"] += 1

        if should_stop():
            maybe_stop("max_examples_total reached")
            break

    x_arr = np.array(train_x, dtype=np.float32)
    y_arr = np.array(train_y, dtype=np.float32)
    stats["examples"] = int(len(x_arr))
    stats["targets"] = int(len(y_arr))

    if return_stats:
        return x_arr, y_arr, stats
    return x_arr, y_arr

def build_dataloaders(X, Y, batch_size=BATCH_SIZE):
    if len(X) == 0:
        return None, None

    if len(X) < 2:
        # Avoid empty test set / division by zero in eval
        X_train, Y_train = X, Y
        X_test, Y_test = X, Y
    else:
        # Split Train/Test (10% test)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split_idx = int(len(X) * 0.9)
        if split_idx <= 0:
            split_idx = 1
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        if len(test_idx) == 0:
            # ensure at least one test batch
            test_idx = train_idx

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    class TensorBatcher:
        def __init__(self, X_t, Y_t, batch_size, shuffle):
            self.X_t = X_t
            self.Y_t = Y_t
            self.batch_size = int(batch_size)
            self.shuffle = bool(shuffle)

        def __len__(self):
            n = int(self.X_t.shape[0])
            return (n + self.batch_size - 1) // self.batch_size

        def __iter__(self):
            n = int(self.X_t.shape[0])
            if self.shuffle:
                idx = torch.randperm(n, device=self.X_t.device)
            else:
                idx = torch.arange(n, device=self.X_t.device)
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                yield (
                    self.X_t.index_select(0, batch_idx),
                    self.Y_t.index_select(0, batch_idx),
                )

    # Create tensors (CPU first)
    X_train_torch = torch.as_tensor(X_train, dtype=torch.float32)
    Y_train_torch = torch.as_tensor(Y_train, dtype=torch.float32)
    X_test_torch = torch.as_tensor(X_test, dtype=torch.float32)
    Y_test_torch = torch.as_tensor(Y_test, dtype=torch.float32)

    # HighVRAM mode: keep full datasets on GPU and batch there.
    if args.highvram and getattr(device, "type", None) == "cuda":
        def _gpu_mem_str():
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                return f"free={free_b / (1024**3):.1f}GiB total={total_b / (1024**3):.1f}GiB"
            except Exception:
                return "(mem info unavailable)"

        print(f"HighVRAM: attempting to keep dataset on GPU ({_gpu_mem_str()})")

        # Try float32 first, then bf16 (half the VRAM) if we OOM.
        for attempt_dtype in (torch.float32, torch.bfloat16):
            X_tr = Y_tr = X_te = Y_te = None
            try:
                t0 = time.perf_counter()
                X_tr = X_train_torch.to(device, dtype=attempt_dtype)
                Y_tr = Y_train_torch.to(device, dtype=attempt_dtype)
                X_te = X_test_torch.to(device, dtype=attempt_dtype)
                Y_te = Y_test_torch.to(device, dtype=attempt_dtype)
                dt = time.perf_counter() - t0

                bytes_total = (
                    X_tr.numel() * X_tr.element_size() +
                    Y_tr.numel() * Y_tr.element_size() +
                    X_te.numel() * X_te.element_size() +
                    Y_te.numel() * Y_te.element_size()
                )
                gib = bytes_total / (1024 ** 3)
                print(
                    f"HighVRAM: dataset on GPU as {str(attempt_dtype).replace('torch.', '')} (~{gib:.2f} GiB) in {dt:.2f}s ({_gpu_mem_str()})"
                )
                return (
                    TensorBatcher(X_tr, Y_tr, batch_size=batch_size, shuffle=True),
                    TensorBatcher(X_te, Y_te, batch_size=batch_size, shuffle=False),
                )
            except torch.cuda.OutOfMemoryError:
                # If we OOM after some tensors have already moved, ensure we drop
                # references so VRAM can be released before retrying.
                try:
                    del X_tr, Y_tr, X_te, Y_te
                except Exception:
                    pass
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"HighVRAM: OOM keeping dataset on GPU with {attempt_dtype}; trying smaller dtype...")
            except RuntimeError as e:
                # Some builds surface OOM as RuntimeError.
                if "out of memory" in str(e).lower():
                    try:
                        del X_tr, Y_tr, X_te, Y_te
                    except Exception:
                        pass
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    print(f"HighVRAM: OOM keeping dataset on GPU with {attempt_dtype}; trying smaller dtype...")
                else:
                    raise

        print("HighVRAM: could not fit dataset on GPU; falling back to CPU DataLoader.")

    # Default mode: standard CPU DataLoader
    train_dataset = TensorDataset(X_train_torch, Y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_torch, Y_test_torch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

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
    
    # Project to d_model = 384
    x = layers.Dense(384, activation="silu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Transformer Blocks (8 layers)
    for _ in range(8):
        # Attention
        # 384 dimensions. 8 heads = 48 dim per head.
        att = layers.MultiHeadAttention(num_heads=8, key_dim=48)(x, x)
        att = layers.Dropout(0.1)(att)
        x = layers.LayerNormalization(epsilon=1e-6)(x + att)
        
        # FFN
        ffn = layers.Dense(1536, activation="silu")(x) # 4 * 384
        ffn = layers.Dense(384)(ffn)
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

# High-VRAM CUDA speed knobs
if args.highvram and getattr(device, "type", None) == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        print("HighVRAM: enabled TF32 + cudnn.benchmark")
    except Exception as e:
        print(f"HighVRAM: warning (TF32/benchmark toggle failed): {e}")

def _format_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

def _device_debug_string():
    try:
        if getattr(device, "type", None) == "cuda":
            idx = device.index if device.index is not None else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            mem_gib = props.total_memory / (1024 ** 3)
            return f"cuda:{idx} ({name}, {mem_gib:.1f} GiB, cc {props.major}.{props.minor})"
        if getattr(device, "type", None) == "mps":
            return "mps (Apple Metal)"
        return "cpu"
    except Exception:
        return str(device)

def _safe_count_params(model):
    try:
        return int(model.count_params())
    except Exception:
        return None

def print_debug_summary(pretrain_stats=None, sft_stats=None):
    print("\n=== Debug Info (before training) ===")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"KERAS_BACKEND: {os.environ.get('KERAS_BACKEND')}")
    print(f"torch: {torch.__version__} | keras: {getattr(keras, '__version__', 'unknown')} | numpy: {np.__version__}")
    print(f"Device: {_device_debug_string()}")
    if torch.cuda.is_available():
        print(f"CUDA GPUs visible: {torch.cuda.device_count()}")
    print(f"Threads: torch.get_num_threads()={torch.get_num_threads()}")
    print(f"Config: BATCH_SIZE={BATCH_SIZE}, CONTEXT_SIZE={CONTEXT_SIZE}, EMBEDDING_DIM={EMBEDDING_DIM}, INPUT_DIM={INPUT_DIM}")
    print(f"Args: discriminate={args.discriminate}, highvram={args.highvram}, use_parquets={args.use_parquets}, parquet_dry_run={args.parquet_dry_run}")

    gen_params = _safe_count_params(generator)
    disc_params = _safe_count_params(discriminator)
    if gen_params is not None:
        print(f"Generator params: {_format_int(gen_params)}")
    if disc_params is not None:
        print(f"Discriminator params: {_format_int(disc_params)}")
    if gen_params is not None and disc_params is not None:
        print(f"Total params: {_format_int(gen_params + disc_params)}")

    def _print_dataset_stats(name, st):
        if not st:
            return
        total_tokens = int(st.get('raw_tokens', 0)) + int(st.get('qa_question_tokens', 0)) + int(st.get('qa_answer_tokens', 0))
        print(f"{name} data_dir: {st.get('data_dir')}")
        print(
            f"{name} files: found={_format_int(st.get('files_found'))}, processed={_format_int(st.get('files_processed'))}, "
            f"txt={_format_int(st.get('txt_files'))}, parquet={_format_int(st.get('parquet_files'))}, errors={_format_int(st.get('errors'))}"
        )
        print(
            f"{name} tokens: total={_format_int(total_tokens)}, raw={_format_int(st.get('raw_tokens'))}, "
            f"qa_q={_format_int(st.get('qa_question_tokens'))}, qa_a={_format_int(st.get('qa_answer_tokens'))}"
        )
        print(
            f"{name} targets/examples: {_format_int(st.get('target_tokens'))} (raw_docs={_format_int(st.get('raw_docs'))}, qa_pairs={_format_int(st.get('qa_pairs'))})"
        )
        if st.get('stopped_early'):
            print(f"{name} stopped_early: True ({st.get('stop_reason')})")

    _print_dataset_stats("Pretrain", pretrain_stats)
    _print_dataset_stats("SFT", sft_stats)


def _cuda_vram_usage_str():
    """Return VRAM usage as '<used>/<total>GiB' for the active CUDA device, else 'N/A'.

    Notes:
    - Uses CUDA free/total from `torch.cuda.mem_get_info()`, so "used" reflects
      global device usage (this process + other processes), matching what you
      typically see in `nvidia-smi`.
    """
    try:
        if not torch.cuda.is_available() or getattr(device, "type", None) != "cuda":
            return "N/A"
        idx = device.index if device.index is not None else torch.cuda.current_device()
        try:
            free_b, total_b = torch.cuda.mem_get_info(idx)
        except TypeError:
            # Older torch builds may not accept a device arg.
            free_b, total_b = torch.cuda.mem_get_info()
        used_gib = (float(total_b) - float(free_b)) / (1024 ** 3)
        total_gib = float(total_b) / (1024 ** 3)
        return f"{used_gib:.1f}/{total_gib:.1f}GiB"
    except Exception:
        return "N/A"

# Optimizers (Use PyTorch optimizers for custom loop)
# Added weight decay for regularization
gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.0003, weight_decay=0.01)
disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.0003, weight_decay=0.01)

# Learning Rate Scheduler
# Some torch versions don't support the `verbose` kwarg.
try:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
except TypeError:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer, mode='min', factor=0.5, patience=2
    )

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
def run_training_phase(phase_name, train_loader, test_loader, epochs):
    if train_loader is None or test_loader is None:
        print(f"{phase_name}: No data loaders available. Skipping.")
        return

    print(f"\n=== {phase_name}: {epochs} epochs ===")
    for epoch in range(epochs):
        print(f"\n{phase_name} Epoch {epoch+1}/{epochs}")
    
        # Stage 1: Supervised
        print("Stage 1: Supervised Training")
        # HighVRAM is fast enough that less-frequent prints can feel "stuck".
        # We keep per-step progress updates and show loss every step for responsiveness.
        # If you later want max throughput, lower print frequency again.
        prog_interval = 0.25 if args.highvram else 1.0
        log_every = 1
        use_amp = bool(args.highvram and getattr(device, "type", None) == "cuda")

        # We want: "- sup_loss: 0.5118 vram: 40.2/79.3GiB".
        # Keras Progbar's `values` expects numerics (it averages them). To display
        # formatted strings like "40.2/79.3GiB", we set Progbar's internal `_values`
        # directly and call `update(..., values=[])` so it prints them verbatim.
        progbar = keras.utils.Progbar(len(train_loader), interval=prog_interval)
        sup_loss_sum = 0.0
        sup_loss_steps = 0
        for step, (x_batch, y_batch) in enumerate(train_loader, start=1):
            if not args.highvram:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

            # Forward (bf16 autocast on CUDA for speed)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                y_pred = generator(x_batch)
            # Ensure y_pred is on the correct device (fix for potential device mismatch with new layers)
            if hasattr(y_pred, 'device') and y_pred.device != device:
                y_pred = y_pred.to(device)

            # Compute loss in float32 for stability
            loss = cosine_similarity_loss(y_batch.float(), y_pred.float())
            
            # Backward
            gen_optimizer.zero_grad()
            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            gen_optimizer.step()

            sup_loss_sum += float(loss.detach().cpu())
            sup_loss_steps += 1
            sup_loss_avg = sup_loss_sum / max(1, sup_loss_steps)
            vram_str = _cuda_vram_usage_str()

            if "sup_loss" not in progbar._values_order:
                progbar._values_order.append("sup_loss")
            # Embed VRAM into the loss field so the bar looks like:
            # "- sup_loss: 0.5118 vram: 40.2/79.3GiB" (no extra dash).
            progbar._values["sup_loss"] = f"{sup_loss_avg:.4f} vram: {vram_str}"
            progbar.update(step, values=[])
    
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Stage 2/3: Discriminator + Adversarial
        if args.discriminate:
            print("Stage 2: Discriminator Training")
            progbar = keras.utils.Progbar(len(train_loader), interval=prog_interval)
            disc_loss_sum = 0.0
            disc_loss_steps = 0
            for step, (x_batch, y_batch) in enumerate(train_loader, start=1):
                if not args.highvram:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                
                # Generate fake data
                # We don't want gradients for generator here
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        y_fake = generator(x_batch)
                
                # Concatenate context + y
                real_input = torch.cat([x_batch, y_batch], dim=1)
                fake_input = torch.cat([x_batch, y_fake], dim=1)
                
                # Labels
                real_labels = torch.ones((x_batch.shape[0], 1), device=device)
                fake_labels = torch.zeros((x_batch.shape[0], 1), device=device)
                
                # Forward
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    real_preds = discriminator(real_input)
                    fake_preds = discriminator(fake_input)
                
                loss_real = bce_loss(real_labels, real_preds)
                loss_fake = bce_loss(fake_labels, fake_preds)
                total_loss = (loss_real + loss_fake) / 2
                
                # Backward
                disc_optimizer.zero_grad()
                total_loss.backward()
                disc_optimizer.step()

                disc_loss_sum += float(total_loss.detach().cpu())
                disc_loss_steps += 1
                disc_loss_avg = disc_loss_sum / max(1, disc_loss_steps)
                vram_str = _cuda_vram_usage_str()

                if "disc_loss" not in progbar._values_order:
                    progbar._values_order.append("disc_loss")
                progbar._values["disc_loss"] = f"{disc_loss_avg:.4f} vram: {vram_str}"
                progbar.update(step, values=[])
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            # Stage 3: Adversarial
            print("Stage 3: Adversarial Training")
            progbar = keras.utils.Progbar(len(train_loader), interval=prog_interval)
            adv_loss_sum = 0.0
            adv_loss_steps = 0
            for step, (x_batch, y_batch) in enumerate(train_loader, start=1):
                if not args.highvram:
                    x_batch = x_batch.to(device)
                # y_batch not needed for adversarial loss, but we need x_batch
                
                # Forward Generator
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    y_fake = generator(x_batch)
                
                # Forward Discriminator
                fake_input = torch.cat([x_batch, y_fake], dim=1)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    fake_preds = discriminator(fake_input)
                
                # We want fake_preds to be 1
                loss = bce_loss(torch.ones((x_batch.shape[0], 1), device=device), fake_preds)
                
                # Backward
                gen_optimizer.zero_grad()
                discriminator.zero_grad() # Clear disc grads just in case
                loss.backward()
                gen_optimizer.step()

                adv_loss_sum += float(loss.detach().cpu())
                adv_loss_steps += 1
                adv_loss_avg = adv_loss_sum / max(1, adv_loss_steps)
                vram_str = _cuda_vram_usage_str()

                if "adv_loss" not in progbar._values_order:
                    progbar._values_order.append("adv_loss")
                progbar._values["adv_loss"] = f"{adv_loss_avg:.4f} vram: {vram_str}"
                progbar.update(step, values=[])
            
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
                if not args.highvram:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    y_pred = generator(x_batch)
                loss = cosine_similarity_loss(y_batch.float(), y_pred.float())
                test_loss += loss.item()
                steps += 1

        if steps > 0:
            avg_test_loss = test_loss / steps
            print(f"Test Loss (Cosine Similarity): {avg_test_loss:.4f}")
            scheduler.step(avg_test_loss)
        else:
            print("Test Loss: N/A (no test batches)")


print("Generating training data...")

pretrain_train_loader = None
pretrain_test_loader = None
pretrain_stats = None

if not args.no_pretrain:
    print(f"Loading pretrain data from: {args.pretrain_dir}")
    X_pre, Y_pre, pretrain_stats = load_training_data(args.pretrain_dir, include_raw=True, include_qa=False, return_stats=True)
    print(f"Pretrain examples: {len(X_pre)}")
    pretrain_train_loader, pretrain_test_loader = build_dataloaders(X_pre, Y_pre)

print(f"Loading SFT data from: {args.sft_dir}")
sft_stats = None
X_sft, Y_sft, sft_stats = load_training_data(args.sft_dir, include_raw=True, include_qa=True, return_stats=True)
print(f"SFT examples: {len(X_sft)}")
sft_train_loader, sft_test_loader = build_dataloaders(X_sft, Y_sft)

if (pretrain_train_loader is None or pretrain_test_loader is None) and (sft_train_loader is None or sft_test_loader is None):
    print("No training data found in either pretrain or SFT folders. Exiting.")
    exit(1)

print_debug_summary(pretrain_stats=pretrain_stats, sft_stats=sft_stats)

print("Starting training...")

if not args.no_pretrain and pretrain_train_loader is not None and pretrain_test_loader is not None and args.pretrain_epochs > 0:
    run_training_phase("Pretrain", pretrain_train_loader, pretrain_test_loader, args.pretrain_epochs)

if sft_train_loader is not None and sft_test_loader is not None and args.sft_epochs > 0:
    run_training_phase("SFT", sft_train_loader, sft_test_loader, args.sft_epochs)
