"""
Stream OpenWebText and tokenize to train.bin / val.bin WITHOUT caching the full ~70GB raw
dataset. The original prepare.py calls load_dataset(...) non-streaming, which downloads and
caches all 70GB of raw parquet -> blows the kingcrab home disk quota. Streaming fetches shards
on the fly and we write only the (few-GB) token bins.

Usage:
    python prepare_stream.py [TRAIN_TOKENS] [VAL_TOKENS]
    # e.g. python prepare_stream.py 2e9 1e7      (2B train tokens, 10M val tokens)

val is taken from the head of the stream and train from the following (disjoint) docs.
"""
import os, sys, time
import numpy as np
import tiktoken
from datasets import load_dataset

TRAIN_TOKENS = int(float(sys.argv[1])) if len(sys.argv) > 1 else 2_000_000_000
VAL_TOKENS   = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10_000_000
BATCH_DOCS = 1024
NUM_THREADS = 32
out_dir = os.path.dirname(os.path.abspath(__file__))

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token

def build(split, budget, it):
    path = os.path.join(out_dir, f"{split}.bin")
    f = open(path, "wb")
    total = 0
    batch = []
    t0 = time.time()

    def flush(texts):
        nonlocal total
        ids_list = enc.encode_ordinary_batch(texts, num_threads=NUM_THREADS)
        flat = []
        for ids in ids_list:
            flat.extend(ids)
            flat.append(eot)
        arr = np.asarray(flat, dtype=np.uint16)
        arr.tofile(f)
        total += arr.size

    for ex in it:
        batch.append(ex["text"])
        if len(batch) >= BATCH_DOCS:
            flush(batch); batch = []
            dt = max(time.time() - t0, 1e-9)
            print(f"\r{split}: {total/1e6:7.1f}M / {budget/1e6:.0f}M tokens  ({total/dt/1e6:5.2f}M tok/s)", end="", flush=True)
            if total >= budget:
                break
    if batch and total < budget:
        flush(batch)
    f.close()
    print(f"\n{split}.bin done: {total:,} tokens, {os.path.getsize(path)/1e9:.2f} GB")
    return total

print(f"streaming Skylion007/openwebtext -> val {VAL_TOKENS/1e6:.0f}M, train {TRAIN_TOKENS/1e6:.0f}M tokens", flush=True)
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
it = iter(ds)
build("val", VAL_TOKENS, it)
build("train", TRAIN_TOKENS, it)
print("done")
