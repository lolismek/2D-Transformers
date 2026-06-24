"""Modal launcher for WideReader runs on N x A100-40GB (serverless).

Reproduces the tigerfish setup EXACTLY so val bpb overlays the existing baseline@1605=0.877 and
wide@1605=0.898 -- only the swept knob differs (--h-residual or --reader-layers):
  * same data:      nanochat.dataset -n 24  => train shards 0..23 + val shard 6542  (hash-verified)
  * same tokenizer: tigerfish's tokenizer.pkl + token_bytes.pt in the Volume        (hash-verified)
  * same numerics:  A100(40GB), bf16, --window-pattern=L (no FA3), DBS=16, total-batch auto (524288),
                    seed/schedule unchanged. total-batch-size is keyed on depth (reader excluded from
                    the scaling count), so world_size 2 vs 4 give the SAME effective batch => DDP at
                    nproc=4 is a pure ~2x wall-clock speedup, directly comparable to the 2-GPU runs.
  * same code:      this repo (mounted), incl. nanochat/readers/wide.py + nanochat/h_variants.py

GPU count is set at launch via the MODAL_GPUS env var (decorator is evaluated locally by `modal run`):
    MODAL_GPUS=4 modal run scripts/modal_train.py --action train --steps 1605 \
        --reader-layers 4 --h-residual full --tag d10_wide640_L4_full

Run order (data is seeded once into the persistent Volume, then reused by every run):
    modal run scripts/modal_train.py --action seed
    # residual-free WideReader (the original use; 2x A100):
    modal run scripts/modal_train.py --action train --steps 1605 --h-residual none --tag d10_wide640_nores
    # taller V (this experiment; 4 layers, residuals normal, 4x A100):
    MODAL_GPUS=4 modal run scripts/modal_train.py --action train --steps 20   --reader-layers 4 --tag d10_wide640_L4_pilot
    MODAL_GPUS=4 modal run scripts/modal_train.py --action train --steps 1605 --reader-layers 4 --tag d10_wide640_L4_full

The image replicates tigerfish: uv sync --extra gpu (torch 2.9.1+cu128, rustbpe from PyPI). The repo
is baked in at build time so `uv sync` runs; data+tokenizer+checkpoints live on a persistent Volume.
"""
import os
import subprocess
import hashlib
import modal

REPO = "/root/repo"
CACHE = "/cache"
VENV_PY = f"{REPO}/.venv/bin/python"
VENV_TORCHRUN = f"{REPO}/.venv/bin/torchrun"
LOCAL_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the 2d-Transformers root

# GPU count, set at launch: `MODAL_GPUS=4 modal run ...`. Read locally when `modal run` imports this
# file, so the @app.function(gpu=...) spec registered with Modal reflects it. nproc is passed into the
# remote call (below) from the same value, so torchrun --nproc_per_node always matches the provisioning.
GPUS = int(os.environ.get("MODAL_GPUS", "2"))

# tigerfish hashes -- the comparability guarantee (data source + tokenizer must be byte-identical)
SHA1 = {
    "tokenizer/tokenizer.pkl":             "cec71de967502b56af6063d66e0b498d6684f032",
    "tokenizer/token_bytes.pt":            "e158409833c2ce4771ffa2f13f658089b6c67f5d",
    "base_data_climbmix/shard_00000.parquet": "fc286717edb2c859e1baa5ef0714edf4271941dd",
}

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "build-essential", "pkg-config", "libssl-dev")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",  # rust (rustbpe sdist build)
        "curl -LsSf https://astral.sh/uv/install.sh | sh",                          # uv (nanochat package manager)
    )
    .add_local_dir(
        LOCAL_REPO, REPO, copy=True,
        ignore=["**/.git", "**/__pycache__", "**/*.pyc", "**/.venv", "**/runs", "**/experiments/figs"],
    )
    # build the venv exactly like tigerfish: cu128 torch + rustbpe. cargo on PATH for the rustbpe sdist build.
    .run_commands(
        f"cd {REPO} && PATH=/root/.cargo/bin:/root/.local/bin:$PATH /root/.local/bin/uv sync --extra gpu"
    )
)

vol = modal.Volume.from_name("nanochat-cache", create_if_missing=True)
app = modal.App("nanochat-wide", image=image)


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(label):
    ok = True
    for rel, want in SHA1.items():
        p = os.path.join(CACHE, rel)
        if not os.path.exists(p):
            print(f"  [{label}] MISSING  {rel}"); ok = False; continue
        got = _sha1(p)
        match = got == want
        ok = ok and match
        print(f"  [{label}] {'OK      ' if match else 'MISMATCH'} {rel}  got={got[:12]} want={want[:12]}")
    return ok


@app.function(volumes={CACHE: vol}, timeout=3600)
def seed():
    """Download the exact 24 train + 1 val (6542) shards into the Volume, then hash-verify."""
    os.environ["NANOCHAT_BASE_DIR"] = CACHE
    subprocess.run([VENV_PY, "-m", "nanochat.dataset", "-n", "24"], cwd=REPO, check=True,
                   env=dict(os.environ, NANOCHAT_BASE_DIR=CACHE))
    vol.commit()
    dd = os.path.join(CACHE, "base_data_climbmix")
    print("shards present:", sorted(os.listdir(dd)) if os.path.isdir(dd) else "MISSING")
    print("=== hash check (data now; tokenizer expected after `modal volume put`) ===")
    _verify("seed")


@app.function(gpu=f"A100:{GPUS}", volumes={CACHE: vol}, timeout=6 * 3600)
def train(steps: int, tag: str, dbs: int = 16, h_residual: str = "full",
          reader_layers: int = 2, nproc: int = 0):
    """WideReader training on N x A100-40GB. Refuses to run if the data/tokenizer hashes don't match."""
    if not nproc:
        nproc = GPUS
    if not _verify("pre-train"):
        raise RuntimeError("tokenizer/data hash mismatch -- refusing to train (result would not be comparable)")
    common = ("--depth=10 --window-pattern=L --eval-every=100 --eval-tokens=2097152 "
              "--core-metric-every=-1 --sample-every=-1")
    wide = (f"--reader=wide --reader-dim=640 --h-residual={h_residual} --reader-layers={reader_layers} "
            f"--num-iterations={steps} --device-batch-size={dbs}")
    cmd = (f"{VENV_TORCHRUN} --standalone --nproc_per_node={nproc} -m scripts.base_train -- "
           f"{common} {wide} --model-tag={tag}")
    env = dict(os.environ, NANOCHAT_BASE_DIR=CACHE, OMP_NUM_THREADS="1")
    print(f"=== RUN ({steps} steps, tag={tag}, nproc={nproc}, dbs={dbs}, "
          f"h_residual={h_residual}, reader_layers={reader_layers}):\n{cmd}\n", flush=True)
    log_path = os.path.join(CACHE, f"{tag}.log")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, shell=True, cwd=REPO, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        rc = proc.wait()
    vol.commit()
    print(f"=== DONE rc={rc}  (log persisted to volume:{tag}.log)", flush=True)
    if rc != 0:
        raise RuntimeError(f"training exited with code {rc}")
    return tag


@app.local_entrypoint()
def main(action: str = "train", steps: int = 1605, tag: str = "d10_wide640_L4_full",
         dbs: int = 16, h_residual: str = "full", reader_layers: int = 2):
    if action == "seed":
        seed.remote()
    elif action == "train":
        train.remote(steps=steps, tag=tag, dbs=dbs, h_residual=h_residual,
                     reader_layers=reader_layers, nproc=GPUS)
    else:
        raise SystemExit(f"unknown action: {action!r} (use 'seed' or 'train')")
