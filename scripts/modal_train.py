"""Modal launcher for the residual-free WideReader run (and pilots) on 2x A100-40GB.

Reproduces the tigerfish setup EXACTLY so val bpb overlays the existing baseline@1605=0.877 and
wide@1605=0.898 -- only --h-residual=none differs:
  * same data:      nanochat.dataset -n 24  => train shards 0..23 + val shard 6542  (hash-verified)
  * same tokenizer: tigerfish's tokenizer.pkl + token_bytes.pt in the Volume        (hash-verified)
  * same numerics:  A100(40GB) x2, bf16, --window-pattern=L (no FA3), DBS=16, seed/schedule unchanged
  * same code:      this repo (mounted), incl. the residual-free edits + nanochat/h_variants.py

The image replicates tigerfish: uv sync --extra gpu (torch 2.9.1+cu128, rustbpe from PyPI). The repo
is baked in at build time so `uv sync` runs; data+tokenizer+checkpoints live on a persistent Volume.

Run order (see also the prep steps the assistant runs around it -- tokenizer upload + volume create):
    modal run scripts/modal_train.py --action seed
    modal run scripts/modal_train.py --action train --steps 20   --tag d10_wide_nores_pilot
    modal run scripts/modal_train.py --action train --steps 1605 --tag d10_wide640_nores
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
app = modal.App("nanochat-resfree", image=image)


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


@app.function(gpu="A100:2", volumes={CACHE: vol}, timeout=4 * 3600)
def train(steps: int, tag: str, dbs: int = 16):
    """Residual-free WideReader training on 2x A100-40GB. Refuses to run if hashes don't match."""
    if not _verify("pre-train"):
        raise RuntimeError("tokenizer/data hash mismatch -- refusing to train (result would not be comparable)")
    common = ("--depth=10 --window-pattern=L --eval-every=100 --eval-tokens=2097152 "
              "--core-metric-every=-1 --sample-every=-1")
    wide = (f"--reader=wide --reader-dim=640 --h-residual=none "
            f"--num-iterations={steps} --device-batch-size={dbs}")
    cmd = (f"{VENV_TORCHRUN} --standalone --nproc_per_node=2 -m scripts.base_train -- "
           f"{common} {wide} --model-tag={tag}")
    env = dict(os.environ, NANOCHAT_BASE_DIR=CACHE, OMP_NUM_THREADS="1")
    print(f"=== RUN ({steps} steps, tag={tag}):\n{cmd}\n", flush=True)
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
def main(action: str = "train", steps: int = 1605, tag: str = "d10_wide640_nores", dbs: int = 16):
    if action == "seed":
        seed.remote()
    elif action == "train":
        train.remote(steps=steps, tag=tag, dbs=dbs)
    else:
        raise SystemExit(f"unknown action: {action!r} (use 'seed' or 'train')")
