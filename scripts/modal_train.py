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

GPU count is set at launch via the MODAL_GPUS env var (decorator is evaluated locally by `modal run`).
steps<=0 trains to the compute-optimal horizon; readers auto-size their H+V token budget via the
--target-param-data-ratio derived in scripts/reader_flops.py (reader=bigger model => more tokens).

Run order (data is seeded once into the persistent Volume, then reused by every run):
    modal run scripts/modal_train.py --action seed
    # pilot one gated reader cell at 20 steps (catches OOM / dbs before the full budget):
    MODAL_GPUS=4 modal run scripts/modal_train.py --action train --depth 10 --reader wide \
        --reader-gate scalar --reader-layers 4 --steps 20 --tag hv_pilot
    # the full d10 V-slope: gated readers d10x{L2,L4,L8,L10} + baselines d12,d14,d16 (~12h on 4xA100):
    MODAL_GPUS=4 modal run scripts/modal_train.py --action sweep
    # each run appends a row to experiments/figs/hv_sweep_results.csv; then analyze locally:
    python scripts/plot_hv_sweep.py

The image replicates tigerfish: uv sync --extra gpu (torch 2.9.1+cu128, rustbpe from PyPI). The repo
is baked in at build time so `uv sync` runs; data+tokenizer+checkpoints live on a persistent Volume.
"""
import os
import sys
import csv
import subprocess
import hashlib
import modal

REPO = "/root/repo"
CACHE = "/cache"
VENV_PY = f"{REPO}/.venv/bin/python"
VENV_TORCHRUN = f"{REPO}/.venv/bin/torchrun"
LOCAL_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the 2d-Transformers root
sys.path.insert(0, LOCAL_REPO)  # so the local entrypoint can import scripts.reader_flops (H+V budget)

# The H×V sweep result table, written locally by the entrypoint (experiments/figs is excluded from the
# image and ephemeral in-container, so the CSV lives on the host, where plot_hv_sweep.py reads it).
SWEEP_CSV = os.path.join(LOCAL_REPO, "experiments", "figs", "hv_sweep_results.csv")
CSV_FIELDS = ["tag", "depth", "n_embd", "reader", "reader_layers", "reader_gate",
              "num_scaling_params", "num_flops_per_token", "num_iterations", "total_batch_size",
              "total_tokens", "total_flops", "final_val_bpb", "min_val_bpb", "final_gate",
              "target_param_data_ratio"]


def _append_csv(row):
    """Append one run's results dict (returned by train.remote) as a row to SWEEP_CSV."""
    if not row:
        print("WARNING: empty result, not appending to CSV")
        return
    os.makedirs(os.path.dirname(SWEEP_CSV), exist_ok=True)
    exists = os.path.exists(SWEEP_CSV)
    with open(SWEEP_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"appended {row.get('tag')} (val_bpb={row.get('final_val_bpb')}) -> {SWEEP_CSV}", flush=True)

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
def train(steps: int, tag: str, depth: int = 10, reader: str = "wide", reader_layers: int = 2,
          reader_gate: str = "scalar", dbs: int = 16, h_residual: str = "full",
          target_param_data_ratio: float = 12.0, nproc: int = 0):
    """base_train on N x A100-40GB. Refuses to run if the data/tokenizer hashes don't match.

    steps > 0 pins --num-iterations (pilots / iso-step debugging); steps <= 0 uses the compute-optimal
    --target-param-data-ratio (readers pass their H+V ratio so they earn proportionally more tokens).
    reader='none' trains a plain baseline. Returns the per-run results dict base_train writes to the
    volume ({tag}.result.json), so the local entrypoint can append it to the sweep CSV.
    """
    import json
    if not nproc:
        nproc = GPUS
    if not _verify("pre-train"):
        raise RuntimeError("tokenizer/data hash mismatch -- refusing to train (result would not be comparable)")
    common = (f"--depth={depth} --window-pattern=L --eval-every=100 --eval-tokens=2097152 "
              "--core-metric-every=-1 --sample-every=-1")
    # Training horizon: an explicit step count overrides the ratio-based compute-optimal budget.
    horizon = (f"--num-iterations={steps}" if steps and steps > 0
               else f"--target-param-data-ratio={target_param_data_ratio}")
    if reader == "none":
        arch = f"--reader=none --device-batch-size={dbs}"
    else:
        arch = (f"--reader={reader} --reader-layers={reader_layers} --reader-gate={reader_gate} "
                f"--h-residual={h_residual} --device-batch-size={dbs}")
    cmd = (f"{VENV_TORCHRUN} --standalone --nproc_per_node={nproc} -m scripts.base_train -- "
           f"{common} {arch} {horizon} --model-tag={tag}")
    env = dict(os.environ, NANOCHAT_BASE_DIR=CACHE, OMP_NUM_THREADS="1")
    print(f"=== RUN tag={tag} depth={depth} reader={reader} layers={reader_layers} gate={reader_gate} "
          f"nproc={nproc} dbs={dbs} horizon='{horizon}':\n{cmd}\n", flush=True)
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
    # base_train writes a compact results sidecar to the volume; read it back and return it.
    result_path = os.path.join(CACHE, f"{tag}.result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    print(f"WARNING: no result sidecar at {result_path}", flush=True)
    return {"tag": tag}


@app.function(gpu="A100", volumes={CACHE: vol}, timeout=2 * 3600)
def ablate(tag: str, step: int = 0, dbs: int = 16, eval_tokens: int = 2097152):
    """Top-rung ablation probe on a trained reader checkpoint (single A100, eval only).

    Replaces every rung of the ladder with the top rung h_L and re-measures val bpb: unchanged
    => the reader collapsed to h_L (effective identity); worse => it genuinely reads the ladder.
    Eval is a no_grad single forward, so 1 GPU (even a 40GB card) is ample.
    """
    if not _verify("pre-ablate"):
        raise RuntimeError("tokenizer/data hash mismatch -- refusing (eval would not be comparable)")
    step_arg = f" --step {step}" if step else ""
    cmd = (f"{VENV_PY} -m scripts.ablate_reader --model-tag {tag}{step_arg} "
           f"--device-batch-size {dbs} --eval-tokens {eval_tokens}")
    env = dict(os.environ, NANOCHAT_BASE_DIR=CACHE, OMP_NUM_THREADS="1")
    print(f"=== ABLATE (tag={tag}, dbs={dbs}, eval_tokens={eval_tokens}):\n{cmd}\n", flush=True)
    proc = subprocess.Popen(cmd, shell=True, cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    print(f"=== DONE rc={rc}", flush=True)
    if rc != 0:
        raise RuntimeError(f"ablate exited with code {rc}")


# The d10 V-slope run set (see experiments/gated_reader_plan.md): four gated WideReader cells at d10
# (dbs drops to 8 for the two tall readers to fit 40GB) + three plain baselines that bracket them on
# the FLOPs axis. d10=0.877 is reused from tigerfish (not retrained here; pass --with-d10 to add it).
SWEEP_READERS = [  # (reader_layers, device_batch_size); L4+ drop to dbs=8 to fit 40GB A100s
    (2, 16), (4, 8), (8, 8), (10, 8),
]
SWEEP_BASELINES = [12, 14, 16]  # depth; dbs 16


@app.local_entrypoint()
def main(action: str = "train", steps: int = 0, tag: str = "hv_pilot",
         depth: int = 10, reader: str = "wide", reader_layers: int = 2, reader_gate: str = "scalar",
         dbs: int = 16, h_residual: str = "full", target_param_data_ratio: float = -1.0,
         with_d10: bool = False, step: int = 0, eval_tokens: int = 2097152):
    """steps<=0 => compute-optimal horizon (readers auto-size their H+V ratio). 'sweep' runs the d10
    V-slope (4 readers + 3 baselines) sequentially on one box, appending each result to the CSV."""
    from scripts.reader_flops import reader_ratio

    def _ratio(reader_, depth_, V_, given):
        if given is not None and given >= 0:
            return given
        return reader_ratio(depth_, V_) if reader_ != "none" else 12.0

    if action == "seed":
        seed.remote()
    elif action == "train":
        r = _ratio(reader, depth, reader_layers, target_param_data_ratio)
        row = train.remote(steps=steps, tag=tag, depth=depth, reader=reader, reader_layers=reader_layers,
                           reader_gate=reader_gate, dbs=dbs, h_residual=h_residual,
                           target_param_data_ratio=r, nproc=GPUS)
        _append_csv(row)
    elif action == "sweep":
        runs = []  # build the full plan, then run sequentially
        if with_d10:
            runs.append(dict(tag="hv_d10_base", depth=10, reader="none", reader_layers=2,
                             reader_gate="none", dbs=16, target_param_data_ratio=12.0))
        for V, rdbs in SWEEP_READERS:
            runs.append(dict(tag=f"hv_d10_L{V}_g", depth=10, reader="wide", reader_layers=V,
                             reader_gate="scalar", dbs=rdbs, target_param_data_ratio=reader_ratio(10, V)))
        for H in SWEEP_BASELINES:
            runs.append(dict(tag=f"hv_d{H}_base", depth=H, reader="none", reader_layers=2,
                             reader_gate="none", dbs=16, target_param_data_ratio=12.0))
        print(f"=== SWEEP: {len(runs)} runs (sequential on {GPUS}xA100) ===")
        for i, rc in enumerate(runs):
            print(f"--- [{i+1}/{len(runs)}] {rc['tag']} ---", flush=True)
            row = train.remote(steps=steps, h_residual=h_residual, nproc=GPUS, **rc)
            _append_csv(row)
        print(f"=== SWEEP done -> {SWEEP_CSV} ===")
    elif action == "ablate":
        ablate.remote(tag=tag, step=step, dbs=dbs, eval_tokens=eval_tokens)
    else:
        raise SystemExit(f"unknown action: {action!r} (use 'seed', 'train', 'sweep', or 'ablate')")
