"""Analytic FLOP/compute model for the gated depth-reader sweep (no torch, no GPU).

Computes, for the nanochat backbone H (depth d, n_embd = 64*d rounded to head_dim 128) and the
full-width WideReader V (reader_layers), the corrected per-token FLOPs, the compute-optimal token
budget (data:param ratio 12), and the TOTAL training compute. Then matches each reader cell
(H=h, V=v) to the plain baseline depth h' whose compute-optimal run costs the SAME total FLOPs --
the per-FLOP "equal-compute baseline" each reader must beat.

Importable: the helper functions (n_embd_int, scaling_base, reader_matrix, base_point, reader_point,
matched_depth, ...) are side-effect-free at module level, so scripts/modal_train.py can import them
to size the H+V token budget. Running the file as a script (`python3 scripts/reader_flops.py`)
self-validates against three ground-truth anchors and prints the run set:
  * d10 baseline fwd+bwd FLOPs/token = 5.780e8   (check_wide_reader.py E_BASE, phase_a.log)
  * d10 WideReader L2 matrix params  = 9,830,400  (check_wide_reader.py)
  * d10 WideReader L2 reader FLOPs   = 648.8e6 -> total 2.122x  (README / nanochat_10p2_reader.md)
"""
import math

# --- constants (from gpt.py GPTConfig + base_train.py defaults; reader runs use --window-pattern=L) ---
ASPECT   = 64        # model_dim = depth * aspect_ratio
HEAD_DIM = 128       # --head-dim
T        = 2048      # sequence_len / --max-seq-len  (full context, window=L)
VOCAB    = 32768     # padded vocab (32768 already a multiple of 64)
RATIO    = 12        # --target-param-data-ratio (compute-optimal-ish; Chinchilla=20)
MLP_MULT = 4         # backbone & reader MLP hidden multiplier


def n_embd_int(d):
    """Code-exact width for an integer depth: round d*ASPECT up to a multiple of HEAD_DIM."""
    base = d * ASPECT
    return ((base + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM


# A transformer block (attn c_q/c_k/c_v/c_proj = 4*e^2 with n_kv_head=n_head; MLP c_fc+c_proj = 8*e^2)
# is 12*e^2 matrix params. A WideReader VBlock is the same shape -> also 12*e^2.
def block_matrix(e):            return 12 * e * e
def backbone_matrix(d, e):      return d * block_matrix(e)                 # transformer.h
def lm_head(e):                 return e * VOCAB
def scaling_base(d, e):         return backbone_matrix(d, e) + lm_head(e)  # = num_scaling_params (code)
def attn_flops(d, e):           return d * 12 * e * T                       # 12*h*q*T per layer, full ctx

def backbone_fpt(d, e):         return 6 * scaling_base(d, e) + attn_flops(d, e)   # fwd+bwd / token

def reader_matrix(d, e, V):     return V * block_matrix(e)                  # V VBlocks at native width e
def reader_fpt(d, e, V):        return 6 * reader_matrix(d, e, V) * (d + 1) # each matrix hit on all d+1 rungs


# --- compute-optimal totals -------------------------------------------------------------------------
def base_point(d):
    e = n_embd_int(d)
    sp = scaling_base(d, e)
    fpt = backbone_fpt(d, e)
    toks = RATIO * sp
    return dict(d=d, e=e, params=sp, fpt=fpt, tokens=toks, flops=fpt * toks)

def reader_point(d, V):
    e = n_embd_int(d)
    sp_h = scaling_base(d, e)
    rm = reader_matrix(d, e, V)
    sp_hv = sp_h + rm                       # budget on H+V (the reader is a bigger model)
    fpt = backbone_fpt(d, e) + reader_fpt(d, e, V)
    toks = RATIO * sp_hv
    return dict(d=d, V=V, e=e, params=sp_hv, rm=rm, fpt=fpt, mult=fpt / backbone_fpt(d, e),
                tokens=toks, flops=fpt * toks)


def reader_ratio(d, V):
    """The --target-param-data-ratio that gives a reader cell its H+V compute-optimal token budget.
    base_train sizes tokens = ratio * num_scaling_params, and num_scaling_params == scaling_base(H)
    (reader excluded), so ratio = RATIO * (scaling_base + reader_matrix) / scaling_base yields
    tokens = RATIO * (H + V) params. Used by the Modal launcher."""
    e = n_embd_int(d)
    return RATIO * (scaling_base(d, e) + reader_matrix(d, e, V)) / scaling_base(d, e)


# --- baseline frontier as a CONTINUOUS function of depth, for inverting C_base(h') = C_reader -------
def base_flops_continuous(x):
    """Total compute-optimal baseline FLOPs at (real) depth x, with e = ASPECT*x (smooth)."""
    e = ASPECT * x
    sp = 12 * x * e * e + e * VOCAB
    fpt = 6 * sp + 12 * x * e * T
    return fpt * (RATIO * sp)

def matched_depth(target_flops):
    """Smallest real depth h' whose compute-optimal baseline costs target_flops (binary search)."""
    lo, hi = 2.0, 200.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if base_flops_continuous(mid) < target_flops:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fmt(x):
    for unit, div in (("e21", 1e21), ("e18", 1e18), ("e15", 1e15), ("e12", 1e12), ("e9", 1e9), ("e6", 1e6)):
        if abs(x) >= div:
            return f"{x/div:6.2f}{unit}"
    return f"{x:8.1f}"


if __name__ == "__main__":
    # ================================================================================================
    # 1) VALIDATION ("the tests") -- must reproduce the repo's known-correct d10 numbers
    # ================================================================================================
    e10 = n_embd_int(10)
    assert e10 == 640, e10
    assert reader_matrix(10, e10, 2) == 9_830_400, reader_matrix(10, e10, 2)
    assert math.isclose(backbone_fpt(10, e10), 5.780293e8, rel_tol=2e-3), backbone_fpt(10, e10)
    assert math.isclose(reader_fpt(10, e10, 2), 648.8e6, rel_tol=2e-3), reader_fpt(10, e10, 2)
    rp10 = reader_point(10, 2)
    assert math.isclose(rp10["mult"], 2.122, rel_tol=3e-3), rp10["mult"]
    assert math.isclose(scaling_base(10, e10), 70.1e6, rel_tol=5e-3), scaling_base(10, e10)
    print("anchors OK: d10 fpt=5.78e8, readerL2 params=9,830,400, reader fpt=648.8e6 (2.122x), scaling=70.1M\n")

    # ================================================================================================
    # 2) BASELINE FRONTIER (compute-optimal, ratio 12)
    # ================================================================================================
    print("=" * 78)
    print("BASELINE FRONTIER  (reader=none, ratio-12 compute-optimal)")
    print("=" * 78)
    print(f"{'depth':>5} {'n_embd':>6} {'heads':>5} {'params':>9} {'FLOPs/tok':>10} {'tokens':>9} {'total FLOPs':>11}")
    for d in [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40]:
        p = base_point(d)
        print(f"{d:>5} {p['e']:>6} {p['e']//HEAD_DIM:>5} {fmt(p['params']):>9} {fmt(p['fpt']):>10} "
              f"{fmt(p['tokens']):>9} {fmt(p['flops']):>11}")

    # ================================================================================================
    # 3) READER CELLS  (the d10 V-slope + the deferred H-sweep)  -> matched equal-compute baseline h'
    # ================================================================================================
    HS = [10, 14, 18, 22]
    VS = [2, 4, 8, 10]
    print("\n" + "=" * 78)
    print("READER CELLS  (WideReader, gated, ratio-12 on H+V)  ->  equal-compute baseline h'")
    print("=" * 78)
    print(f"{'H':>3} {'V':>3} {'n_embd':>6} {'fpt x base':>10} {'ratio':>6} {'tokens':>9} {'total FLOPs':>11} "
          f"{'match hp':>8}")
    cells = []
    for h in HS:
        for V in VS:
            r = reader_point(h, V)
            hp = matched_depth(r["flops"])
            cells.append((h, V, r, hp))
            print(f"{h:>3} {V:>3} {r['e']:>6} {r['mult']:>9.2f}x {reader_ratio(h, V):>6.1f} {fmt(r['tokens']):>9} "
                  f"{fmt(r['flops']):>11} {hp:>7.1f}")

    # ================================================================================================
    # 4) MATCHING MATRIX  (which even-depth baseline to actually train per reader cell)
    # ================================================================================================
    def nearest_even(x):  return int(round(x / 2.0) * 2)
    print("\n" + "=" * 78)
    print("MATRIX: reader (H,V)  ->  nearest even baseline depth h' to train as its equal-compute control")
    print("=" * 78)
    header = "  H\\V " + "".join(f"{('L'+str(V)):>8}" for V in VS)
    print(header)
    needed = set()
    for h in HS:
        row = f"{h:>4}  "
        for V in VS:
            r = reader_point(h, V)
            hp = nearest_even(matched_depth(r["flops"]))
            needed.add(hp)
            row += f"{('d'+str(hp)):>8}"
        print(row)
    print(f"\nbaseline depths to train (cover all h'): {sorted(needed)}")
    print(f"reader backbone depths:                  {sorted(HS)}")
