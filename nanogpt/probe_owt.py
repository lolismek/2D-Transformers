"""Quick streaming probe to find an OpenWebText source that works with datasets 5.x.
Streaming=True fetches only the first example, so this is fast and downloads almost nothing."""
from datasets import load_dataset

candidates = ["Skylion007/openwebtext", "openwebtext", "stas/openwebtext-10k"]
for name in candidates:
    try:
        ds = load_dataset(name, split="train", streaming=True)
        ex = next(iter(ds))
        t = ex.get("text", "")
        print(f"OK   {name} | keys={list(ex.keys())} | len(text)={len(t)} | {t[:80]!r}")
    except Exception as e:
        print(f"FAIL {name} -> {type(e).__name__}: {str(e)[:240]}")
