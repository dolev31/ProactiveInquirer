"""Relative weight distance between LoRA adapters: did a resumed DPO run keep its learning?

rung2-8b-headline-notdone-both (seed 0) resumed from its own checkpoint-1800 and saved a final whose
weights differ byte-wise from its SFT init, yet on the dev ranking pairs it behaves like the init
(margin corr 0.9997 with the init; seed 1, trained without a resume, reads 0.925). A byte hash cannot
tell "trained, then reset to init by the resume" from "trained normally". Distances can:

  if the resume RESET the policy to the init and then ran the remaining ~168 steps,
      d(final, init) is small, d(ckpt-1800, init) is large, d(final, ckpt-1800) is large;
  if the run trained normally throughout,
      d(final, ckpt-1800) is small and both are far from the init.

Reports ||W - W_ref|| / ||W_ref|| over all adapter tensors, on CPU.
"""

from __future__ import annotations

import sys

import torch
from safetensors.torch import load_file


def rel_dist(a: dict, b: dict) -> float:
    keys = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise SystemExit(f"REFUSING: tensor names differ ({len(set(a) ^ set(b))} names)")
    num = sum(float(torch.sum((a[k].float() - b[k].float()) ** 2)) for k in keys)
    den = sum(float(torch.sum(b[k].float() ** 2)) for k in keys)
    return (num / den) ** 0.5


def main() -> int:
    args = sys.argv[1:]
    # At least two adapters, every one written LABEL=path. (An earlier guard here rejected any ODD
    # number of inputs, a condition unrelated to what it claimed to check.)
    if len(args) < 2 or any("=" not in a for a in args):
        raise SystemExit("usage: adapter_distance.py LABEL=path LABEL=path [LABEL=path ...]")
    named = dict(a.split("=", 1) for a in args)
    w = {k: load_file(v, device="cpu") for k, v in named.items()}
    labels = list(named)
    print("relative distance ||W_row - W_col|| / ||W_col||")
    for r in labels:
        print(
            "  "
            + r.ljust(22)
            + "  ".join(f"{c[:14]:>14}={rel_dist(w[r], w[c]):.4f}" for c in labels if c != r)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
