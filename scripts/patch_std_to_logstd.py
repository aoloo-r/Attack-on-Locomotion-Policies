#!/usr/bin/env python
"""Convert a scalar-std rsl_rl checkpoint to log-std in place (new file).

The Gaussian actor can be parameterized two ways:
  - scalar std : actor param ``distribution.std_param`` holds std directly. Gradient
                 descent can push an element below 0, and torch.normal then crashes
                 with "normal expects all elements of std >= 0.0".
  - log std    : actor param ``distribution.log_std_param`` holds log(std); std is
                 recovered as exp(log_std), positive by construction -> no crash.

A checkpoint trained with scalar std must be converted before it can seed a run
configured with ``std_type="log"``; otherwise the log-std parameter is never
loaded (key mismatch) and the fix silently does nothing. The conversion is exact:
  log_std_param = log(std_param)
so the learned per-joint exploration schedule transfers unchanged.

Usage:
    python scripts/patch_std_to_logstd.py <in.pt> <out.pt>
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="scalar-std checkpoint (read-only)")
    ap.add_argument("dst", help="output log-std checkpoint")
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    actor = ckpt["actor_state_dict"]

    if "distribution.log_std_param" in actor:
        raise SystemExit(f"{args.src} is already log-std (has distribution.log_std_param)")
    if "distribution.std_param" not in actor:
        raise SystemExit(f"{args.src} has no distribution.std_param to convert; keys={list(actor)}")

    std = actor.pop("distribution.std_param")
    if torch.any(std <= 0):
        raise SystemExit(f"std has non-positive elements, cannot take log: {std}")
    log_std = torch.log(std)
    actor["distribution.log_std_param"] = log_std

    print(f"converted {std.numel()} std elements -> log_std")
    print(f"  std     [:5] = {std.flatten()[:5].tolist()}")
    print(f"  log_std [:5] = {log_std.flatten()[:5].tolist()}")
    print(f"  std range = [{std.min():.4f}, {std.max():.4f}]")

    torch.save(ckpt, args.dst)
    print(f"wrote {args.dst}  (iter={ckpt.get('iter')})")


if __name__ == "__main__":
    main()
