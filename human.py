"""Thin CLI entrypoint:  python human.py train | demo | validate   (from the human/ directory)

  train     run the training loop (config in config.py)
  demo      render a moving human from a checkpoint
  validate  check the from-scratch SMPL LBS against smplx (needs a model file)
"""
from __future__ import annotations
import argparse
from config import Config


def main():
    ap = argparse.ArgumentParser(prog="human")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    d = sub.add_parser("demo")
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--out", default="demo.mp4")
    d.add_argument("--orbit", action="store_true")
    d.add_argument("--fps", type=int, default=24)
    sub.add_parser("validate")
    args = ap.parse_args()
    cfg = Config()

    if args.cmd == "train":
        from train import train
        train(cfg)
    elif args.cmd == "demo":
        from demo import run
        run(cfg, args.checkpoint, args.out, args.orbit, args.fps)
    elif args.cmd == "validate":
        from smpl_model import validate_against_smplx
        validate_against_smplx(cfg.smpl_model_path, cfg.smpl_gender, cfg.device)


if __name__ == "__main__":
    main()
