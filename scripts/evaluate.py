import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.clip_evaluator import CLIPEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--edited", required=True)
    parser.add_argument("--source", default="sneaker")
    parser.add_argument("--target", default="sandal")
    args = parser.parse_args()
    from PIL import Image
    original = np.asarray(Image.open(args.original).convert("L"), dtype="float32")[..., None] / 255.0
    edited = np.asarray(Image.open(args.edited).convert("L"), dtype="float32")[..., None] / 255.0
    evaluator = CLIPEvaluator(enabled=True)
    result = evaluator.evaluate(original, edited, args.source, args.target)
    print(result)


if __name__ == "__main__":
    main()
