"""Smoke-test the native trainer's no-validation checkpoint/resume path."""
from __future__ import annotations

import argparse
import struct
import subprocess
import tempfile
from pathlib import Path


def write_tiny_corpus(path: Path) -> None:
    tokens = [257, 65, 66, 258]
    masks = [0, 0, 1, 1]
    payload = bytearray(struct.pack("<8sIIQ", b"FLAWDTOK", 2, 259, len(tokens)))
    for token, mask in zip(tokens, masks):
        payload.extend(struct.pack("<HB", token, mask))
    path.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    args = parser.parse_args()
    trainer = Path(args.trainer).resolve()
    if not trainer.is_file():
        raise SystemExit(f"Trainer executable not found: {trainer}")

    with tempfile.TemporaryDirectory(prefix="flawd-trainer-smoke-") as temporary:
        root = Path(temporary)
        corpus = root / "tiny.tokens"
        checkpoint = root / "resume.flawd"
        write_tiny_corpus(corpus)

        base = [str(trainer), str(corpus), str(checkpoint), "1", "-", "-",
                "4", "4", "8", "1", "2"]
        first = subprocess.run(base, capture_output=True, text=True, check=False)
        if first.returncode != 0:
            raise SystemExit("Initial training failed:\n" + first.stdout + first.stderr)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise SystemExit("Initial training did not save a checkpoint.")

        resumed = list(base)
        resumed[5] = str(checkpoint)
        second = subprocess.run(resumed, capture_output=True, text=True, check=False)
        if second.returncode != 0:
            raise SystemExit("Checkpoint resume failed:\n" + second.stdout + second.stderr)
        if "optimizer updates: 2" not in second.stdout:
            raise SystemExit("Resume did not restore the optimizer update counter:\n" + second.stdout)

        print("FLAWD trainer CLI no-validation checkpoint/resume integration passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
