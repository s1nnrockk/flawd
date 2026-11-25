"""Exercise Python BPE export, native training, and native chat as one pipeline."""
import json
from pathlib import Path
import subprocess
import tempfile

from flawd.corpus import tokenize_jsonl
from flawd.tokenizer import ByteBPETokenizer


def main() -> int:
    repository = Path(__file__).resolve().parent.parent
    train_exe = repository / "build" / "Release" / "flawd_train.exe"
    chat_exe = repository / "build" / "Release" / "flawd_chat.exe"
    if not train_exe.is_file() or not chat_exe.is_file():
        raise RuntimeError("Build the Release C++ executables before running this integration test")

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        train_jsonl = directory / "train.jsonl"
        validation_jsonl = directory / "validation.jsonl"
        tokenizer_path = directory / "tokenizer.bpe"
        train_tokens = directory / "train.tokens"
        validation_tokens = directory / "validation.tokens"
        checkpoint = directory / "model.flawd"

        train_texts = [
            "hello world, this is FLAWD.",
            "hello there, this is a language model.",
            "FLAWD learns from clean training examples.",
        ] * 8
        validation_texts = [
            "a separate validation sentence.",
            "the validation corpus is held out.",
        ]
        train_jsonl.write_text(
            "".join(json.dumps({"text": text}) + "\n" for text in train_texts),
            encoding="utf-8",
        )
        validation_jsonl.write_text(
            "".join(json.dumps({"text": text}) + "\n" for text in validation_texts),
            encoding="utf-8",
        )

        tokenizer = ByteBPETokenizer.train(train_texts, vocab_size=280, min_frequency=2)
        tokenizer.save(tokenizer_path)
        tokenize_jsonl(train_jsonl, train_tokens, tokenizer_path=tokenizer_path)
        tokenize_jsonl(validation_jsonl, validation_tokens, tokenizer_path=tokenizer_path)

        training = subprocess.run(
            [str(train_exe), str(train_tokens), str(checkpoint), "1", str(validation_tokens),
             "-", "64", "8", "16", "1", "4"],
            check=True, capture_output=True, text=True, timeout=120,
        )
        if "FLAWD training complete" not in training.stdout or not checkpoint.is_file():
            raise RuntimeError("Native BPE training did not produce a checkpoint")

        chat = subprocess.run(
            [str(chat_exe), str(checkpoint), "0", "40", str(tokenizer_path)],
            input="hello\nwhat is 2+2?\n\n", capture_output=True, text=True, timeout=60,
        )
        if (chat.returncode != 0 or "FLAWD local model loaded" not in chat.stdout
                or chat.stdout.count("You>") < 2):
            raise RuntimeError(
                "Native BPE inference failed.\n"
                + chat.stdout + "\n" + chat.stderr
            )
        wrong_merges = list(tokenizer.merges)
        replacement_pair = next(
            pair for pair in ((0, 0), (1, 1), (2, 2), (3, 3))
            if pair not in wrong_merges
        )
        wrong_merges[-1] = replacement_pair
        wrong_tokenizer_path = directory / "wrong-tokenizer.bpe"
        wrong_tokenizer = ByteBPETokenizer(wrong_merges)
        wrong_tokenizer.save(wrong_tokenizer_path)
        wrong_validation_tokens = directory / "wrong-validation.tokens"
        tokenize_jsonl(validation_jsonl, wrong_validation_tokens,
                       tokenizer_path=wrong_tokenizer_path)
        validation_mismatch = subprocess.run(
            [str(train_exe), str(train_tokens), str(directory / "invalid.flawd"),
             "1", str(wrong_validation_tokens)],
            capture_output=True, text=True, timeout=30,
        )
        if (validation_mismatch.returncode == 0
                or "tokenizer fingerprints do not match" not in validation_mismatch.stderr):
            raise RuntimeError("Native trainer accepted validation data from a different BPE tokenizer")
        mismatch = subprocess.run(
            [str(chat_exe), str(checkpoint), "0", "40", str(wrong_tokenizer_path)],
            input="hello\\n\\n", capture_output=True, text=True, timeout=30,
        )
        if mismatch.returncode == 0 or "does not match the checkpoint tokenizer" not in mismatch.stderr:
            raise RuntimeError("Native chat accepted a different BPE merge table with the same vocabulary size")
        print("Python BPE export -> C++ training -> multi-turn inference and tokenizer validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
