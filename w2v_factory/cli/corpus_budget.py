"""Create deterministic nested word-token budgets for Table 2 scaling experiments.

One streaming pass; no synthetic padding or repeated corpus epochs. Sentence
boundaries are preserved except for the final partial sentence at a budget edge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

from ..data.text_reader import iter_tokens


def make_corpus_budgets(
    inputs: list[str],
    budgets: list[int],
    output_dir: str | Path,
    *,
    lowercase: bool = True,
    tokenizer: str = "simple",
    max_sent_len: int = 10000,
    force: bool = False,
) -> dict:
    if not inputs or not all(Path(p).is_file() for p in inputs):
        raise ValueError("Provide existing input text files")
    if not budgets or any(type(n) is not int or n <= 0 for n in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be distinct positive integer token counts")
    if max_sent_len <= 0:
        raise ValueError("max_sent_len must be positive")
    ordered = sorted(budgets)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    targets = [root / f"tokens-{n}.txt" for n in ordered]
    manifest = root / "budgets.json"
    if not force and (manifest.exists() or any(p.exists() for p in targets)):
        raise FileExistsError("Budget outputs already exist; use --force explicitly")
    temporary = [p.with_suffix(".txt.partial") for p in targets]
    counts = [0] * len(ordered)
    sentences = [0] * len(ordered)
    digests = [hashlib.sha256() for _ in ordered]
    source_digest = hashlib.sha256()
    processed_tokens = 0
    try:
        with ExitStack() as stack:
            handles = [stack.enter_context(p.open("w", encoding="utf-8", newline="\n")) for p in temporary]
            for tokens in iter_tokens(inputs, lowercase, tokenizer, max_sent_len):
                source_digest.update((" ".join(tokens) + "\n").encode("utf-8"))
                processed_tokens += len(tokens)
                for i, limit in enumerate(ordered):
                    take = min(len(tokens), limit - counts[i])
                    if take <= 0:
                        continue
                    line = " ".join(tokens[:take]) + "\n"
                    handles[i].write(line)
                    digests[i].update(line.encode("utf-8"))
                    counts[i] += take
                    sentences[i] += 1
                if counts[-1] == ordered[-1]:
                    break
        if counts != ordered:
            raise ValueError(
                f"Source contains too few tokens: consumed={processed_tokens}, requested_max={ordered[-1]}"
            )
        for tmp, dest in zip(temporary, targets, strict=True):
            os.replace(tmp, dest)
        data = {
            "format": "word2vec-factory-nested-token-budgets-v1",
            "input_paths": [str(Path(p).resolve()) for p in inputs],
            "tokenizer": tokenizer,
            "lowercase": lowercase,
            "max_sent_len": max_sent_len,
            "source_prefix_token_sha256": source_digest.hexdigest(),
            "processed_source_tokens": processed_tokens,
            "selection": "original document order, fixed prefix; source order must be randomized upstream if appropriate",
            "slices": [
                {"requested_tokens": n, "written_tokens": count,
                 "written_sentences": num_sentences, "sha256": digest.hexdigest(),
                 "file": str(target)}
                for n, count, num_sentences, digest, target in zip(
                    ordered, counts, sentences, digests, targets, strict=True
                )
            ],
        }
        temp_manifest = manifest.with_suffix(".json.partial")
        temp_manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp_manifest, manifest)
        return data
    finally:
        for tmp in temporary:
            tmp.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic nested token budgets for 1301.3781 Table 2")
    p.add_argument("--input", nargs="+", required=True, help="Ordered input text files")
    p.add_argument("--budgets", nargs="+", required=True, type=int)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tokenizer", default="simple")
    p.add_argument("--max-sent-len", type=int, default=10000)
    p.add_argument("--preserve-case", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    report = make_corpus_budgets(
        args.input, args.budgets, args.out_dir,
        lowercase=not args.preserve_case, tokenizer=args.tokenizer,
        max_sent_len=args.max_sent_len, force=args.force,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
