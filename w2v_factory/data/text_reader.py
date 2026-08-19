from __future__ import annotations

from collections.abc import Callable, Iterable

from ..registry import TOKENIZER_REG


@TOKENIZER_REG.register("simple")
def simple_tokenizer(line: str, lowercase: bool = True) -> list[str]:
    """简单分词：按空白切分；可统一小写。"""
    if lowercase:
        line = line.lower()
    return line.strip().split()


def iter_tokens(files: list[str], lowercase: bool, tokenizer_name: str, max_sent_len: int) -> Iterable[list[str]]:
    """逐行读取多个文件，产出 token 序列（单行即一条“句子”）。"""
    tokenizer: Callable[[str], list[str]] = TOKENIZER_REG.get(tokenizer_name)
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                toks = tokenizer(line, lowercase=lowercase)
                if not toks:
                    continue
                if len(toks) > max_sent_len:
                    toks = toks[:max_sent_len]  # 防止超长行占满内存
                yield toks
