"""Keep the repository's canonical Markdown navigation from drifting.

Historic implementation plans under docs/superpowers are intentionally excluded;
they are preserved as dated evidence rather than current documentation.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def test_current_documentation_links_resolve():
    documents = [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "configs/README.md",
        *(ROOT / "docs").glob("*.md"),
        ROOT / "docs/superpowers/README.md",
    ]
    missing: list[str] = []
    for document in documents:
        assert document.is_file()
        content = document.read_text(encoding="utf-8")
        for link in LINK.findall(content):
            destination = link.strip().split()[0].strip("<>")
            if destination.startswith(("#", "mailto:", "http:", "https:")):
                continue
            clean = unquote(urlsplit(destination).path)
            if not clean:
                continue
            path = document.parent / clean
            if not path.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {destination}")
    assert not missing, "Broken relative Markdown links:\n" + "\n".join(missing)
