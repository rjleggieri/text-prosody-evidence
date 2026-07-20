"""Fail on broken local Markdown links."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    broken: list[str] = []
    for document in sorted(root.rglob("*.md")):
        if ".git" in document.parts:
            continue
        for raw_target in LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = unquote(target.split("#", maxsplit=1)[0])
            if path_text and not (document.parent / path_text).resolve().exists():
                broken.append(f"{document.relative_to(root)} -> {target}")
    if broken:
        raise SystemExit("Broken local Markdown links:\n" + "\n".join(broken))


if __name__ == "__main__":
    main()
