#!/usr/bin/env python
"""
Convert docs/dymos_book Jupyter notebooks to pure markdown in docs/markdown.

- Markdown cells are emitted as-is.
- Code cells become ```python fenced blocks (outputs are dropped).
- Raw cells are emitted as-is (typically RST/MyST directives).
- All other assets (images, .md, .yml, .bib, etc.) are copied unchanged.
- .ipynb files become .md files; everything else keeps its original name.
"""

import json
import shutil
from pathlib import Path


SRC = Path(__file__).parent / 'dymos_book'
DST = Path(__file__).parent / 'markdown'


def cell_to_text(cell: dict) -> str | None:
    """Return the markdown representation of a single notebook cell, or None to skip."""
    cell_type = cell.get('cell_type', 'code')
    source = cell.get('source', [])

    # source is either a list of strings or a single string
    content = ''.join(source) if isinstance(source, list) else source

    if not content.strip():
        return None

    if cell_type == 'markdown':
        return content

    if cell_type == 'code':
        # Preserve any execution-engine tags as a comment at the top of the fence
        tags = cell.get('metadata', {}).get('tags', [])
        tag_comment = ''
        if tags:
            tag_comment = '# tags: ' + ', '.join(tags) + '\n'
        return f'```python\n{tag_comment}{content}\n```'

    if cell_type == 'raw':
        # Raw cells often contain MyST/RST markup; emit as plain text
        return content

    return None


def notebook_to_markdown(nb_path: Path) -> str:
    """Parse a .ipynb file and return its markdown representation."""
    with open(nb_path, encoding='utf-8') as fh:
        nb = json.load(fh)

    parts = []
    for cell in nb.get('cells', []):
        text = cell_to_text(cell)
        if text is not None:
            parts.append(text)

    return '\n\n'.join(parts) + '\n'


def convert(src: Path = SRC, dst: Path = DST) -> None:
    converted = 0
    copied = 0
    skipped = 0

    for src_path in sorted(src.rglob('*')):
        if src_path.is_dir():
            continue

        rel = src_path.relative_to(src)
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if src_path.suffix == '.ipynb':
            md_path = dst_path.with_suffix('.md')
            try:
                content = notebook_to_markdown(src_path)
                md_path.write_text(content, encoding='utf-8')
                print(f'  converted  {rel}  ->  {md_path.relative_to(dst)}')
                converted += 1
            except Exception as exc:
                print(f'  ERROR      {rel}: {exc}')
                skipped += 1
        else:
            shutil.copy2(src_path, dst_path)
            print(f'  copied     {rel}')
            copied += 1

    print(f'\nDone: {converted} notebooks converted, {copied} files copied, {skipped} errors.')


if __name__ == '__main__':
    convert()
