# Zensical Reference Notes

Zensical is a Rust-driven static documentation generator — the successor to Material for MkDocs.
It reads `zensical.toml` (preferred) or `mkdocs.yml` (backward-compatible).
Source: https://zensical.org/docs/

---

## CLI

```bash
zensical new .                    # scaffold project (won't overwrite existing files)
zensical serve                    # dev server with hot-reload (default: localhost:8000)
zensical serve -o                 # open browser automatically
zensical serve -a localhost:9000  # custom address
zensical build                    # build static site → site/
zensical build -c                 # clean build (clear cache first)
zensical build -f path/to/zensical.toml  # explicit config file
```

Not supported (unlike MkDocs): `gh-deploy`, `get-deps`, `--theme`, `--strict`, `--dirty`.

---

## zensical.toml structure

```toml
[project]
site_name        = "Site Name"          # required
site_url         = "https://..."        # needed for instant nav, canonical URLs
site_description = "..."
site_author      = "..."
copyright        = "© 2025 ..."         # footer; HTML allowed
docs_dir         = "docs"              # default
site_dir         = "site"              # default
dev_addr         = "localhost:8000"    # default
use_directory_urls = true              # true → /page/  false → /page.html

extra_css        = ["stylesheets/extra.css"]
extra_javascript = ["javascripts/mathjax.js"]

nav = [...]  # see Navigation section

[project.repo]
url    = "https://github.com/owner/repo"
branch = "main"

[project.theme]
variant    = "modern"          # "modern" (default) or "classic" (matches Material for MkDocs)
language   = "en"              # 60+ language codes
direction  = "ltr"             # or "rtl"
logo       = "assets/logo.png"
favicon    = "assets/favicon.ico"
custom_dir = "overrides"       # MiniJinja template overrides

features = [
    # Navigation
    "navigation.instant",
    "navigation.instant.prefetch",   # experimental
    "navigation.instant.progress",
    "navigation.tracking",
    "navigation.tabs",
    "navigation.sections",
    "navigation.expand",
    "navigation.path",
    "navigation.indexes",
    "navigation.prune",
    "navigation.top",
    "toc.follow",
    "toc.integrate",
    # Content
    "content.code.copy",
    "content.code.select",
    "content.footnote.tooltips",
    "content.tabs.link",
    "content.action.edit",
    "content.action.view",
    # Header / search
    "header.autohide",
    "announce.dismiss",
    "search.highlight",
]

# Color palette (light/dark toggle)
[[project.theme.palette]]
scheme  = "default"   # light
primary = "indigo"    # or "custom"
accent  = "blue"

[[project.theme.palette]]
scheme  = "slate"     # dark
primary = "indigo"
accent  = "blue"
# toggle icon/name goes here for the button

[project.markdown_extensions]
# ... see below
```

### extra_javascript with attributes

```toml
[[project.extra_javascript]]
path  = "javascripts/extra.js"
type  = "module"    # ES module
async = true
defer = false
```

---

## Navigation

```toml
[project]
nav = [
    { "Home"         = "index.md" },
    { "About"        = "about.md" },
    { "External"     = "https://github.com/example" },
    { "User Guide"   = [
        { "Installation" = "guide/install.md" },
        { "Usage"        = "guide/usage.md" },
        { "Advanced"     = [
            { "Config" = "guide/advanced/config.md" },
        ]},
    ]},
]
```

- If `nav` is omitted, Zensical auto-generates it from the folder structure.
- Any path that doesn't resolve to a `.md` file is treated as an external URL.
- `navigation.indexes` feature: a section whose first entry matches the section name
  becomes a clickable index page for that section.

---

## Markdown extensions

Zensical enables a rich set by default (unlike bare MkDocs).
To revert to minimal MkDocs behavior: `markdown_extensions = {}`.

### Python-Markdown built-ins

```toml
[project.markdown_extensions.abbr]         # abbreviation tooltips
[project.markdown_extensions.admonition]   # !!! note / !!! warning blocks
[project.markdown_extensions.attr_list]    # {.class #id} attributes on elements
[project.markdown_extensions.def_list]     # definition lists
[project.markdown_extensions.footnotes]    # [^1] footnotes
[project.markdown_extensions.md_in_html]   # markdown="1" inside HTML tags
[project.markdown_extensions.tables]       # GFM-style tables

[project.markdown_extensions.toc]
permalink       = true
toc_depth       = "1-6"
title           = "Contents"   # optional custom heading
```

### pymdownx extensions

```toml
[project.markdown_extensions."pymdownx.arithmatex"]
# Math — see Math section below

[project.markdown_extensions."pymdownx.caret"]
# ^^insert^^ and A^sup^

[project.markdown_extensions."pymdownx.details"]
# ??? collapsible admonitions

[project.markdown_extensions."pymdownx.emoji"]
# :smile: shortcodes; Twemoji SVGs by default

[project.markdown_extensions."pymdownx.highlight"]
anchor_linenums = true
line_spans      = "__span"
# Note: pygments_lang_as_class is not a valid option in current pymdownx versions

[project.markdown_extensions."pymdownx.inlinehilite"]
# `:::python inline_code`

[project.markdown_extensions."pymdownx.keys"]
# ++ctrl+alt+del++ → styled key symbols

[project.markdown_extensions."pymdownx.mark"]
# ==highlighted==

[project.markdown_extensions."pymdownx.smartsymbols"]
# (c) → © , (tm) → ™ , fractions, arrows

[project.markdown_extensions."pymdownx.snippets"]
# --8<-- "path/to/file"  (transclusion)

[project.markdown_extensions."pymdownx.superfences"]
# arbitrary nesting; Mermaid diagrams
# custom_fences = [
#   { name = "mermaid", class = "mermaid",
#     format = "pymdownx.superfences.fence_code_format" }
# ]

[project.markdown_extensions."pymdownx.tabbed"]
alternate_style = true   # required for modern styling

[project.markdown_extensions."pymdownx.tasklist"]
custom_checkbox = true

[project.markdown_extensions."pymdownx.tilde"]
# ~~strikethrough~~ and H~sub~
```

---

## Math support

### Math rendering

The `pymdownx.arithmatex` extension enables both inline (`$...$`) and block (`$$...$$`) math.

To add MathJax (full LaTeX support):

```toml
[project]
extra_javascript = [
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
]
```

To add KaTeX (faster, simpler):

```toml
[project]
extra_css = [
    "https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.css",
]
extra_javascript = [
    "https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.js",
    "https://cdn.jsdelivr.net/npm/katex@0.16/dist/contrib/auto-render.min.js",
]
```

**Note:** The `generic_mode` option is not valid in current pymdownx versions and should not be used.

### Syntax in markdown

```
Inline: $E = mc^2$  or  \(E = mc^2\)

Block:
$$
\begin{align}
  \frac{dx}{dt} &= v \sin\theta \\
  \frac{dy}{dt} &= -v \cos\theta
\end{align}
$$
```

---

## Sphinx Alternative

A Sphinx configuration is also available for comparison and development. See `docs/SPHINX_BUILD.md` for details.

### When to use Sphinx instead of Zensical:

- **Code execution**: Sphinx can execute Python code blocks and display results (via `jupyter-sphinx`)
- **PDF output**: Built-in PDF generation (requires LaTeX)
- **Versioning**: Native documentation versioning support
- **API docs**: Better integration with Python docstrings (`sphinx.ext.autodoc`)
- **Multiple formats**: ePub, man pages, etc.

### When to use Zensical:

- **Speed**: Zensical builds significantly faster
- **Simplicity**: Easier configuration for basic documentation
- **Modern UX**: Zensical's theme is more modern out-of-the-box
- **TOML config**: Simpler configuration format vs Python

### Setup:

```bash
cd docs
pip install -r requirements-sphinx.txt
make html
make serve
```

---

## Math rendering issues

### Symptom: LaTeX math not rendering (align blocks display as raw text)

**Cause:** Block math with `\begin{align}...\end{align}` must be wrapped in `$$...$$` delimiters for pymdownx.arithmatex to recognize and process them.

**Solution:** Ensure all LaTeX blocks use the proper delimiter syntax:

```markdown
# Correct:
$$
\begin{align}
  x = y + z
\end{align}
$$

# Incorrect (won't render):
\begin{align}
  x = y + z
\end{align}
```

---

## Troubleshooting

### Warning: unresolved link reference (bibliography anchors)

**Symptom:** Build shows warnings like:
```
Warning: anchor does not exist
[Raymer (2012)](bibliography.md#aircraft-design-a-conceptual-approach)
```

**Cause:** The anchor ID you specified in the markdown doesn't match the auto-generated heading ID that Zensical creates. Zensical converts headings to lowercase, hyphenated IDs.

**Solution:** Create explicit subsection headings in your bibliography file with simple, predictable anchor IDs. For example:

```markdown
## Aeronautics and Aircraft Design

### Raymer 2012
...

## Water Rocket Physics

### Prusa 2000
...
```

Then link to `#raymer-2012` or `#prusa-2000` (Zensical auto-converts to lowercase and hyphens).

Avoid anchor IDs that contain special characters or long descriptive text, as the auto-generated ID may not match your link.

---

### Warning: unresolved link reference (citations)

**Symptom:** Build shows warnings like:
```
Warning: unresolved link reference
[@raymer2012aircraft]
```

**Cause:** Zensical does not currently support Pandoc-style citations (`[@key]`) or Jupyter Book `{cite}` directive out of the box. These are being interpreted as broken link references.

**Solution:** Zensical is a static site generator without built-in bibliography support. For now:

1. **Remove inline citations**: Replace `[@key]` with plain text references like "Raymer 2012"
2. **Use footnotes instead** (if full citation needed):
   ```markdown
   This is a reference[^1].

   [^1]: Raymer, D. P. (2012). Aircraft Design: A Conceptual Approach. ...
   ```
3. **Build a bibliography page**: Create a `bibliography.md` file with full citations and link to it

For more robust citation support in future versions, Zensical may add bibliography plugins (targeted for 2026).

---

### Warning: unresolved link reference (table cells)

**Symptom:** Warnings for table cells containing text like `[initial]` or `[final]`:
```
Warning: unresolved link reference
[initial]
```

**Cause:** Text in the format `[word]` in any markdown context is interpreted as a link reference.

**Solution:** Escape the brackets or use HTML entities:
```markdown
| Column1        | Column2        | Description    |
|----------------|----------------|----------------|
| phase\[initial\] | other\[final\] | description   |
```

Or use inline code:
```markdown
| Column1        | Column2        | Description    |
|----------------|----------------|----------------|
| phase `[initial]` | other `[final]` | description   |
```

---

### KeyError: 'nav'

**Symptom:** `zensical serve` fails with `KeyError: 'nav'` or similar when rendering markdown.

**Cause:** In TOML, once a subsection like `[project.theme]` is opened, you cannot add simple keys back to the parent `[project]` section at the end of the file. The key gets misinterpreted as part of the last subsection.

**Solution:** Ensure all simple `[project]` keys (like `nav`, `site_name`, etc.) are defined **before** any subsections (like `[project.repo]`, `[project.theme]`, `[project.markdown_extensions]`). The correct structure is:

```toml
[project]
site_name = "..."
nav = [...]    # Must come before subsections

[project.repo]
url = "..."

[project.theme]
# ...
```

---

### KeyError: 'generic_mode'

**Symptom:** `zensical serve` fails with `KeyError: 'generic_mode'` in the pymdownx.arithmatex extension.

**Cause:** The `generic_mode` option is not a valid configuration parameter in current versions of pymdownx.

**Solution:** Remove `generic_mode` from the `[project.markdown_extensions."pymdownx.arithmatex"]` section in `zensical.toml`. Instead, configure math rendering by adding the appropriate CDN links to `extra_javascript` or `extra_css` in the main `[project]` section.

---

### KeyError: 'pygments_lang_as_class'

**Symptom:** `zensical serve` fails with `KeyError: 'pygments_lang_as_class'` in the pymdownx.highlight extension.

**Cause:** The `pygments_lang_as_class` option is not a valid configuration parameter in current versions of pymdownx. It may have been removed or never existed.

**Solution:** Remove `pygments_lang_as_class` and `auto_title` from the `[project.markdown_extensions."pymdownx.highlight"]` section in `zensical.toml`. Keep only valid options like `anchor_linenums` and `line_spans`.

---

### TOMLDecodeError: Cannot declare ('project',) twice

**Symptom:** `zensical serve` fails with `TOMLDecodeError: Cannot declare ('project',) twice (at line X, column Y)`

**Cause:** The `zensical.toml` file contains duplicate `[project]` section headers. TOML does not allow the same section to be declared twice.

**Solution:** Merge all `[project]` configuration into a single `[project]` block at the top of the file. Sub-tables like `[project.theme]` and `[project.markdown_extensions]` are allowed, but the root `[project]` header should appear only once. For example:

```toml
# Correct structure
[project]
site_name = "Site"
# ... all project-level settings ...

[project.theme]
# ... theme settings ...

[project.markdown_extensions]
# ... extensions ...

nav = [...]  # Can be directly under [project] scope at the end of the file
```

---

## mkdocstrings (API docs)

Status: **preliminary** as of v0.0.11. Being rethought for greater flexibility.

```bash
pip install mkdocstrings-python
```

```toml
[project.markdown_extensions.mkdocstrings]
handlers = "python"

[project.markdown_extensions.mkdocstrings.python]
paths              = ["src"]
docstring_style    = "numpy"    # or "google"
inherited_members  = true
show_source        = true
```

Usage in a `.md` file:

```markdown
::: dymos.Phase
    options:
      show_source: true
      docstring_style: numpy
```

Known limitations:
- No backlinks yet
- File watching only covers `docs_dir`; external source paths need symlinks
- `objects.inv` generation available (v0.0.19+) for cross-project links

---

## Front matter (per-page metadata)

```yaml
---
title: Custom Page Title
description: SEO description
icon: material/rocket
status: new          # or "deprecated"
tags: [optimal-control, dymos]
hide:
  - navigation
  - toc
  - path
search:
  exclude: true
---
```

---

## Template overrides (MiniJinja)

```
overrides/
  main.html          # override base template
  partials/
    footer.html
```

```html
{% extends "base.html" %}

{% block scripts %}
  {{ super() }}
  <script src="{{ 'javascripts/extra.js' | url }}"></script>
{% endblock %}
```

Available blocks: `htmltitle`, `scripts`, `styles`, `header`, `footer`,
`announce`, `content`, `tabs`, `toc`, `sidebar`, `navigation`, `main`.

---

## Code execution

**Zensical has no native code execution.** It is a static site generator.
For executed code blocks, use an external pre-processing step that runs the
Python code and injects outputs into the markdown before `zensical build`.

---

## Plugin / module system

Currently road-testing internally; public API targeted for **early 2026**.
Supported MkDocs plugins are mapped to native Zensical modules (search,
redirects, tags, blog, git, mkdocstrings, etc.).

---

## Key differences from MkDocs / Material for MkDocs

| Aspect | MkDocs + Material | Zensical |
|---|---|---|
| Config format | `mkdocs.yml` (YAML) | `zensical.toml` (TOML); also reads `mkdocs.yml` |
| Build engine | Python | Rust (`ZRX` differential runtime) |
| Rebuild speed | Full rebuild | 4-5× faster via differential builds |
| Templating | Jinja2 | MiniJinja |
| Extensions | Opt-in | Rich set enabled by default |
| Material status | Entering maintenance mode | Active development |
| Code execution | Via mkdocs-jupyter / nbconvert | Not built-in |
| Module system | Plugin hooks | Explicit dependency graph (coming 2026) |

Zensical reads `mkdocs.yml` natively through a compatibility layer —
existing MkDocs projects work with zero changes.
