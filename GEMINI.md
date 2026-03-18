# Gemini CLI Context

This file contains persistent instructions and context for Gemini CLI when working in this repository.

## Coding Standards
- **Linting:** All Python changes must pass `ruff check .` cleanly before being considered complete.
- **String Formatting:** Always prefer single quotes (`'`) for strings.
  - **Exception 1:** Docstrings must always use triple double-quotes (`"""`).
  - **Exception 2:** When nesting strings (e.g., inside an f-string or dictionary access), generally prefer single quotes on the inner level if possible, using double quotes on the outer level if necessary to avoid escaping (e.g., `f"User: {data['name']}"`).
