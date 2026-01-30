# CLAUDE.md

---
name: dymos
description: A package for dynamical system optimization using OpenMDAO.
---

## Project Overview

This repository contains the code for a fitness tracking application. The goal is to provide a clean, modern interface and robust backend services.

## Code Style & Conventions

Unless otherwise directed...

*   Follow PEP8 whenever possible. All changes should pass `ruff check` with rules setup in the pyproject.toml file.
*   Prefer single quotes over double quotes, except for the triple-quotes specifying docstrings. If nested quotes are necessary, err on the side of the user seeing single quotes in output.
*   Avoid use of emojis in output.

## Environment defaults
*   Use the "dev" environment in pixi.toml for development work.
*   Tests should be executed under the "dev" environment using the pixi task `pixi run test`

## Other behavior rules
*   Do not modify the environment, adding or removing packages, without explicit permission.
*   Do not commit to git, allow the user to do so.

