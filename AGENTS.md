# Agents Guide — noble-dolphin

This project is a pure-Python reimplementation of Karpathy's MicroGPT
(`microgpt.py`), a 200-line dependency-free GPT trainer and sampler.

## Environment — MUST use `uv`

**All work in this folder must be done through [`uv`](https://docs.astral.sh/uv/).**

The Python environment, virtualenv, and dependency lock are all managed by uv.
Do NOT use system `python`, `pip`, or any other virtualenv tool — it will
desync the lockfile and break reproducibility.

### Setup (first time)

```bash
uv sync          # creates .venv and installs from uv.lock
```

### Running scripts

Always prefix commands with `uv run` so they execute inside the managed venv:

```bash
uv run python microgpt.py
```

### Adding a dependency

```bash
uv add <package>        # adds to pyproject.toml and updates uv.lock
uv remove <package>     # removes and relocks
```

Never edit `pyproject.toml` dependencies by hand without re-running `uv lock`.

### Key files

| File                   | Purpose                                      |
|------------------------|----------------------------------------------|
| `microgpt.py`          | Pure-Python GPT (train + inference)          |
| `pyproject.toml`       | Project metadata + dependencies              |
| `uv.lock`              | Pinned, reproducible lockfile (commit this)  |
| `.python-version`      | Python version pinned by uv                  |
| `.venv/`               | uv-managed virtualenv (do NOT commit)        |
| `.gitignore`           | Excludes `.venv/`, caches, and generated files |

### Checklist before finishing a task

1. `uv run python <script>` runs without import errors.
2. `uv.lock` is in sync — run `uv lock --check` if unsure.
3. Do NOT commit `.venv/` — it is in `.gitignore` for a reason.