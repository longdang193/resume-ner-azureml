# Mypy Type Safety Rollout Plan (Cursor + `uvx`) — `resume-ner-azureml`

## Goal

Introduce **enforced Python type safety** for the maintainable code in `src/` (and later `tests/`) using **Mypy**, in a way that:

- Catches type/shape bugs early (especially around config/dataset/model outputs)
- Makes refactors safer
- Improves Cursor agent reliability (less “guessing” about structures)
- Avoids boiling-the-ocean changes by rolling out folder-by-folder

Non-goals (initial rollout):

- Type-checking notebooks directly
- Rewriting major architecture to satisfy types

---

## Preconditions

- You can run commands from the repo root (`/workspaces/resume-ner-azureml`).
- Preferred workflow uses **headless tool execution**:

```bash
uvx mypy ...
```

If `uvx` isn’t available, install `uv` once (environment dependent).

---

## Phase 0 — Baseline & Scope

### Decide the initial scope

Start with the highest-signal code paths:

- `src/training/core`
- Then: `src/data`, `src/core`, `src/evaluation`, `src/selection`
- Finally: `tests`

Keep notebooks as orchestration; move reusable logic into `src/` and type-check there.

---

## Phase 1 — Add Mypy configuration (`pyproject.toml`)

### 1.1 Create `pyproject.toml` (repo root)

Add a minimal config that:

- Targets `src` and `tests`
- Enables strict mode (or “almost strict” if needed)
- Ignores missing stubs for heavy ML deps (to reduce noise)

Recommended starter config (TOML override style, suitable for `pyproject.toml`):

```toml
[tool.mypy]
python_version = "3.10"

# src/ layout: tell mypy where your code lives
mypy_path = ["src"]
explicit_package_bases = true

# Start by checking only src/ during rollout; add "tests" later in the plan
files = ["src"]

# Start strict, but relax via overrides if needed (see below)
strict = true

# High-signal flags that keep the config healthy
warn_unused_ignores = true
warn_redundant_casts = true
warn_unreachable = true
show_error_context = true

# Exclude things we never want to type-check (adjust as needed)
exclude = "(^|/)(\\.venv|\\.git|\\.mypy_cache|mlruns|outputs|tmp|aml_logs)/"

# Per-module overrides (pyproject.toml style)
[[tool.mypy.overrides]]
module = ["torch", "torch.*"]
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["transformers", "transformers.*"]
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["mlflow", "mlflow.*"]
ignore_missing_imports = true

# Example: temporarily relax a noisy area during rollout
# [[tool.mypy.overrides]]
# module = ["training.hpo.*"]
# ignore_errors = true  # or selectively disable some strictness flags
```

Keep `strict = true` and, if noise appears, temporarily relax strictness only via `[[tool.mypy.overrides]]` for specific packages. Remove those overrides as you clean each area so the project drifts toward full strictness rather than away from it.

### 1.2 Confirm Mypy runs

```bash
cd /workspaces/resume-ner-azureml
uvx mypy --version
uvx mypy src --show-error-codes
```

Success criteria:

- Mypy runs and produces a list of actionable errors (even if large).

During rollout, prefer `uvx mypy src --show-error-codes` instead of a bare `uvx mypy` so you do not accidentally type-check `tests/` before the tests phase. Once tests are in scope (files includes `"tests"`), you can switch CI to plain `uvx mypy`.

### 1.3 Confirm src/ package layout

Before tightening strictness, ensure the `src/` layout is coherent:

- Packages intended for import have an `__init__.py`.
- Imports inside `src/` use a consistent style (e.g., `from training.core.trainer import ...`).
- If you see “Cannot find implementation or library stub” for your own modules, revisit `mypy_path` / `explicit_package_bases` and package structure before loosening types.

---

## Phase 1.5 — Set up Cursor Rules for automated type safety (recommended)

To make the Cursor agent automatically follow type safety standards across all sessions, create **persistent Cursor Rules** (`.mdc` files) that will be applied automatically.

### 1.5.1 Create `.cursor/rules/` directory

```bash
mkdir -p .cursor/rules
```

### 1.5.2 Create three rule files

**Rule A: Type Safety + Mypy Loop** (`.cursor/rules/python-type-safety.mdc`)

```markdown
---
name: python-type-safety-mypy
description: Enforce Python type safety using Mypy and automated fix loops.
globs:
  - "src/**/*.py"
  - "tests/**/*.py"
alwaysApply: true
---

## Type Safety Requirements

When modifying Python code in `src/` or `tests/`:
- **Prefer precise types**: Avoid `Any` unless absolutely necessary.
- **Use narrow types**: Prefer `TypedDict`, `dataclass`, `Protocol`, `Literal`, and generics over `dict[str, Any]`.
- If unsure about the exact shape, prefer a minimal `TypedDict` or a generic `Mapping[str, object]` and tighten it later; do not invent deep structures that may not match runtime reality.
- **Suppress narrowly**: If you must suppress an error, use `# type: ignore[error-code]` with a short comment explaining why.

## Mypy Enforcement Loop

When making Python changes:
1. Run: `uvx mypy src --show-error-codes` (or target a specific folder like `src/training/core`).
2. Fix all errors without changing runtime behavior.
3. Re-run Mypy until the targeted scope is clean.
4. Prefer adding shared types in `src/common/types.py` rather than duplicating annotations.

## Reference Configuration

See `@pyproject.toml` for Mypy strictness settings.
```

**Rule B: Dependency Hygiene** (`.cursor/rules/python-deps.mdc`)

```markdown
---
name: python-deps-hygiene
description: Keep dependencies reproducible; avoid ad-hoc installs in library code.
globs:
  - "**/*"
alwaysApply: true
---

## Dependency Rules

- **No `pip install` in library code**: Do not add `pip install` commands to `src/` code or tests.
- **Notebooks exception**: Notebooks may use `%pip install` for interactive sessions, but prefer documenting dependencies in one place.
- **Use headless tools**: When running tooling, prefer `uvx <tool>` (e.g., `uvx mypy`, `uvx pytest`) rather than installing tools into the environment.
```

**Rule C: Notebook Policy** (`.cursor/rules/notebooks-thin.mdc`)

```markdown
---
name: notebooks-thin-orchestration
description: Keep notebooks as orchestration; move reusable logic into src/.
globs:
  - "notebooks/**/*.ipynb"
alwaysApply: true
---

## Notebook Rules

- **Keep notebooks thin**: Notebooks should be orchestration, visualization, and calling functions.
- **Extract reusable logic**: If logic grows (selection, preprocessing, metrics, config parsing), extract it into `src/` as typed functions and import them.
- **Type-check the real logic**: New reusable logic must go into `src/` with type hints so Mypy can check it.
```

### 1.5.3 Enable command allow-list in Cursor

In Cursor settings, add these commands to the **command allow-list** (Settings → Agent → Command Allow-list or similar):

- `uvx mypy`
- `uvx pytest` (optional, if you use it)
- `uvx ruff` / `uvx black` (optional, if you use them)

This allows the agent to run type checks autonomously without asking permission each time.

### 1.5.4 Verify rules are active

After creating the rules, start a new Cursor chat and ask:

> "What are the type safety requirements for this project?"

The agent should reference the rules automatically.

Success criteria:

- `.cursor/rules/` directory exists with 3 `.mdc` files.
- Cursor agent automatically follows type safety standards in new chats.
- Agent can run `uvx mypy` without manual approval (if allow-list is configured).

---

## Phase 1.6 — Prevent redundant features and reduce test churn (recommended)

Strict typing helps correctness, but it does not automatically prevent:

- **Redundant code/features** (agent creates new modules instead of reusing existing ones)
- **High test churn** (refactors force deleting/rewriting many tests)

To address this, add **guardrails** that force reuse, stabilize entrypoints, and focus tests on contracts.

### 1.6.1 Add “reuse before create” rule

Create `.cursor/rules/python-reuse-first.mdc`:

```markdown
---
name: python-reuse-first
description: Prefer reusing and extending existing modules over adding new ones.
globs:
  - "src/**/*.py"
  - "tests/**/*.py"
alwaysApply: true
---

## Reuse-first rules

Before creating any new:
- module/package in `src/`
- public function/class
- test module under `tests/`

the agent must:

1. **Search for existing implementations**
   - Use symbol and text search in `src/` (e.g., grep / code search) for similar names/behavior.
2. **Propose a reuse plan**
   - If an existing module is related, extend or refactor it instead of creating a new one.
3. **Document the decision**
   - In the PR description or comments, briefly state:
     - “Existing options considered: <modules>”
     - “Reason new code was necessary: <reason>”
- Reuse-first does not justify large, risky refactors; prefer small extensions to existing modules over massive rewrites.
```

Day-to-day prompt:

> Before adding anything new, list existing modules that cover similar concerns and propose how to extend them instead of creating new ones.

### 1.6.2 Add workflow-entrypoints rule

This repo already has workflow-style code (e.g., `src/evaluation/selection/workflows/...`) and workflow tests (e.g., `tests/workflows/...`). Make workflows the stable “golden path” entrypoints.

Create `.cursor/rules/workflow-entrypoints.mdc`:

```markdown
---
name: workflow-entrypoints
description: Consolidate orchestration into workflow modules and avoid duplicated flows.
globs:
  - "src/**/workflows/*.py"
  - "notebooks/**/*.ipynb"
  - "tools/**/*.py"
alwaysApply: true
---

## Workflow rules

- Orchestration (multi-step processes: training, selection, evaluation) must live in `src/**/workflows/`.
- Scripts and notebooks should call existing workflow functions instead of re-implementing the steps.
- When a new flow is needed, first check existing workflows for an extension point.
```

Expected effect:

- New “features” become parameters/branches in existing workflows instead of new duplicated modules.
- Tests target workflow entrypoints rather than ad-hoc scripts.

### 1.6.3 Add testing strategy rule (contracts + layers)

Create `.cursor/rules/testing-strategy.mdc`:

```markdown
---
name: testing-strategy
description: Keep tests stable by focusing on contracts and workflows, not internal wiring.
globs:
  - "tests/**/*.py"
alwaysApply: true
---

## Testing rules

- Prefer testing **public APIs and workflows** (e.g., `run_*_workflow`) instead of internal helpers.
- Avoid over-specific assertions that break on harmless refactors (e.g., exact log strings, full dict equality when only a few fields matter).
- When refactoring internals, keep workflow function signatures and behavior stable so most tests stay valid.
- Before deleting or rewriting a test, check if it can be adapted to the new contract instead.
```

Prompt to reduce churn during refactors:

> Do not delete tests unless they are clearly redundant with an existing workflow-level test. Prefer updating them to exercise the new or changed contract.

### 1.6.4 Apply `@meta` metadata (from `docs/rules/CLEAN_CODE.md`) to heavy files

Use `@meta` blocks (R0) in:

- Entry-point scripts / orchestration / workflows
- Test modules (especially workflow/e2e tests)

This helps both humans and the agent quickly find “what owns what,” reducing accidental duplication and simplifying safe test selection.

### 1.6.5 Upgrade prompts to force impact analysis

Before implementing changes, require the agent to identify reuse candidates and update dependencies coherently:

> I want to change X.
>
> 1) List all modules and tests that currently implement or depend on this behavior.
> 2) Propose a minimal-change plan that updates those in a consistent way.
> 3) Then implement the changes.

Success criteria (Phase 1.6):

- Cursor Rules exist for **reuse-first**, **workflow entrypoints**, and **testing strategy**.
- New work extends existing modules by default (new modules only with documented justification).
- Refactors preserve stable workflow contracts, and most tests are updated (not rewritten).

---

## Phase 2 — Fix types in a tight slice (`src/training/core`)

### 2.1 Run Mypy on the first folder

```bash
uvx mypy src/training/core --show-error-codes
```

### 2.2 Cursor "self-correction" loop

**If Cursor Rules are set up (Phase 1.5):**
The agent will automatically follow type safety standards. You can use a simple prompt:

> Fix all Mypy errors in `src/training/core` and keep runtime behavior unchanged.

The rules will ensure the agent:

- Avoids `Any` unless necessary
- Uses narrow types (`TypedDict`, `Protocol`, `Literal`, generics)
- Adds shared types to `src/common/types.py` when appropriate
- Runs `uvx mypy` iteratively until clean

**If rules are not set up yet:**
Use a more explicit prompt with constraints:

> Fix all `mypy` errors under `src/training/core`.  
> Rules:  
>
> - Avoid `Any` unless absolutely necessary; prefer narrow types (`TypedDict`, `Protocol`, `Literal`, generics).  
> - Prefer adding small, shared types in `src/common/` rather than duplicating annotations.  
> - Keep runtime behavior identical (no functional refactors unless required).  
> - If an API is genuinely dynamic, use a targeted `cast()` or `typing.overload` with a short comment.  
> After each change, rerun `uvx mypy src/training/core --show-error-codes` until clean.

### 2.3 Typical fixes you should expect

- **Add return types** for functions lacking them (`-> None`, `-> tuple[...]`, etc.)
- Replace broad dicts like `dict[str, Any]` with:
  - `TypedDict` for configs/records
  - `Mapping[str, ...]` where mutation isn’t required
- Add explicit `Optional[...]` handling (guard or assert)
- Introduce small helper types:
  - `ConfigTraining`, `ConfigModel`, etc.
  - dataset record shapes

### 2.4 Where to put shared types

Create a small type module:

- `src/common/types.py` (or `src/common/typing.py`)

Use it to centralize `TypedDict` shapes shared across training/eval/selection.

Success criteria (Phase 2):

- `uvx mypy src/training/core --show-error-codes` returns **no errors**.

---

## Phase 3 — Expand folder-by-folder

Repeat the same loop for:

```bash
uvx mypy src/data --show-error-codes
uvx mypy src/core --show-error-codes
uvx mypy src/evaluation --show-error-codes
uvx mypy src/selection --show-error-codes
```

Rules of thumb:

- Fix in the smallest scope first; don’t expand to the next folder until the current one is clean.
- Prefer precise local types over large “global” refactors.

Success criteria (Phase 3):

- `uvx mypy src --show-error-codes` returns **no errors**.

---

## Phase 4 — Bring notebooks under control (without type-checking notebooks)

### 4.1 Move reusable notebook logic into `src/`

For example:

- Notebook currently computes “best config selection” logic.
- Extract the selection logic into `src/selection/best_config.py` with typed inputs/outputs.
- Notebook becomes orchestration that imports typed functions.

Success criteria:

- Notebook logic lives in typed, tested code under `src/`.
- Mypy covers the “real” logic.

Optional:

- Add `nbqa` later if you want to type-check notebooks directly.

---

## Phase 5 — Add a CI gate (optional but recommended)

Once `uvx mypy src` is clean and stable, add a CI step.

### Example (GitHub Actions concept)

- Install `uv`
- Run:

```bash
uvx mypy src --show-error-codes
```

Success criteria:

- PRs fail when type safety regresses.

---

## Operational guidance (how to keep this sustainable)

### Type Safety Practices

- Prefer **narrow types** over `Any`.
- Use `ignore[...]` comments sparingly and always with the specific error code.
- Don't try to type everything at once; keep changes reviewable.
- Keep notebooks thin; keep core logic in `src/`.

### Leveraging Cursor Rules

- **Rules persist across sessions**: Once Phase 1.5 rules are set up, every new Cursor chat automatically enforces type safety standards.
- **No need to repeat instructions**: The agent will automatically run `uvx mypy` and fix errors according to your rules.
- **Start new chats for discrete tasks**: This keeps the context window clean and ensures rules are applied fresh.
- **Review agent changes**: While the agent is effective, occasionally review for overly broad `Any` usage and prompt to narrow types if needed.
