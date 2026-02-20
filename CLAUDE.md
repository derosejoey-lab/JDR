# CLAUDE.md — AI Assistant Guide for JDR

This file provides guidance for AI coding assistants (Claude, Copilot, etc.) working in this repository.

---

## Repository Overview

| Field | Value |
|-------|-------|
| **Name** | JDR (JD Repository) |
| **State** | Early-stage / empty project |
| **Branch model** | Feature branches off `master` |
| **Remote** | `origin` (derosejoey-lab/JDR) |

The project currently contains only an initialised git repository and a minimal `README.md`. All structural conventions below are intended to guide development as the project grows.

---

## Repository Structure

```
JDR/
├── CLAUDE.md        # This file — AI assistant guidance
└── README.md        # Project overview (minimal at present)
```

As source code, tests, and configuration are added, update this section to reflect the actual layout.

---

## Git Workflow

### Branching

- `master` — stable, production-ready code
- `feature/<short-description>` — new features
- `fix/<short-description>` — bug fixes
- `chore/<short-description>` — maintenance / non-functional changes
- `claude/<task-id>` — branches created by AI assistants for specific tasks

### Commits

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<optional scope>): <short summary>

[optional body]
```

Common types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`

Examples:
```
feat: add user authentication module
fix(api): handle null response from external service
docs: update README with setup instructions
chore: add .gitignore for Python artefacts
```

### Push Rules

- Never force-push to `master`.
- Always open a pull request for changes targeting `master`.
- AI assistant branches must follow the pattern `claude/<task-id>`.

---

## Development Conventions (to be refined as the project takes shape)

### Code Style

- Agree on a formatter and linter before writing the first source file (e.g. `black`/`ruff` for Python, `prettier`/`eslint` for JavaScript/TypeScript).
- Enforce style via pre-commit hooks so it is never a PR concern.

### Testing

- Tests live alongside source code or in a top-level `tests/` directory.
- Every non-trivial function should have at least one unit test.
- PRs must not reduce overall test coverage.
- Add the test command here once the stack is chosen (e.g. `pytest`, `npm test`, `cargo test`).

### Documentation

- Keep `README.md` up to date with setup, run, and test instructions.
- Update this `CLAUDE.md` whenever project structure, tooling, or conventions change.

---

## For AI Assistants

### Before Making Changes

1. Read this file and any referenced documentation.
2. Run existing tests (once they exist) to confirm a clean baseline.
3. Understand the task scope — avoid changes beyond what is explicitly requested.

### While Working

- Prefer editing existing files over creating new ones.
- Keep commits small and focused; one logical change per commit.
- Do not introduce new dependencies without noting them in the PR description.
- Do not add comments or docstrings to code you did not modify.
- Remove unused code rather than commenting it out.

### When the Project Stack Is Decided

Update the following sections with concrete commands:

```markdown
## Quick Start
# install dependencies
# run the application
# run tests
# lint / format
```

### Avoid

- Backwards-compatibility shims for code that no longer exists.
- Feature flags or extra configuration for hypothetical future requirements.
- Over-engineered abstractions for one-off operations.
- Security anti-patterns: command injection, SQL injection, hard-coded secrets, XSS.

---

## Current Status

This repository is at the very beginning of its lifecycle. The immediate next steps for any contributor (human or AI) are:

1. Define the project's purpose and technology stack.
2. Update `README.md` with that information.
3. Add a `.gitignore` appropriate for the chosen stack.
4. Set up a linter/formatter and pre-commit hooks.
5. Create an initial project skeleton with at least one passing test.
6. Return here and fill in the blanks in this `CLAUDE.md`.
