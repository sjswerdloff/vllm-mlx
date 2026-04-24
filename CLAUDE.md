# CLAUDE.md — vllm-mlx (Kindled fork)

## Branch Workflow

**Never commit directly to `kindled-main`.** All changes go through feature branches and PRs:

1. Branch from `kindled-main` (e.g. `fix/description`, `feat/description`)
2. Push to `origin` (`sjswerdloff/vllm-mlx`)
3. Merge after CI passes on GitHub

For upstream PRs to `waybarrios/vllm-mlx`, branch from `origin/main` instead.

## Remotes

All clones use consistent naming:
- `origin` — `sjswerdloff/vllm-mlx` (our fork)
- `upstream` — `waybarrios/vllm-mlx` (upstream)

## Testing Kindled Changes

**Every kindled-specific change must have a test in a `test_kindled_*.py` file.** Never add kindled tests to upstream test files — upstream merges will overwrite them.

- `tests/test_kindled_regressions.py` — guards for fixes we carry on top of upstream
- Add new `tests/test_kindled_*.py` files as needed for new features

This ensures `pytest tests/test_kindled_*.py -v` catches any functionality lost after merging upstream.

## After Upstream Merge

1. Merge `origin/main` into `kindled-main` (on a branch, via PR)
2. Run kindled regression tests: `pytest tests/test_kindled_*.py -v`
3. If any fail, our patches were overwritten — restore before merging

**Do NOT lint-fix upstream files during merges.** Upstream's files pass upstream's CI. Running ruff auto-fix on their files removes imports they depend on elsewhere (e.g. `Dict`, `Optional` from `typing`) because the auto-fixer only sees the current merge state, not the full codebase usage. Only fix lint in files we created or modified.

## Kindled-Specific Code

Our patches on top of upstream:
- **Hybrid cache eval** in `mllm_batch_generator.py` — evals both KVCache (.keys/.values) and ArraysCache (.state) between chunks
- **Text-only MLLM guard** in `engine/batched.py` — `num_images > 0` check skips processor for text-only requests
- **Nemotron/JSON hybrid parsing** in `tool_parsers/hermes_tool_parser.py` — JSON fallback when no `<parameter>` tags found
- **Anthropic vision test skip** in `tests/test_anthropic_vision_tools.py` — `pytest.importorskip` + server connectivity check

## Pre-commit Hooks

**Never use `--no-verify`.** If mypy fails on pre-existing upstream issues, skip only mypy:

```bash
SKIP=mypy git commit -m "message"
```

This still runs ruff, ruff-format, trailing whitespace, merge conflict checks — everything except mypy.

## Commit Co-Authorship

```
Co-Authored-By: clement-7074f29f <clement-7074f29f@sjstargetedsolutions.co.nz>
```
