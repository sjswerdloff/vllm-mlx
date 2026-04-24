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

## After Upstream Merge

Run kindled regression tests to catch overwritten fixes:
```bash
pytest tests/test_kindled_regressions.py -v
```

These guard: hybrid cache eval (OOM fix), text-only MLLM processor skip, Nemotron/JSON tool parsing, prefix cache for vision history.

## Kindled-Specific Code

Our patches on top of upstream:
- **Hybrid cache eval** in `mllm_batch_generator.py` — evals both KVCache (.keys/.values) and ArraysCache (.state) between chunks
- **Text-only MLLM guard** in `engine/batched.py` — `num_images > 0` check skips processor for text-only requests
- **Nemotron/JSON hybrid parsing** in `tool_parsers/hermes_tool_parser.py` — JSON fallback when no `<parameter>` tags found
- **Anthropic vision test skip** in `tests/test_anthropic_vision_tools.py` — `pytest.importorskip` + server connectivity check

## Commit Co-Authorship

```
Co-Authored-By: clement-7074f29f <clement-7074f29f@sjstargetedsolutions.co.nz>
```
