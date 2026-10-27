# Canvas Agents — Thinking Variant

Five independent local-model agents take turns adding exactly one RGB pixel to a shared 256×256 canvas. They receive a compact preview of the current canvas, know that other agents may have acted, and are told only to help make the canvas beautiful—no subject, style, or composition is prescribed.

Each turn writes:

- an append-only trace under `traces/`
- a sparse canvas snapshot under `snapshots/`
- a replay record in `state/history.json`
- the current canvas in `state/canvas.json`
- a directly viewable current image in `canvas.svg`

The `docs/` viewer replays the history in a browser and is suitable for GitHub Pages.

Once Pages is enabled, the live viewer will be at `https://rohanxg0072.github.io/canvas-agents-thinking/`. GitHub also renders `canvas.svg` directly when you open it in the repository.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python canvas_agents.py --agents 5
python3 -m http.server 8000 --directory docs
```

Open <http://localhost:8000> and use the play controls.

The default model is `Qwen/Qwen3-0.6B` in thinking mode. Each trace stores the raw output, visible `<think>...</think>` reasoning block, and final answer. This variant runs hourly by default. Set `CANVAS_MODEL` to another Hugging Face Transformers-compatible causal model if desired. The first run downloads model weights; later GitHub Actions runs restore the model cache.

Agent output passes through a fuzzy normalizer that prefers the valid candidate requiring the fewest repairs, then uses lexical similarity as a tie-breaker. It handles prose/fences, quote and key normalization, trailing commas, and loose `x`, `y`, `rgb` formats before using the safe fallback.

For a quick repository-only test without model weights:

```bash
python canvas_agents.py --agents 5 --dry-run
```

## GitHub Actions

`.github/workflows/agents.yml` runs five sequential turns every hour and can also be started manually. The five turns are sequential so two agents cannot overwrite each other’s state. The workflow commits traces, snapshots, state, and viewer data back to the default branch. `.github/workflows/watchdog.yml` runs hourly at a different minute and dispatches a recovery run if no successful canvas run has completed in three hours.

After creating the GitHub repository, enable Pages from the `docs/` folder if you want a hosted replay viewer.

## Create the remote repository

From this directory, after refreshing GitHub CLI authentication if needed:

```bash
gh auth login -h github.com
gh repo create canvas-agents-thinking --public --source=. --remote=origin --push
```

