#!/usr/bin/env python3
"""Run independent tiny-model agents against a shared sparse 256x256 canvas."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

WIDTH = HEIGHT = 256
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
TRACE_DIR = ROOT / "traces"
SNAPSHOT_DIR = ROOT / "snapshots"
DOCS_DIR = ROOT / "docs"
MODEL_NAME = os.getenv("CANVAS_MODEL", "Qwen/Qwen3-0.6B")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dirs() -> None:
    for directory in (STATE_DIR, TRACE_DIR, SNAPSHOT_DIR, DOCS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_svg(canvas: dict, path: Path) -> None:
    rects = []
    for key, rgb in sorted(canvas["pixels"].items()):
        x, y = key.split(",")
        color = "#%02x%02x%02x" % tuple(rgb)
        rects.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{color}"/>')
    svg = "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="768" height="768" shape-rendering="crispEdges">',
        '<rect width="256" height="256" fill="#000000"/>',
        *rects,
        '</svg>',
        '',
    ])
    path.write_text(svg)


def load_canvas() -> dict[str, list[int]]:
    canvas = load_json(STATE_DIR / "canvas.json", {"width": WIDTH, "height": HEIGHT, "pixels": {}})
    canvas.setdefault("width", WIDTH)
    canvas.setdefault("height", HEIGHT)
    canvas.setdefault("pixels", {})
    return canvas


def canvas_hash(canvas: dict) -> str:
    payload = json.dumps(canvas, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def preview(canvas: dict, size: int = 16) -> str:
    """Render a compact 16x16 average-color preview for a text-only model."""
    pixels = canvas["pixels"]
    lines = []
    for py in range(size):
        row = []
        for px in range(size):
            r_total = g_total = b_total = count = 0
            x0, x1 = px * WIDTH // size, (px + 1) * WIDTH // size
            y0, y1 = py * HEIGHT // size, (py + 1) * HEIGHT // size
            for y in range(y0, y1):
                for x in range(x0, x1):
                    color = pixels.get(f"{x},{y}")
                    if color:
                        r_total += color[0]
                        g_total += color[1]
                        b_total += color[2]
                        count += 1
            row.append("#000000" if not count else "#%02x%02x%02x" % (r_total // count, g_total // count, b_total // count))
        lines.append(" ".join(row))
    return "\n".join(lines)


def occupancy_preview(canvas: dict, size: int = 16) -> str:
    """Show how many of the 16x16 full-resolution pixels are already painted."""
    pixels = canvas["pixels"]
    lines = []
    for py in range(size):
        row = []
        y0, y1 = py * HEIGHT // size, (py + 1) * HEIGHT // size
        for px in range(size):
            x0, x1 = px * WIDTH // size, (px + 1) * WIDTH // size
            count = sum(
                1
                for y in range(y0, y1)
                for x in range(x0, x1)
                if f"{x},{y}" in pixels
            )
            row.append("." if count == 0 else "1" if count < 16 else "2" if count < 64 else "3" if count < 256 else "F")
        lines.append("".join(row))
    return "\n".join(lines)


def prompt_for(agent_number: int, canvas: dict, history: list[dict]) -> str:
    recent = history[-8:]
    recent_text = "\n".join(
        f"step {item['step']}: agent {item['agent']} chose ({item['x']},{item['y']}) rgb={item['rgb']}"
        for item in recent
    ) or "(no previous turns)"
    return f"""You are agent {agent_number} in a shared pixel canvas experiment. Your only broad instruction is: help make the canvas beautiful.

Other agents may have already changed the canvas, and other agents will act after you. You are not told what the canvas is supposed to depict. Inspect the current preview and independently choose one place and color that help make it beautiful. Do not invent a subject or explain a drawing plan.

Current canvas preview, downsampled to 16x16 (#rrggbb; each row is one line). This is only a visual map, not the coordinate system:
{preview(canvas)}

Cell occupancy map (same 16x16 layout): . means empty, 1 means 1-15 pixels painted, 2 means 16-63, 3 means 64-255, F means full. Prefer a cell with room for another mark:
{occupancy_preview(canvas)}

The separate answer step asks for a preview-cell selection, so use this phase to quickly consider a cell and color. The occupancy map indicates which cells still have room.

Recent turns:
{recent_text}

The values in the shape below are placeholders; do not copy them by default. Make an independent choice based on the current preview.

This is the private reasoning phase. Analyze which preview cell and color would improve the canvas. Do not try to produce JSON in this phase; a separate short answer pass will convert your choice into the required object."""


def answer_prompt_for(agent_number: int, canvas: dict, history: list[dict]) -> str:
    """Build a compact answer prompt using simple 16x16 preview-cell coordinates."""
    recent = history[-12:]
    recent_text = ", ".join(f"({item['x']},{item['y']})" for item in recent) or "none"
    recent_colors = {
        tuple(item["rgb"])
        for item in history[-8:]
        if isinstance(item.get("rgb"), list) and len(item["rgb"]) == 3
    }
    recent_color_text = ", ".join("#%02x%02x%02x" % color for color in sorted(recent_colors)) or "none"
    return f"""You are agent {agent_number}. Choose one unpainted-looking location that helps make this shared canvas beautiful.

Choose a preview cell using x and y indices from 0 to 15. Code will place your mark at a free full-resolution pixel inside that cell. The preview has 16 columns and 16 rows.

PREVIEW:
{preview(canvas)}

OCCUPANCY (same grid): . empty, 1 lightly filled, 2 partly filled, 3 mostly filled, F full:
{occupancy_preview(canvas)}

Recently used exact full-canvas coordinates (avoid repeating their area): {recent_text}
Colors used in the most recent moves (prefer a different color when it fits): {recent_color_text}

Output one JSON object only. Use keys x and y for preview column and row (integers 0..15), and rgb as either three integers 0..255 or a six-digit hex color. No explanation or markdown."""


def parse_decision(text: str) -> tuple[int, int, list[int]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    matches = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
    candidates = []
    canonical = '{"x":0,"y":0,"rgb":[0,0,0]}'
    for candidate in matches or [cleaned]:
        repairs = 0
        attempts = [candidate]
        repaired = candidate.replace("“", '"').replace("”", '"').replace("'", '"')
        repaired = re.sub(r"([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        if repaired != candidate:
            repairs = 1
            attempts.append(repaired)
        value = None
        for attempt in attempts:
            try:
                value = json.loads(attempt)
                break
            except json.JSONDecodeError:
                pass
        if value is None:
            loose = re.search(
                r"x\s*[:=]\s*(-?\d+).*?y\s*[:=]\s*(-?\d+).*?rgb\s*[:=]\s*[\[(]\s*(-?\d+)\s*[,; ]\s*(-?\d+)\s*[,; ]\s*(-?\d+)",
                candidate,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if loose:
                value = {"x": int(loose.group(1)), "y": int(loose.group(2)), "rgb": [int(loose.group(3)), int(loose.group(4)), int(loose.group(5))]}
                repairs = 2
        if isinstance(value, dict) and all(key in value for key in ("x", "y", "rgb")):
            if isinstance(value["rgb"], str):
                hex_color = re.fullmatch(r"#?([0-9a-fA-F]{6})", value["rgb"].strip())
                if hex_color:
                    encoded = hex_color.group(1)
                    value = dict(value)
                    value["rgb"] = [int(encoded[index:index + 2], 16) for index in (0, 2, 4)]
            shape = json.dumps({"x": value["x"], "y": value["y"], "rgb": value["rgb"]}, separators=(",", ":"))
            similarity = difflib.SequenceMatcher(None, shape, canonical).ratio()
            candidates.append((repairs, -similarity, value))
    if not candidates:
        raise ValueError("model did not contain a repairable pixel decision")
    value = min(candidates, key=lambda item: (item[0], item[1]))[2]
    x, y = int(value["x"]), int(value["y"])
    rgb = [int(channel) for channel in value["rgb"]]
    if not (0 <= x < WIDTH and 0 <= y < HEIGHT and len(rgb) == 3 and all(0 <= n <= 255 for n in rgb)):
        raise ValueError("decision is outside the canvas or RGB range")
    return x, y, rgb


def canonicalize_decision(
    canvas: dict,
    x: int,
    y: int,
    rgb: list[int],
    history: list[dict],
) -> tuple[int, int, list[int]]:
    """Map a preview cell to a free pixel, spreading recent moves within that cell."""
    if not (0 <= x < 16 and 0 <= y < 16):
        # Be forgiving if the model still returns full-canvas coordinates.
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            x, y = x // 16, y // 16
        else:
            raise ValueError("preview-cell coordinate is outside the 16x16 grid")

    x0, y0 = x * 16, y * 16
    recent_points = [
        (item["x"], item["y"])
        for item in history[-24:]
        if x0 <= item.get("x", -1) < x0 + 16 and y0 <= item.get("y", -1) < y0 + 16
    ]
    candidates = [
        (x0 + dx, y0 + dy)
        for dy in range(16)
        for dx in range(16)
        if f"{x0 + dx},{y0 + dy}" not in canvas["pixels"]
    ]
    if not candidates:
        raise ValueError("the model-selected preview cell is full")

    def nearest_recent_distance(point: tuple[int, int]) -> int:
        if not recent_points:
            return 32
        return min(max(abs(point[0] - old_x), abs(point[1] - old_y)) for old_x, old_y in recent_points)

    spaced = [point for point in candidates if nearest_recent_distance(point) >= 8]
    pool = spaced or [point for point in candidates if nearest_recent_distance(point) >= 4] or candidates
    center = (x0 + 8, y0 + 8)
    chosen_x, chosen_y = min(
        pool,
        key=lambda point: (-nearest_recent_distance(point), abs(point[0] - center[0]) + abs(point[1] - center[1])),
    )
    return chosen_x, chosen_y, rgb


class TinyModel:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.tokenizer = None
        self.model = None
        if not dry_run:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")
            self.model.eval()

    def ask(self, prompt: str) -> dict[str, str]:
        if self.dry_run:
            return {"raw_output": "", "reasoning": "", "answer": ""}
        import torch

        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = self.tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        think_match = re.search(r"<think>(.*?)(?:</think>|$)", raw_output, flags=re.DOTALL | re.IGNORECASE)
        reasoning = think_match.group(1).strip() if think_match else ""
        answer = (
            raw_output[:think_match.start()] + raw_output[think_match.end():]
            if think_match
            else raw_output
        ).strip()
        return {"raw_output": raw_output, "reasoning": reasoning, "answer": answer}

    def ask_json_only(self, prompt: str) -> str:
        """Run the compact answer pass with thinking disabled."""
        if self.dry_run:
            return ""
        import torch

        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def fallback_decision(canvas: dict, agent_number: int) -> tuple[int, int, list[int]]:
    seed = f"{canvas_hash(canvas)}:{agent_number}:{now()}"
    rng = random.Random(seed)
    return rng.randrange(WIDTH), rng.randrange(HEIGHT), [rng.randrange(256) for _ in range(3)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    canvas = load_canvas()
    history = load_json(STATE_DIR / "history.json", [])
    model = TinyModel(args.dry_run)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for agent_number in range(1, args.agents + 1):
        prompt = prompt_for(agent_number, canvas, history)
        answer_prompt = answer_prompt_for(agent_number, canvas, history)
        response = model.ask(prompt)
        raw_output = response["raw_output"]
        reasoning_trace = response["reasoning"]
        answer_output = model.ask_json_only(answer_prompt)
        repair_output = ""
        source = "model-answer"
        try:
            x, y, rgb = parse_decision(answer_output)
            x, y, rgb = canonicalize_decision(canvas, x, y, rgb, history)
        except Exception as exc:
            x, y, rgb = fallback_decision(canvas, agent_number)
            source = f"fallback: {exc}"

        step = len(history) + 1
        key = f"{x},{y}"
        previous = canvas["pixels"].get(key)
        canvas["pixels"][key] = rgb
        entry = {
            "step": step,
            "run": run_id,
            "timestamp": now(),
            "agent": agent_number,
            "x": x,
            "y": y,
            "rgb": rgb,
            "previous_rgb": previous,
            "source": source,
            "canvas_hash": canvas_hash(canvas),
        }
        trace = {
            "entry": entry,
            "model": MODEL_NAME,
            "prompt": prompt,
            "answer_prompt": answer_prompt,
            "raw_output": raw_output,
            "reasoning_trace": reasoning_trace,
            "answer_output": answer_output,
            "repair_output": repair_output,
        }
        save_json(TRACE_DIR / f"{run_id}-step-{step:06d}-agent-{agent_number}.json", trace)
        snapshot = {"width": WIDTH, "height": HEIGHT, "step": step, "pixels": canvas["pixels"]}
        save_json(SNAPSHOT_DIR / f"step-{step:06d}.json", snapshot)
        history.append(entry)
        save_json(STATE_DIR / "canvas.json", canvas)
        save_json(STATE_DIR / "history.json", history)
        save_json(DOCS_DIR / "history.json", history)
        save_svg(canvas, ROOT / "canvas.svg")
        save_svg(canvas, DOCS_DIR / "canvas.svg")
        print(json.dumps(entry, sort_keys=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
