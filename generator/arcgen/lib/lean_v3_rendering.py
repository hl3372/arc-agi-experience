"""PNG rendering and deterministic anonymized human-review materials for V3."""

from __future__ import annotations

from pathlib import Path
from random import Random
from string import ascii_lowercase, digits
from typing import Any, Sequence
import json

from PIL import Image, ImageDraw, ImageFont

from .util import atomic_write_json, atomic_write_text, content_hash, read_json


ARC_COLORS = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default()


def _cell_size(task: dict[str, Any]) -> int:
    largest = max(
        max(len(pair[field]), len(pair[field][0]))
        for split in ("train", "test")
        for pair in task[split]
        for field in ("input", "output")
    )
    return max(5, min(18, 360 // largest))


def _wrap_text(
    text: str, font: ImageFont.ImageFont, maximum_width: int
) -> list[str]:
    """Wrap prose by rendered pixel width without truncating the rule."""

    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.getlength(candidate) <= maximum_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    grid: Sequence[Sequence[int]],
    x: int,
    y: int,
    cell: int,
) -> None:
    for row_index, row in enumerate(grid):
        for column_index, color in enumerate(row):
            left = x + column_index * cell
            top = y + row_index * cell
            draw.rectangle(
                (left, top, left + cell - 1, top + cell - 1),
                fill=ARC_COLORS[int(color)],
                outline="#454545" if cell >= 7 else ARC_COLORS[int(color)],
                width=1,
            )
    draw.rectangle(
        (
            x - 1,
            y - 1,
            x + len(grid[0]) * cell,
            y + len(grid) * cell,
        ),
        outline="#111111",
        width=2,
    )


def render_episode(
    task: dict[str, Any],
    *,
    title: str,
    subtitle: str,
    rule: str | None = None,
    destination: str | Path,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cell = _cell_size(task)
    margin = 28
    arrow_space = 72
    base_heading_height = 76
    label_height = 42
    section_gap = 26
    sections = [
        (f"TRAIN {index}", pair)
        for index, pair in enumerate(task["train"], 1)
    ] + [
        (f"TEST {index}", pair)
        for index, pair in enumerate(task["test"], 1)
    ]
    widths: list[int] = []
    heights: list[int] = []
    for _, pair in sections:
        widths.append(
            margin * 2
            + len(pair["input"][0]) * cell
            + arrow_space
            + len(pair["output"][0]) * cell
        )
        heights.append(
            label_height
            + max(len(pair["input"]), len(pair["output"])) * cell
            + section_gap
        )
    canvas_width = max(max(widths), 620)
    rule_font = _font(13)
    rule_lines = (
        _wrap_text(rule, rule_font, canvas_width - margin * 2 - 22)
        if rule
        else []
    )
    rule_line_height = 19
    rule_box_height = 0 if not rule_lines else 42 + len(rule_lines) * rule_line_height
    heading_height = base_heading_height + rule_box_height
    canvas = Image.new(
        "RGB",
        (canvas_width, heading_height + sum(heights) + margin),
        "#F7F7F5",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), title, fill="#111111", font=_font(23, bold=True))
    draw.text((margin, 46), subtitle, fill="#555555", font=_font(13))
    if rule_lines:
        box_top = base_heading_height
        box_bottom = box_top + rule_box_height - 12
        draw.rounded_rectangle(
            (margin, box_top, canvas_width - margin, box_bottom),
            radius=8,
            fill="#FFFFFF",
            outline="#C8C8C4",
            width=1,
        )
        draw.text(
            (margin + 11, box_top + 9),
            "INTENDED RULE",
            fill="#333333",
            font=_font(12, bold=True),
        )
        text_y = box_top + 29
        for line in rule_lines:
            draw.text(
                (margin + 11, text_y),
                line,
                fill="#242424",
                font=rule_font,
            )
            text_y += rule_line_height
    y = heading_height
    for section_name, pair in sections:
        input_grid = pair["input"]
        output_grid = pair["output"]
        input_width = len(input_grid[0]) * cell
        input_height = len(input_grid) * cell
        output_height = len(output_grid) * cell
        draw.text((margin, y), section_name, fill="#222222", font=_font(15, bold=True))
        grid_y = y + label_height
        draw.text((margin, grid_y - 18), "INPUT", fill="#555555", font=_font(12, bold=True))
        output_x = margin + input_width + arrow_space
        draw.text(
            (output_x, grid_y - 18),
            "OUTPUT" if section_name.startswith("TRAIN") else "EXPECTED OUTPUT",
            fill="#555555",
            font=_font(12, bold=True),
        )
        _draw_grid(draw, input_grid, margin, grid_y, cell)
        draw.text(
            (
                margin + input_width + 23,
                grid_y + max(input_height, output_height) // 2 - 12,
            ),
            "→",
            fill="#222222",
            font=_font(22, bold=True),
        )
        _draw_grid(draw, output_grid, output_x, grid_y, cell)
        y += label_height + max(input_height, output_height) + section_gap
    canvas.save(destination, format="PNG", optimize=True)
    return destination


def build_contact_sheet(
    images: Sequence[tuple[str, Path]],
    destination: str | Path,
    *,
    columns: int | None = None,
) -> Path | None:
    if not images:
        return None
    tiles: list[tuple[str, Image.Image]] = []
    for label, path in images:
        image = Image.open(path).convert("RGB")
        image.thumbnail((760, 720), Image.Resampling.LANCZOS)
        tiles.append((label, image.copy()))
    margin = 24
    gap = 24
    label_height = 30
    if columns is None:
        columns = 2 if len(tiles) > 1 else 1
    elif columns < 1:
        raise ValueError("columns must be at least 1")
    columns = min(columns, len(tiles))
    tile_width = max(image.width for _, image in tiles)
    rows = (len(tiles) + columns - 1) // columns
    row_heights = []
    for row in range(rows):
        row_heights.append(
            max(
                tiles[index][1].height
                for index in range(row * columns, min((row + 1) * columns, len(tiles)))
            )
        )
    sheet = Image.new(
        "RGB",
        (
            margin * 2 + columns * tile_width + (columns - 1) * gap,
            margin * 2 + sum(row_heights) + rows * label_height + (rows - 1) * gap,
        ),
        "#EFEFED",
    )
    draw = ImageDraw.Draw(sheet)
    y = margin
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if index >= len(tiles):
                continue
            label, image = tiles[index]
            x = margin + column * (tile_width + gap)
            draw.text((x, y), label, fill="#111111", font=_font(17, bold=True))
            sheet.paste(image, (x, y + label_height))
        y += label_height + row_heights[row] + gap
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", optimize=True)
    return destination


def render_run_tasks(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    state = read_json(run_dir / "state.json")
    destination = run_dir / "rendered_tasks"
    destination.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, Path]] = []
    manifest: list[dict[str, Any]] = []
    records = sorted(
        (
            record
            for record in state["records"].values()
            if record.get("complete_task")
        ),
        key=lambda item: (
            item.get("editorial_brief", {}).get("development_priority", 99),
            item["candidate_id"],
        ),
    )
    for index, record in enumerate(records, 1):
        filename = f"{index:02d}_{record['candidate_id']}.png"
        path = render_episode(
            record["complete_task"],
            title=f"{record['candidate_id']} — {record['status']}",
            subtitle=(
                f"{record['search_mode']} · "
                f"{record.get('portfolio_placement_status', 'not yet placed')}"
            ),
            rule=record.get("architecture", {}).get("intended_rule"),
            destination=destination / filename,
        )
        rendered.append((record["candidate_id"], path))
        manifest.append(
            {
                "candidate_id": record["candidate_id"],
                "intrinsic_quality_status": record.get("intrinsic_quality_status"),
                "portfolio_placement_status": record.get(
                    "portfolio_placement_status"
                ),
                "intended_rule_included": bool(
                    record.get("architecture", {}).get("intended_rule")
                ),
                "png": filename,
                "task_json": str(
                    (run_dir / "tasks" / f"{record['candidate_id']}.arc.json").resolve()
                ),
            }
        )
    contact = build_contact_sheet(rendered, destination / "contact_sheet.png")
    result = {
        "task_count": len(manifest),
        "contact_sheet": str(contact.resolve()) if contact else None,
        "tasks": manifest,
    }
    atomic_write_json(destination / "manifest.json", result)
    return result


def _anonymous_id(rng: Random, used: set[str]) -> str:
    alphabet = ascii_lowercase + digits
    while True:
        value = "".join(rng.choice(alphabet) for _ in range(12))
        if value not in used:
            used.add(value)
            return value


def _v1_core_sources(repository_root: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    directories = (
        repository_root / "runs" / "lean-authoring-v1-real-20260728" / "exports" / "core",
        repository_root
        / "runs"
        / "lean-authoring-v1-continuation-20260729"
        / "exports"
        / "core",
    )
    for directory in directories:
        for task_path in sorted(directory.glob("*.json")):
            authoring_candidates = (
                task_path.parents[2] / "tasks" / f"{task_path.stem}.authoring.json",
                task_path.parents[2]
                / "authoring"
                / "core"
                / f"{task_path.stem}.authoring.json",
            )
            authoring = next(
                (
                    read_json(path)
                    for path in authoring_candidates
                    if path.is_file()
                ),
                {},
            )
            architecture = authoring.get("architecture_contract", {})
            sources.append(
                {
                    "source_id": task_path.stem,
                    "task": read_json(task_path),
                    "rule": architecture.get("intended_rule")
                    or authoring.get("authoring_metadata", {}).get("intended_rule", ""),
                    "source_path": str(task_path.resolve()),
                }
            )
    return sources


def build_anonymized_review_pack(
    run_dir: str | Path,
    *,
    repository_root: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Mix every new realization with a small deterministic V1 core sample."""

    run_dir = Path(run_dir).resolve()
    root = Path(repository_root).resolve()
    state = read_json(run_dir / "state.json")
    new = [
        {
            "provenance": "v3",
            "source_id": record["candidate_id"],
            "task": record["complete_task"],
            "rule": record.get("architecture", {}).get("intended_rule", ""),
            "source_path": str(
                (run_dir / "tasks" / f"{record['candidate_id']}.arc.json").resolve()
            ),
        }
        for record in state["records"].values()
        if record.get("complete_task")
    ]
    rng = Random(seed + 8300)
    v1 = _v1_core_sources(root)
    v1_sample = rng.sample(v1, min(4, len(new), len(v1)))
    sources = [*new, *({**item, "provenance": "v1"} for item in v1_sample)]
    rng.shuffle(sources)

    destination = run_dir / "anonymized_review"
    tasks_dir = destination / "tasks"
    images_dir = destination / "images"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    public_manifest: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    rendered: list[tuple[str, Path]] = []
    for item in sources:
        anonymous = _anonymous_id(rng, used)
        atomic_write_json(tasks_dir / f"{anonymous}.json", item["task"])
        image = render_episode(
            item["task"],
            title=f"Anonymous ARC episode {anonymous}",
            subtitle="Provenance and author rule withheld",
            destination=images_dir / f"{anonymous}.png",
        )
        rendered.append((anonymous, image))
        public_manifest.append(
            {
                "anonymous_id": anonymous,
                "task_json": f"tasks/{anonymous}.json",
                "image": f"images/{anonymous}.png",
                "task_hash": content_hash(item["task"]),
            }
        )
        answer_key.append(
            {
                "anonymous_id": anonymous,
                "provenance": item["provenance"],
                "source_id": item["source_id"],
                "source_path": item["source_path"],
                "intended_rule": item["rule"],
            }
        )
    contact = build_contact_sheet(rendered, destination / "review_sheet.png")
    public_manifest.sort(key=lambda item: item["anonymous_id"])
    answer_key.sort(key=lambda item: item["anonymous_id"])
    manifest = {
        "schema_version": 1,
        "purpose": (
            "Human comparison of conceptual interest, visual economy, functional "
            "richness, discoverability, and overall preference."
        ),
        "provenance_hidden": True,
        "rules_withheld_from_first_page": True,
        "task_count": len(public_manifest),
        "new_v3_count": len(new),
        "v1_reference_count": len(v1_sample),
        "contact_sheet": "review_sheet.png" if contact else None,
        "tasks": public_manifest,
    }
    atomic_write_json(destination / "manifest.json", manifest)
    atomic_write_json(
        destination / "DO_NOT_OPEN_FIRST.answer_key.json",
        {"warning": "Open only after recording the blind review.", "answers": answer_key},
    )
    lines = [
        "# Anonymous V3 versus V1 ARC review",
        "",
        "Review the contact sheet and anonymous task JSON before opening the appendix.",
        "No author rule or provenance appears on this first page.",
        "",
        "Rate conceptual interest, visual economy, functional richness, discoverability,",
        "and overall preference. This packet is not used automatically by the pipeline.",
        "",
        f"Contact sheet: `review_sheet.png`",
        "",
        "| Anonymous ID | Task | Image |",
        "|---|---|---|",
    ]
    for item in public_manifest:
        lines.append(
            f"| `{item['anonymous_id']}` | `{item['task_json']}` | `{item['image']}` |"
        )
    atomic_write_text(destination / "review_sheet.md", "\n".join(lines) + "\n")
    appendix = [
        "# Answer and rule appendix",
        "",
        "This page reveals provenance. Open it only after the blind comparison.",
        "",
    ]
    for item in answer_key:
        appendix.extend(
            [
                f"## {item['anonymous_id']}",
                "",
                f"Provenance: `{item['provenance']}`  ",
                f"Source: `{item['source_id']}`",
                "",
                f"Intended rule: {item['intended_rule'] or 'Not available.'}",
                "",
            ]
        )
    atomic_write_text(destination / "answer_rule_appendix.md", "\n".join(appendix))
    return {
        **manifest,
        "directory": str(destination.resolve()),
        "answer_key": str(
            (destination / "DO_NOT_OPEN_FIRST.answer_key.json").resolve()
        ),
        "answer_rule_appendix": str(
            (destination / "answer_rule_appendix.md").resolve()
        ),
    }


__all__ = [
    "build_anonymized_review_pack",
    "build_contact_sheet",
    "render_episode",
    "render_run_tasks",
]
