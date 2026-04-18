"""
process_trajs.py

This script filters and merges rollout trajectories collected by
`AndroidWorld/run_diy.py`.

- keeps successful trajectories only
- trims repeated trailing answer actions
- attaches image paths and click bboxes to each step
- writes a merged trajectory JSON under the rollout directory

- Input: a rollout directory produced by `AndroidWorld/run_diy.py`
- Output: a merged JSON file such as `data_merge_success.json` under the rollout directory
- Next stage: the merged file will be used to convert to ShareGPT-style training data

Example:
  python AndroidWorld/process_trajs.py \
    --runs_dir AndroidWorld/runs/rollout \
    --output-name data_merge.json
"""

from __future__ import annotations
import argparse
import json
import re
import os
import hashlib
from pathlib import Path
from PIL import Image, ImageChops
from tqdm import tqdm


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _find_result_json(traj_dir: Path) -> Path | None:
    # Support both naming conventions: result.json / results.json.
    for name in ("result.json", "results.json"):
        p = traj_dir / name
        if p.exists():
            return p
    return None


def _iter_tool_calls(response_text: str):
    for m in TOOL_CALL_RE.finditer(response_text or ""):
        try:
            yield json.loads(m.group(1))
        except Exception:
            continue


def _load_metadata_json(traj_dir: Path) -> dict:
    p = traj_dir / "metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"metadata.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _metadata_steps_map(metadata: dict) -> dict[int, dict]:
    steps = (metadata or {}).get("steps") or []
    out: dict[int, dict] = {}
    for s in steps:
        try:
            out[int(s.get("step"))] = s
        except Exception:
            continue
    return out


def _extract_mobile_use_action_and_coord(response_text: str) -> tuple[str | None, list[float] | None]:
    """Extract the action and coordinate from a <tool_call> block."""
    for call in _iter_tool_calls(response_text or ""):
        if (call or {}).get("name") != "mobile_use":
            continue
        args = (call or {}).get("arguments") or {}
        action = args.get("action")
        coord = args.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            try:
                return str(action) if action is not None else None, [float(coord[0]), float(coord[1])]
            except Exception:
                return str(action) if action is not None else None, None
        return str(action) if action is not None else None, None
    return None, None


def _images_identical_pixels(p1: Path, p2: Path) -> bool:
    try:
        with Image.open(p1) as im1, Image.open(p2) as im2:
            im1 = im1.convert("RGB")
            im2 = im2.convert("RGB")
            if im1.size != im2.size:
                return False
            return ImageChops.difference(im1, im2).getbbox() is None
    except Exception:
        raise SystemExit(f"error in images_identical_pixels: {p1} or {p2}")


def _coord_to_logical_pixels(coord: list[float], logical_w: int, logical_h: int) -> tuple[int, int]:
    """Map tool-call coordinates to metadata logical_screen_size pixels.

    - If the coordinates look like 0-1000 normalized values, map them into [0, W/H).
    - Otherwise treat them as pixel coordinates and only round + clamp them.
    """
    if not coord or len(coord) != 2:
        return (0, 0)
    x, y = float(coord[0]), float(coord[1])

    # Heuristic: 0-1000 normalized coords.
    if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0 and logical_w > 0 and logical_h > 0:
        xp = int(round(x / 1000.0 * float(logical_w)))
        yp = int(round(y / 1000.0 * float(logical_h)))
    else:
        xp = int(round(x))
        yp = int(round(y))

    # clamp to screen
    if logical_w > 0:
        xp = max(0, min(logical_w - 1, xp))
    else:
        xp = 0
    if logical_h > 0:
        yp = max(0, min(logical_h - 1, yp))
    else:
        yp = 0
    return (xp, yp)


def _pick_min_area_bbox(ui_elements: list[dict], x: int, y: int) -> dict | None:
    """Select the smallest bbox in `ui_elements` that contains point `(x, y)`."""
    best = None
    best_area = None
    for el in ui_elements or []:
        bbox = (el or {}).get("bbox_pixels")
        if not isinstance(bbox, dict):
            continue
        try:
            x_min = int(bbox.get("x_min"))
            x_max = int(bbox.get("x_max"))
            y_min = int(bbox.get("y_min"))
            y_max = int(bbox.get("y_max"))
        except Exception:
            continue
        if x_max <= x_min or y_max <= y_min:
            continue
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        area = (x_max - x_min) * (y_max - y_min)
        if best_area is None or area < best_area:
            best_area = area
            best = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    return best


def _is_success_response(response_text: str) -> bool:
    """Treat `terminate(status=success)` or `answer` as success."""
    for call in _iter_tool_calls(response_text):
        if (call or {}).get("name") != "mobile_use":
            continue
        args = (call or {}).get("arguments") or {}
        action = args.get("action")
        if action == "answer":
            return True
        if action == "terminate" and args.get("status") == "success":
            return True
    return False


def _is_answer_step(step_obj: dict) -> bool:
    for call in _iter_tool_calls(step_obj.get("response", "")):
        if (call or {}).get("name") == "mobile_use" and ((call.get("arguments") or {}).get("action") == "answer"):
            return True
    return False


def _is_success_terminal(result_path: Path) -> bool:
    """Check whether the last step in result(s).json is successful."""
    data = json.loads(result_path.read_text(encoding="utf-8"))
    traj = data.get("trajectory") or []
    if not traj:
        return False

    return _is_success_response(traj[-1].get("response", ""))


def _trim_answer_tail(traj: list[dict]) -> list[dict]:
    """If the trajectory ends with consecutive answer steps, keep only the first one."""
    if len(traj) < 2 or not _is_answer_step(traj[-1]):
        return traj

    i = len(traj) - 1
    while i >= 0 and _is_answer_step(traj[i]):
        i -= 1
    first_answer_idx = i + 1  # Start of the trailing answer segment.
    return traj[: first_answer_idx + 1]


def _image_relpath(traj_dir: Path, step_id: int) -> str:
    # Do not add existence or suffix fallbacks here; screenshots are expected to be PNGs.
    return f"{traj_dir.name}/screenshot_step{step_id}.png"


def _with_image_keys(traj_dir: Path, data: dict, metadata: dict, base_dir=None) -> dict:
    traj = data.get("trajectory") or []
    traj = _trim_answer_tail(traj)
    meta_map = _metadata_steps_map(metadata)
    new_traj: list[dict] = []
    for idx, step_obj in enumerate(traj):
        step_id = step_obj.get("step", idx)
        step_id = int(step_id)
        s = dict(step_obj)
        s["image"] = _image_relpath(traj_dir, step_id)
        # Only populate bbox for click/long_press actions (including left_click).
        action, coord = _extract_mobile_use_action_and_coord(step_obj.get("response", ""))
        bbox = None
        if action in {"click", "left_click", "long_press"} and coord is not None:
            meta_step = meta_map.get(step_id)
            if not meta_step:
                raise SystemExit(f"metadata step not found: {traj_dir}/metadata.json step={step_id}")
            logical = meta_step.get("logical_screen_size") or []
            if not (isinstance(logical, (list, tuple)) and len(logical) == 2):
                raise SystemExit(f"bad logical_screen_size in metadata: {traj_dir}/metadata.json step={step_id}")
            logical_w, logical_h = int(logical[0]), int(logical[1])
            x, y = _coord_to_logical_pixels(coord, logical_w, logical_h)
            bbox = _pick_min_area_bbox(meta_step.get("ui_elements") or [], x, y)
        s["bbox"] = bbox
        if base_dir:
            if not os.path.exists(os.path.join(base_dir, s["image"])):
                print(f"image not found: {s['image']}")
        new_traj.append(s)

    # Mark a step as invalid if the next screenshot is pixel-identical.
    # Exception: keep it valid when the next action is `type` (e.g. focusing an input box).
    # This only applies when both screenshots exist and can be compared pixel-wise.
    base_path = Path(base_dir) if base_dir else None
    for i in range(len(new_traj)):
        new_traj[i]["is_valid"] = True
    for i in range(len(new_traj) - 1):
        cur = new_traj[i]
        nxt = new_traj[i + 1]
        if not base_path:
            continue
        p1 = base_path / str(cur.get("image", ""))
        p2 = base_path / str(nxt.get("image", ""))
        if not (p1.exists() and p2.exists()):
            raise SystemExit(f"image not found: {p1} or {p2}")
        if _images_identical_pixels(p1, p2):
            next_action, _ = _extract_mobile_use_action_and_coord(nxt.get("response", ""))
            if next_action != "type":
                cur["is_valid"] = False
    out = dict(data)
    # Keep only the subdirectory name in merged data, not the original absolute path.
    out["save_dir"] = traj_dir.name
    out["trajectory"] = new_traj
    return out


def _count_per_app(merged: list[dict], data_meta_path: Path) -> None:
    data_meta = json.loads(data_meta_path.read_text(encoding="utf-8"))
    dir_2_app = {}
    for item in data_meta:
        app_name = item["app"]
        dir_name = str(item["sample_id"]) + "_" + item["base_task_name"]
        dir_2_app[dir_name] = app_name

    app_2_num = {}
    missing_dirs: list[str] = []
    for item in merged:
        dir_name = item["save_dir"]
        app_name = dir_2_app.get(dir_name)
        if app_name is None:
            missing_dirs.append(dir_name)
            continue
        app_2_num[app_name] = app_2_num.get(app_name, 0) + 1

    print("Num per app:")
    for app, num in sorted(app_2_num.items()):
        print(f"{app}: {num}")
    if missing_dirs:
        print(f"warning: {len(missing_dirs)} merged trajectories missing app mapping in {data_meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Process AndroidWorld trajectories and count successes.")
    parser.add_argument(
        "--runs_dir",
        type=Path,
        required=True,
        help="Root runs directory where each subdirectory stores one trajectory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="data_merge_success.json",
        help="Output filename to be saved under runs_dir.",
    )
    parser.add_argument(
        "--data-meta",
        type=Path,
        default=None,
        help="Optional task metadata JSON used to count trajectories per app.",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    if not runs_dir.exists():
        raise SystemExit(f"runs_dir does not exist: {runs_dir}")

    total_dirs = 0
    has_result = 0
    terminate_success = 0
    merged: list[dict] = []

    for traj_dir in tqdm(sorted(p for p in runs_dir.iterdir() if p.is_dir())):
        total_dirs += 1
        result_path = _find_result_json(traj_dir)
        if not result_path:
            continue
        has_result += 1
        try:
            if _is_success_terminal(result_path):
                terminate_success += 1
                data = json.loads(result_path.read_text(encoding="utf-8"))
                metadata = _load_metadata_json(traj_dir)  # Successful runs must include metadata.json.
                merged.append(_with_image_keys(traj_dir, data, metadata, base_dir=runs_dir))
        except FileNotFoundError as e:
            # A successful trajectory without metadata.json is treated as a hard error.
            raise SystemExit(str(e))
        except Exception:
            # Treat parse failures as non-success while still counting the result file.
            print(f"parse failed: {result_path}")
            continue

    print(f"runs_dir: {runs_dir}")
    print(f"total_traj_dirs: {total_dirs}")
    print(f"with_result_json: {has_result}")
    print(f"agent_success(terminate_success_or_answer): {terminate_success}")

    out_path = runs_dir / args.output_name
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path} (items={len(merged)})")

    # Count total and average numbers of steps.
    total_steps = 0
    for item in merged:
        total_steps += len(item["trajectory"])
    print(f"total_steps: {total_steps}")
    print(f"average_steps: {total_steps / len(merged):.2f}")
    # Count the number of steps with is_valid=False.
    invalid_steps = 0
    for item in merged:
        for step in item["trajectory"]:
            if step.get("is_valid", True) is False:
                invalid_steps += 1
    print(f"invalid_steps: {invalid_steps}")

    # Count trajectories per app when metadata is provided.
    if args.data_meta is not None:
        _count_per_app(merged, args.data_meta)

    # Count weak/strong policy steps.
    weak_steps = 0
    strong_steps = 0
    unknown_policy_steps = 0
    for item in merged:
        for step in item.get("trajectory", []):
            src = step.get("policy_source")
            if src == "weak":
                weak_steps += 1
            elif src == "strong":
                strong_steps += 1
            else:
                unknown_policy_steps += 1

    print(f"weak_steps: {weak_steps}")
    print(f"strong_steps: {strong_steps}")
    if unknown_policy_steps:
        print(f"unknown_policy_steps: {unknown_policy_steps}")
    

if __name__ == "__main__":
    main()

