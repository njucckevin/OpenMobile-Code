"""
process_refine.py

This is an optional post-processing script for step-level trajectory text. Used to adapt the response format for Qwen2.5/3VL series.

- reads a merged trajectory JSON
- uses a strong multimodal model to add or rewrite either:
  - `conclusion`
  - `thinking_refine`

- Input: a merged JSON produced by `AndroidWorld/process_trajs.py`
- Output: a refined JSON with extra text fields
- Next stage: the refined file will be used to convert to ShareGPT-style training data

Example:
  python AndroidWorld/process_refine.py \
    --input AndroidWorld/runs/rollout/data_merge.json \
    --output AndroidWorld/runs/rollout/data_merge_conclusion.json \
    --mode conclusion \
    --base-url "$OPENAI_BASE_URL" \
    --api-key "$OPENAI_API_KEY"
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import os
import re
import threading
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw
from tqdm import tqdm


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINKING_RE = re.compile(r"<thinking>\s*([\s\S]*?)\s*</thinking>", re.I)

CONCLUSION_SYSTEM_PROMPT = """You are a high-quality GUI trajectory annotator.

Task:
- You will receive one Android GUI step with:
  1) user instruction (overall goal),
  2) step history text,
  3) the model response (maybe include thinking and conclusion) for the current step,
  4) current screenshot (possibly with a red marker if the step is click/long_press).
- Your task is to write or rewrite a concise conclusion of the action executed in the current step.

Output rule:
- The conclusion should summarize this step operation.
- Write 1-2 English sentences as <conclusion> for this step.
- For simple thinking, directly describe what was done in this step.
- For complex thinking (e.g., reflection after a wrong previous move, or trying a new strategy), describe the cause and effect around this step.
- If the thinking contains task-instruction-related information that should be remembered for later steps, include it in the conclusion.
- The conclusion should summarize the intended action and immediate progress/result for this step.
- Some model responses may contain something like "Based on the intervention context", but you should assume you are generating the conclusion without any external intervention.
- Keep it factual and grounded in the provided text + screenshot.
- Output plain conclusion text only (no <conclusion> tags, no quotes, no extra wrappers)

Examples:
1) I opened the Audio Recorder app from the app drawer.
2) I mistakenly entered text in the filename field in the previous step, so in this step I tried to use a long press to select all and delete it.
"""

THINKING_SYSTEM_PROMPT = """You are a high-quality GUI trajectory annotator.

Task:
- You will receive one Android GUI step with:
  1) user instruction (overall goal),
  2) step history text,
  3) the model response for the current step,
  4) current screenshot (possibly with a red marker if the step is click/long_press).
- In previous data, the model response may miss <thinking> entirely or contain overly brief thinking.
- Your task is to synthesize from instruction, history, response, and screenshot, and reconstruct a detailed, in-depth thinking with clear reasoning.
- Note that previous steps may contain mistakes; if correction is needed, analyze it carefully in the thinking.

Output rule:
- Write a refined, in-depth thinking paragraph for this step.
- The thinking should explain the user instruction, analyze the current screen state, analyze prior history (including possible reflection), and then reason toward the current action.
- The thinking should simulate the agent's forward reasoning process, not post-hoc verification.
- DO NOT use patterns like "because the action is xxx, it means xxx"; instead, provide a plausible step-by-step forward rationale.
- DO NOT include any specific coordinates in the thinking.
- The thinking length should be around 100-200 words.
- Keep it factual and grounded in the provided text + screenshot.
- Some model responses may contain something like "Based on the intervention context", but you should assume you are generating the thinking without any external intervention.
- Output plain thinking text only (no <thinking> tags, no quotes, no extra wrappers).
"""

def _iter_tool_calls(response_text: str):
    for m in TOOL_CALL_RE.finditer(response_text or ""):
        yield json.loads(m.group(1))


def _extract_action_and_coord(response_text: str) -> tuple[str | None, list[float] | None]:
    for call in _iter_tool_calls(response_text or ""):
        if call.get("name") != "mobile_use":
            continue
        args = call.get("arguments") or {}
        action = args.get("action")
        coord = args.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            return (str(action) if action is not None else None, [float(coord[0]), float(coord[1])])
        return (str(action) if action is not None else None, None)
    return (None, None)


def _extract_thinking_from_response(response_text: str) -> str:
    m = THINKING_RE.search(response_text or "")
    return m.group(1).strip() if m else ""


def _word_count_by_space(text: str) -> int:
    t = (text or "").strip()
    return len(t.split()) if t else 0


def _avg_original_thinking_words(data: list[dict]) -> float:
    total_steps = 0
    total_words = 0
    for ep in data:
        for step in ep.get("trajectory", []):
            total_steps += 1
            th = _extract_thinking_from_response(str(step.get("response", "") or ""))
            total_words += _word_count_by_space(th)
    return (total_words / total_steps) if total_steps > 0 else 0.0


def _avg_refined_thinking_words(data: list[dict]) -> float:
    total_steps = 0
    total_words = 0
    for ep in data:
        for step in ep.get("trajectory", []):
            total_steps += 1
            total_words += _word_count_by_space(str(step.get("thinking_refine", "") or ""))
    return (total_words / total_steps) if total_steps > 0 else 0.0


def _coord_0_999_to_pixel(coord: list[float], width: int, height: int) -> tuple[int, int]:
    x = float(coord[0])
    y = float(coord[1])
    xp = int(round(x / 999.0 * float(width)))
    yp = int(round(y / 999.0 * float(height)))
    xp = max(0, min(width, xp))
    yp = max(0, min(height, yp))
    return xp, yp


def _image_to_data_url_with_optional_marker(
    image_path: Path,
    action: str | None,
    coord: list[float] | None,
    marked_output_path: Path | None = None,
) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        marked = False
        if action in {"click", "long_press"} and coord is not None:
            x, y = _coord_0_999_to_pixel(coord, im.width, im.height)
            draw = ImageDraw.Draw(im)
            r = max(18, int(min(im.width, im.height) * 0.04))
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                outline=(255, 0, 0),
                width=max(4, int(r * 0.28)),
            )
            marked = True

        if marked and marked_output_path is not None:
            marked_output_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(marked_output_path)

        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


def _build_user_text(
    instruction: str,
    step_history: str,
    response: str,
    mode: str,
    monitor_reason: str = "",
) -> str:
    if mode == "conclusion":
        tail = "Now output ONE concise conclusion for this step."
    else:
        tail = "Now output ONE refined in-depth thinking paragraph for this step."
    monitor_text = ""
    if monitor_reason:
        monitor_text = (
            "\nMonitor intervention context:\n"
            f"The strong monitor decided this step required intervention because: {monitor_reason}\n"
            "Use this as additional context when writing the conclusion or thinking, and consider possible deviation and recovery if relevant.\n"
        )
    return (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Step history:\n"
        f"{step_history}\n\n"
        "Current step model response:\n"
        f"{response}\n\n"
        f"{monitor_text}"
        + tail
    )


def _normalize_one_line(text: str) -> str:
    out = re.sub(r"\s+", " ", (text or "").strip())
    if not out:
        raise ValueError("Empty conclusion returned by model.")
    return out


def _normalize_paragraph(text: str) -> str:
    out = (text or "").strip()
    if not out:
        raise ValueError("Empty thinking returned by model.")
    return out


def _target_key_for_mode(mode: str) -> str:
    if mode == "conclusion":
        return "conclusion"
    return "thinking_refine"


def _has_existing_output(step: dict, mode: str) -> bool:
    key = _target_key_for_mode(mode)
    value = step.get(key)
    return isinstance(value, str) and bool(value.strip())


def _gen_text(
    client: OpenAI,
    model: str,
    instruction: str,
    step_history: str,
    response: str,
    image_data_url: str,
    request_timeout: float,
    mode: str,
    monitor_reason: str = "",
    interactive: bool = False,
) -> str:
    if mode == "conclusion":
        system_prompt = CONCLUSION_SYSTEM_PROMPT
    else:
        system_prompt = THINKING_SYSTEM_PROMPT
    user_text = _build_user_text(
        instruction,
        step_history,
        response,
        mode=mode,
        monitor_reason=monitor_reason,
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        temperature=0.4,
        timeout=request_timeout,
    )
    content = completion.choices[0].message.content
    if interactive:
        print(user_text)
        print("\n--- SYSTEM PROMPT ---\n")
        print(system_prompt)
        print("\n--- MODEL OUTPUT ---\n")
        print(content)
        input()
    if mode == "conclusion":
        return _normalize_one_line(content)
    return _normalize_paragraph(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine per-step conclusion or thinking for *_refine.json.")
    parser.add_argument("--input", type=Path, required=True, help="e.g. runs/rollout/data_merge.json")
    parser.add_argument("--output", type=Path, default=None, help="e.g. runs/rollout/data_merge_conclusion.json")
    parser.add_argument("--image-root", type=Path, default=None, help="default: input.parent")
    parser.add_argument("--model", type=str, default=os.getenv("CONCLUSION_MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--mode", type=str, default="conclusion", choices=["conclusion", "thinking"])
    parser.add_argument("--interactive", action="store_true", help="run step-by-step with print/input")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="only process steps whose target output field is still missing/empty",
    )
    parser.add_argument(
        "--save-temp-images",
        action="store_true",
        help="save click/long_press marked images under <image_root>/temp_images",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if args.output is not None:
        output_path: Path = args.output
    elif args.mode == "conclusion":
        output_path = input_path.with_name(f"{input_path.stem}_conclusion.json")
    else:
        output_path = input_path.with_name(f"{input_path.stem}_thinking.json")
    image_root: Path = args.image_root or input_path.parent
    temp_images_root = image_root / "temp_images" if args.save_temp_images else None
    if temp_images_root is not None:
        temp_images_root.mkdir(parents=True, exist_ok=True)

    if not args.api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (or pass --api-key).")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    tl = threading.local()
    if args.mode == "thinking":
        print(f"avg_words_original_thinking: {_avg_original_thinking_words(data):.2f}")

    jobs: list[tuple[int, int, str, str, str, str, str]] = []
    for ep_idx, ep in enumerate(data):
        instruction = ep["goal"]
        traj = ep["trajectory"]
        for i, step in enumerate(traj):
            if args.only_missing and _has_existing_output(step, args.mode):
                continue
            response = step["response"]
            if i == 0:
                step_history = ""
            else:
                step_history = traj[i - 1]["step_history"]
            image_rel = step["image"]
            monitor_output = step.get("monitor_output") or {}
            monitor_reason = ""
            if (
                isinstance(monitor_output, dict)
                and monitor_output.get("should_intervene") is True
                and isinstance(monitor_output.get("reason"), str)
            ):
                monitor_reason = monitor_output["reason"]
            jobs.append(
                (
                    ep_idx,
                    i,
                    instruction,
                    step_history,
                    response,
                    image_rel,
                    monitor_reason,
                )
            )

    def _get_client() -> OpenAI:
        c = getattr(tl, "client", None)
        if c is None:
            c = OpenAI(api_key=args.api_key, base_url=args.base_url)
            tl.client = c
        return c

    def _process_one(job: tuple[int, int, str, str, str, str, str]) -> tuple[int, int, str]:
        ep_idx, step_idx, instruction, step_history, response, image_rel, monitor_reason = job
        action, coord = _extract_action_and_coord(response)
        image_path = image_root / image_rel
        marked_output_path = (temp_images_root / image_rel) if temp_images_root is not None else None
        image_data_url = _image_to_data_url_with_optional_marker(
            image_path, action, coord, marked_output_path=marked_output_path
        )
        text_out = _gen_text(
            client=_get_client(),
            model=args.model,
            instruction=instruction,
            step_history=step_history,
            response=response,
            image_data_url=image_data_url,
            request_timeout=float(args.request_timeout),
            mode=args.mode,
            monitor_reason=monitor_reason,
            interactive=args.interactive,
        )
        if args.sleep > 0:
            time.sleep(args.sleep)
        return ep_idx, step_idx, text_out

    if args.interactive:
        for job in tqdm(jobs, total=len(jobs), desc="steps"):
            try:
                ep_idx, step_idx, text_out = _process_one(job)
                key = _target_key_for_mode(args.mode)
                data[ep_idx]["trajectory"][step_idx][key] = text_out
            except Exception as e:
                ep_idx, step_idx, instruction, _step_history, _response, image_rel, _monitor_reason = job
                print(
                    f"[process_refine error] mode={args.mode} ep={ep_idx} step={step_idx} "
                    f"image={image_rel} instruction={instruction!r} error={e!r}"
                )
                key = _target_key_for_mode(args.mode)
                if args.mode == "conclusion":
                    data[ep_idx]["trajectory"][step_idx][key] = ""
                else:
                    data[ep_idx]["trajectory"][step_idx][key] = ""
    else:
        with cf.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            fut_to_job = {ex.submit(_process_one, job): job for job in jobs}
            for fut in tqdm(cf.as_completed(fut_to_job), total=len(fut_to_job), desc="steps"):
                try:
                    ep_idx, step_idx, text_out = fut.result()
                    key = _target_key_for_mode(args.mode)
                    data[ep_idx]["trajectory"][step_idx][key] = text_out
                except Exception as e:
                    ep_idx, step_idx, instruction, _step_history, _response, image_rel, _monitor_reason = fut_to_job[fut]
                    print(
                        f"[process_refine error] mode={args.mode} ep={ep_idx} step={step_idx} "
                        f"image={image_rel} instruction={instruction!r} error={e!r}"
                    )
                    key = _target_key_for_mode(args.mode)
                    if args.mode == "conclusion":
                        data[ep_idx]["trajectory"][step_idx][key] = ""
                    else:
                        data[ep_idx]["trajectory"][step_idx][key] = ""

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.mode == "thinking":
        print(f"avg_words_thinking_refine: {_avg_refined_thinking_words(data):.2f}")
    print(f"[OK] saved: {output_path}")


if __name__ == "__main__":
    main()

