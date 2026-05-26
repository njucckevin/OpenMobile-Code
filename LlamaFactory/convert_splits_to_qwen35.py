#!/usr/bin/env python3
"""Convert OpenMobile ShareGPT splits from Qwen3-VL SFT format to Qwen3.5-VL format.

Changes applied to each training sample:
  1. system: replace Qwen3-VL-style prompt (JSON tool_call spec) with QWEN35_SYSTEM_PROMPT.
  2. assistant: rewrite each <tool_call>...</tool_call> body from JSON to Qwen3.5 XML.

Input format (LLaMA-Factory ShareGPT), e.g. split1.json:
  [{"id": "...", "messages": [{"role": "system", ...}, ...], "images": [...]}, ...]

Usage:
  python convert_splits_to_qwen35.py --input data/split1.json --output data/split1_qwen35.json
  python convert_splits_to_qwen35.py --input data --output data_qwen35 --glob "split*.json"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Import QWEN35_SYSTEM_PROMPT from AndroidWorld agents.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_DIR = _REPO_ROOT / "AndroidWorld" / "android_world" / "agents"
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from PROMPT import QWEN35_SYSTEM_PROMPT  # noqa: E402

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.IGNORECASE)


def parse_tool_call_json_block(text: str) -> dict[str, Any] | None:
    m = _TOOL_CALL_RE.search(text or "")
    if not m:
        return None
    payload = m.group(1).strip()
    if payload.startswith("<function="):
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "name" not in data:
        return None
    return data


def tool_call_dict_to_xml(tool_call: dict[str, Any]) -> str:
    """Serialize tool call dict to Qwen3.5 XML inside <tool_call> tags."""
    name = str(tool_call.get("name", "mobile_use"))
    raw_args = tool_call.get("arguments", {})
    args = raw_args if isinstance(raw_args, dict) else {}

    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            val_str = json.dumps(value, ensure_ascii=False)
        else:
            val_str = str(value)
        lines.extend([f"<parameter={key}>", val_str, "</parameter>"])
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def convert_assistant_content(content: str, *, skip_if_already_xml: bool) -> tuple[str, bool]:
    """Return (new_content, changed)."""
    if skip_if_already_xml and "<function=" in (content or ""):
        return content, False
    tool_call = parse_tool_call_json_block(content)
    if tool_call is None:
        return content, False
    xml_block = tool_call_dict_to_xml(tool_call)
    # Use a callable replacement: re.sub treats string replacements as escape templates
    # (e.g. "\d" in Action JSON breaks); a lambda inserts xml_block literally.
    new_content = _TOOL_CALL_RE.sub(lambda _m: xml_block, content, count=1)
    return new_content, new_content != content


def convert_sample(sample: dict[str, Any], *, skip_if_already_xml: bool) -> dict[str, Any]:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError("sample missing 'messages' list")

    new_messages: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        if role == "system":
            content = QWEN35_SYSTEM_PROMPT.strip()
        elif role == "assistant":
            content, _ = convert_assistant_content(
                content, skip_if_already_xml=skip_if_already_xml
            )

        new_messages.append({"role": role, "content": content})

    out = dict(sample)
    out["messages"] = new_messages
    return out


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of samples")
    return data


def save_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    skip_if_already_xml: bool,
) -> dict[str, int]:
    samples = load_json(input_path)
    stats = {
        "samples": len(samples),
        "system_replaced": 0,
        "assistant_tool_calls_converted": 0,
        "assistant_unchanged": 0,
    }

    converted: list[dict[str, Any]] = []
    for sample in samples:
        messages = sample.get("messages", [])
        if any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
            stats["system_replaced"] += 1

        before_assistant = [
            m.get("content", "")
            for m in messages
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        out = convert_sample(sample, skip_if_already_xml=skip_if_already_xml)
        after_assistant = [
            m.get("content", "")
            for m in out.get("messages", [])
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        for b, a in zip(before_assistant, after_assistant):
            if b != a:
                stats["assistant_tool_calls_converted"] += 1
            else:
                stats["assistant_unchanged"] += 1

        converted.append(out)

    save_json(output_path, converted)
    return stats


def collect_inputs(input_path: Path, glob_pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    files = sorted(input_path.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {glob_pattern} under {input_path}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenMobile ShareGPT JSON from Qwen3-VL to Qwen3.5 SFT format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .json file or directory containing split JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .json file (single input) or output directory (batch)",
    )
    parser.add_argument(
        "--glob",
        default="split*.json",
        help="When --input is a directory, match files with this glob (default: split*.json)",
    )
    parser.add_argument(
        "--suffix",
        default="_qwen35",
        help="When batching, output filename stem suffix before .json (default: _qwen35)",
    )
    parser.add_argument(
        "--skip-if-already-xml",
        action="store_true",
        help="Do not rewrite assistant turns that already contain <function=...>",
    )
    args = parser.parse_args()

    inputs = collect_inputs(args.input, args.glob)
    batch = len(inputs) > 1 or args.input.is_dir()

    if batch:
        args.output.mkdir(parents=True, exist_ok=True)

    for inp in inputs:
        if batch:
            out = args.output / f"{inp.stem}{args.suffix}.json"
        else:
            out = args.output
        stats = convert_file(
            inp,
            out,
            skip_if_already_xml=args.skip_if_already_xml,
        )
        print(
            f"[OK] {inp.name} -> {out} | samples={stats['samples']} "
            f"system={stats['system_replaced']} "
            f"tool_call_converted={stats['assistant_tool_calls_converted']} "
            f"assistant_unchanged={stats['assistant_unchanged']}"
        )


if __name__ == "__main__":
    main()
