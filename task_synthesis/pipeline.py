"""
pipeline.py
"""

import argparse
import os
import json
from pathlib import Path

from process_transfer import (
    build_unique_screens,
    annotate_unique_screens_with_elements,
    filter_unique_screens_with_elements_by_task,
)
from process_task import synthesize_tasks
from eval_task import evaluate_tasks_file
from process_task_refine import refine_tasks_file, dedup_new_tasks_against_existing_files


HERE = Path(__file__).resolve().parent
ANDROIDWORLD_EXPLORE_RESULTS = HERE.parent / "AndroidWorld" / "explore_results"


def _default_state_transfer_path() -> str:
    candidates = [
        ANDROIDWORLD_EXPLORE_RESULTS / "state_transfer_explore.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _default_screenshots_dir() -> str:
    return str(ANDROIDWORLD_EXPLORE_RESULTS / "screenshots")


def step1_state_transfer_to_unique_screen(
    dataset_id: str,
    state_transfer_path: str,
    screenshots_dir: str,
    outputs_root: str = "output",
    similarity_threshold: float = 0.95,
    cache_path: str | None = None,
) -> str:
    out_dir = os.path.join(outputs_root, dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    if cache_path is None:
        cache_path = os.path.join(out_dir, f"screen_transition_cache_{dataset_id}.pkl")

    out_json = os.path.join(out_dir, f"unique_screen_{dataset_id}.json")
    if os.path.exists(out_json):
        print(f"Step 1 skipped, output already exists: {out_json}")
        return out_json

    build_unique_screens(
        data_path=state_transfer_path,
        screenshots_dir=screenshots_dir,
        cache_path=cache_path,
        similarity_threshold=similarity_threshold,
        output_path=out_json,
    )

    return out_json


def step2_unique_screen_to_with_elements(
    dataset_id: str,
    state_transfer_path: str,
    screenshots_dir: str,
    outputs_root: str = "output",
    similarity_threshold: float = 0.95,
    cache_path: str | None = None,
    max_workers: int = 64,
) -> str:
    out_dir = os.path.join(outputs_root, dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    if cache_path is None:
        cache_path = os.path.join(out_dir, f"screen_transition_cache_{dataset_id}.pkl")

    unique_screen_path = os.path.join(out_dir, f"unique_screen_{dataset_id}.json")
    out_json = os.path.join(out_dir, f"unique_screen_{dataset_id}_with_elements.json")
    if os.path.exists(out_json):
        print(f"Step 2 skipped, output already exists: {out_json}")
        return out_json

    annotate_unique_screens_with_elements(
        unique_screen_path=unique_screen_path,
        output_path=out_json,
        state_transfer_path=state_transfer_path,
        screenshots_dir=screenshots_dir,
        cache_path=cache_path,
        similarity_threshold=similarity_threshold,
        num_threads=int(max_workers),
    )

    return out_json


def step3_with_elements_to_filtered(
    dataset_id: str,
    state_transfer_path: str,
    outputs_root: str = "output",
    min_as_screen_before: int = 1,
) -> str:
    out_dir = os.path.join(outputs_root, dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    in_json = os.path.join(out_dir, f"unique_screen_{dataset_id}_with_elements.json")
    out_json = os.path.join(out_dir, f"unique_screen_{dataset_id}_with_elements_filter.json")
    if os.path.exists(out_json):
        print(f"Step 3 skipped, output already exists: {out_json}")
        return out_json

    filter_unique_screens_with_elements_by_task(
        unique_screen_with_elements_path=in_json,
        state_transfer_path=state_transfer_path,
        output_path=out_json,
        min_as_screen_before=int(min_as_screen_before),
    )

    return out_json


def step4_synthesize_tasks(
    dataset_id: str,
    state_transfer_path: str,
    screenshots_dir: str,
    outputs_root: str = "output",
    max_num: int = 10000,
    max_workers: int = 64,
    context_embedding_model: str = "all-MiniLM-L6-v2",
) -> str:
    out_dir = os.path.join(outputs_root, dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    # Input: filtered screens used as synthesis anchors.
    filtered_unique_screen_path = os.path.join(out_dir, f"unique_screen_{dataset_id}_with_elements_filter.json")

    # Context knowledge base: the full unique_screen_with_elements file.
    context_unique_screen_path = os.path.join(out_dir, f"unique_screen_{dataset_id}_with_elements.json")

    out_json = os.path.join(out_dir, f"synthesized_tasks_{dataset_id}.json")
    if os.path.exists(out_json):
        print(f"Step 4 skipped, output already exists: {out_json}")
        return out_json

    synthesize_tasks(
        filtered_unique_screen_path=filtered_unique_screen_path,
        screenshots_dir=screenshots_dir,
        output_path=out_json,
        context_unique_screen_path=context_unique_screen_path,
        context_state_transfer_path=state_transfer_path,
        max_num=int(max_num),
        max_workers=int(max_workers),
        context_embedding_model=str(context_embedding_model),
    )
    return out_json


def step5_eval_tasks(
    dataset_id: str,
    outputs_root: str = "output",
    model: str = None,
    max_workers: int = 64,
) -> str:
    out_dir = os.path.join(outputs_root, dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    in_json = os.path.join(out_dir, f"synthesized_tasks_{dataset_id}.json")
    out_json = os.path.join(out_dir, f"synthesized_tasks_{dataset_id}_judge_gemini.json")
    if os.path.exists(out_json):
        print(f"Step 5 skipped, output already exists: {out_json}")
        return out_json

    evaluate_tasks_file(
        input_path=in_json,
        output_path=out_json,
        max_workers=int(max_workers),
    )

    return out_json


def step6_final_refine_and_add_task_id(
    dataset_id: str,
    state_transfer_path: str,
    outputs_root: str = "output",
) -> str:
    out_dir = os.path.join(outputs_root, dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    judge_path = os.path.join(out_dir, f"synthesized_tasks_{dataset_id}_judge_gemini.json")
    final_path = os.path.join(out_dir, f"synthesized_tasks_{dataset_id}_final.json")
    if os.path.exists(final_path):
        print(f"Step 6 skipped, output already exists: {final_path}")
        return final_path

    # Refine the judged tasks (filtering, de-duplication, and truncation).
    refine_tasks_file(
        input_path=judge_path,
        output_path=final_path,
    )

    # Backfill task_id from the source state_transfer file.
    transfer = json.load(open(state_transfer_path, "r"))
    screen_2_taskid = {}
    for item in transfer:
        task_id = item["task_id"]
        screen_2_taskid[item["screen_before"]] = task_id
        screen_2_taskid[item["screen_after"]] = task_id

    syn_data = json.load(open(final_path, "r"))
    for item in syn_data:
        screen = item["screen"]
        task_id = screen_2_taskid[screen]
        item["task_id"] = task_id

    json.dump(syn_data, open(final_path, "w"), indent=2, ensure_ascii=False)
    return final_path


def step7_dedup_against_existing(
    dataset_id: str,
    existing_tasks_paths: list[str],
    outputs_root: str = "output",
    *,
    threshold: float = 0.9,
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> str:
    out_dir = os.path.join(outputs_root, dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    final_tasks_path = os.path.join(out_dir, f"synthesized_tasks_{dataset_id}_final.json")
    dedup_path = os.path.join(out_dir, f"synthesized_tasks_{dataset_id}_final_dedup.json")
    if os.path.exists(dedup_path):
        print(f"Step 7 skipped, output already exists: {dedup_path}")
        return dedup_path

    dedup_new_tasks_against_existing_files(
        new_tasks_path=final_tasks_path,
        existing_tasks_paths=existing_tasks_paths,
        output_path=dedup_path,
        threshold=float(threshold),
        embedding_model=str(embedding_model),
        batch_size=int(batch_size),
    )
    return dedup_path


def main():
    p = argparse.ArgumentParser(description="CUA_Task pipeline.")
    p.add_argument("--dataset_id", required=True, help="Dataset id used in output file names.")
    p.add_argument("--outputs_root", default=str(HERE / "output"), help="Output root directory. Defaults to task_synthesis/output.")
    p.add_argument("--similarity_threshold", type=float, default=0.95, help="pHash similarity threshold.")
    p.add_argument("--cache_path", default=None, help="Optional cache path. Defaults to output/{dataset_id}/screen_transition_cache_{dataset_id}.pkl.")
    p.add_argument("--state_transfer", default=_default_state_transfer_path(), help="Path to the state_transfer JSON from AndroidWorld exploration.")
    p.add_argument("--screenshots_dir", default=_default_screenshots_dir(), help="Path to the screenshots directory from AndroidWorld exploration.")
    p.add_argument("--max_num_syn_screen", type=int, default=1000, help="Maximum number of screens used for task synthesis.")
    p.add_argument("--max_workers", type=int, default=64, help="Shared worker count for steps 2/4/5.")
    p.add_argument(
        "--context_embedding_model",
        default="all-MiniLM-L6-v2",
        help="Embedding model for step-4 context retrieval: all-MiniLM-L6-v2 or openai/text-embedding-3-large.",
    )
    p.add_argument(
        "--dedup_against",
        nargs="*",
        default=None,
        help="Optional list of existing task JSON files used for final de-duplication.",
    )
    args = p.parse_args()

    print("Step 1: state_transfer -> unique_screen")
    out1 = step1_state_transfer_to_unique_screen(
        dataset_id=args.dataset_id,
        state_transfer_path=args.state_transfer,
        screenshots_dir=args.screenshots_dir,
        outputs_root=args.outputs_root,
        similarity_threshold=float(args.similarity_threshold),
        cache_path=args.cache_path,
    )
    print(out1)

    print("Step 2: unique_screen -> unique_screen_with_elements")
    out2 = step2_unique_screen_to_with_elements(
        dataset_id=args.dataset_id,
        state_transfer_path=args.state_transfer,
        screenshots_dir=args.screenshots_dir,
        outputs_root=args.outputs_root,
        similarity_threshold=float(args.similarity_threshold),
        cache_path=args.cache_path,
        max_workers=int(args.max_workers),
    )
    print(out2)

    print("Step 3: unique_screen_with_elements -> filtered (task-consistent & transitions>=k)")
    out3 = step3_with_elements_to_filtered(
        dataset_id=args.dataset_id,
        state_transfer_path=args.state_transfer,
        outputs_root=args.outputs_root,
    )
    print(out3)

    print("Step 4: synthesize tasks (filtered screens, context built from full unique_screen_with_elements)")
    out4 = step4_synthesize_tasks(
        dataset_id=args.dataset_id,
        state_transfer_path=args.state_transfer,
        screenshots_dir=args.screenshots_dir,
        outputs_root=args.outputs_root,
        max_num=int(args.max_num_syn_screen),
        max_workers=int(args.max_workers),
        context_embedding_model=args.context_embedding_model,
    )
    print(out4)

    print("Step 5: eval tasks (judge)")
    out5 = step5_eval_tasks(
        dataset_id=args.dataset_id,
        outputs_root=args.outputs_root,
        max_workers=int(args.max_workers),
    )
    print(out5)

    print("Step 6: refine + add task_id -> final")
    out6 = step6_final_refine_and_add_task_id(
        dataset_id=args.dataset_id,
        state_transfer_path=args.state_transfer,
        outputs_root=args.outputs_root,
    )
    print(out6)

    if args.dedup_against:
        print("Step 7: de-dup final tasks against existing files (threshold<0.9)")
        out7 = step7_dedup_against_existing(
            existing_tasks_paths=list(args.dedup_against),
            dataset_id=args.dataset_id,
            outputs_root=args.outputs_root,
        )
        print(out7)


if __name__ == "__main__":
    main()

