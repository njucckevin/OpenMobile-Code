
"""
process_task_refine.py

Refine judged synthesized tasks with basic de-duplication, filtering, and
per-app truncation, then write a final rollout-ready JSON file.

Requirements:
1) Follow the deduplication logic in `eval_task.py`: use sentence-transformers
   embeddings + L2 normalization, and greedily keep tasks whose maximum cosine
   similarity to the retained set is < 0.8.
2) Filter out tasks with clarity < 4 or reasonableness < 4 (scores equal to 4
   are kept). Complexity is not used for filtering, but it is used for sorting.
3) Count tasks per app and keep at most 100 instructions for each app:
   - if the total is <= 100, keep all of them after steps (1) and (2)
   - if the total is > 100, prioritize samples with clarity==5 and
     reasonableness==5, and within that pool prefer higher complexity

Notes:
- Greedy deduplication is sensitive to input order. To better match the goal of
  keeping higher-quality samples in step (3), this script globally sorts
  candidates before deduplication (high eval quality first, then higher complexity).
- The deduplication algorithm itself stays aligned with `eval_task.py`
  (greedy representative set + 0.8 threshold).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any
import random
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


_embedding_model: SentenceTransformer | None = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Match the loading strategy in `eval_task.py`: load the model only once.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    print("Loading sentence transformer model...")
    _embedding_model = SentenceTransformer(model_name)
    print("Sentence transformer model loaded.")

    return _embedding_model


def _safe_int(x, default: int | None = None) -> int | None:
    try:
        return int(x)
    except Exception:
        return default


def _get_scores(sample: dict) -> tuple[int | None, int | None, int | None]:
    ev = sample.get("eval") or {}
    c = _safe_int(ev.get("complexity"))
    cl = _safe_int(ev.get("clarity"))
    r = _safe_int(ev.get("reasonableness"))
    return c, cl, r


def filter_by_eval(samples: list[dict], min_clarity: int = 4, min_reasonableness: int = 4) -> list[dict]:
    kept = []
    for s in samples:
        _, cl, r = _get_scores(s)
        if cl is None or r is None:
            continue
        if cl >= min_clarity and r >= min_reasonableness:
            kept.append(s)
    return kept


def quality_sort_key(sample: dict, strategy: str = "both5_complexity_sum") -> tuple:
    """
    Sorting key used before deduplication and before per-app truncation.
    Larger values are preferred when used with `reverse=True`.

    Because greedy deduplication is order-sensitive, this sorting directly
    affects which task is retained inside the same semantic cluster.

    strategy:
    - both5_complexity_sum (default):
        First prioritize samples where both clarity==5 and reasonableness==5;
        within that high-quality pool, prefer higher complexity;
        use the total score as a final tiebreaker.
    - sum:
        Sort directly by (complexity + clarity + reasonableness), descending.
    - weighted:
        Sort by the weighted sum `w_c*complexity + w_cl*clarity + w_r*reasonableness`
        where the weights are provided by the caller.
        Note: this function returns only base scores; the weighted aggregation is handled outside.
    """
    c, cl, r = _get_scores(sample)
    # Treat None as the lowest possible value.
    c = c if c is not None else -1
    cl = cl if cl is not None else -1
    r = r if r is not None else -1

    if strategy == "sum":
        total = c + cl + r
        return (total, c, cl, r)

    # Default strategy: both5_complexity_sum
    both_5 = 1 if (cl == 5 and r == 5) else 0
    total = c + cl + r
    # Prioritize both_5 first, then complexity, and use total score as a tiebreaker.
    return (both_5, c, total, cl, r)


def weighted_quality_sort_key(sample: dict, w_c: float, w_cl: float, w_r: float) -> tuple:
    """Key for weighted sorting; larger values are better."""
    c, cl, r = _get_scores(sample)
    c = c if c is not None else -1
    cl = cl if cl is not None else -1
    r = r if r is not None else -1
    score = w_c * c + w_cl * cl + w_r * r
    return (score, c, cl, r)


def greedy_dedup_by_embedding(
    samples: list[dict],
    threshold: float = 0.8,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> list[dict]:
    """
    Deduplicate tasks following the logic in `eval_task.py`:
    - embed task text and apply L2 normalization
    - build a greedy representative set; if the maximum cosine similarity to
      existing representatives is >= threshold, treat it as duplicate
    """
    tasks: list[str] = []
    idx_map: list[int] = []
    for i, s in enumerate(samples):
        t = (s.get("task") or "").strip()
        if t:
            tasks.append(t)
            idx_map.append(i)

    if not tasks:
        return []

    model = get_embedding_model(model_name)

    # Encode in batches to avoid large peak memory usage, then concatenate into an (n, d) array.
    all_emb = []
    for st in tqdm(range(0, len(tasks), batch_size), desc="Embedding tasks"):
        chunk = tasks[st : st + batch_size]
        emb = model.encode(chunk, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        all_emb.append(emb)
    emb = np.concatenate(all_emb, axis=0)

    # Normalize embeddings, matching `eval_task.py`.
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb = emb / norms

    n = emb.shape[0]
    if n == 1:
        return [samples[idx_map[0]]]

    representatives = [0]
    # Greedy deduplication, one item at a time.
    for i in tqdm(range(1, n), desc="Greedy de-dup"):
        rep_emb = emb[representatives]  # (k, d)
        sims = rep_emb @ emb[i]         # (k,)
        if float(np.max(sims)) < threshold:
            representatives.append(i)

    kept_task_indices = set(representatives)
    kept_samples: list[dict] = []
    for j, orig_i in enumerate(idx_map):
        if j in kept_task_indices:
            kept_samples.append(samples[orig_i])
    return kept_samples


def per_app_topk(
    samples: list[dict],
    k: int = 100,
    sort_strategy: str = "both5_complexity_sum",
    w_c: float = 2.0,
    w_cl: float = 1.0,
    w_r: float = 1.0,
) -> list[dict]:
    by_app: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        app = (s.get("app") or "UNKNOWN").strip() or "UNKNOWN"
        by_app[app].append(s)

    final: list[dict] = []
    for app, items in by_app.items():
        if len(items) <= k:
            final.extend(items)
            continue
        # Sort and truncate within each app only.
        if sort_strategy == "weighted":
            items_sorted = sorted(items, key=lambda s: weighted_quality_sort_key(s, w_c, w_cl, w_r), reverse=True)
        else:
            items_sorted = sorted(items, key=lambda s: quality_sort_key(s, strategy=sort_strategy), reverse=True)
        final.extend(items_sorted[:k])
    return final


def print_app_stats(title: str, samples: list[dict], topn: int = 20) -> None:
    cnt = defaultdict(int)
    for s in samples:
        app = (s.get("app") or "UNKNOWN").strip() or "UNKNOWN"
        cnt[app] += 1
    total = len(samples)
    print(f"\n== {title} ==")
    print(f"Total: {total}")
    pairs = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
    print(f"Unique apps: {len(pairs)}")
    print(f"Top {topn} apps by count:")
    for app, c in pairs[:topn]:
        print(f"  {app}: {c}")


def print_avg_scores(title: str, samples: list[dict]) -> None:
    """Print average scores for the three evaluation dimensions, following `eval_task.py`."""
    sum_c = sum_cl = sum_r = 0
    cnt = 0
    for s in samples:
        c, cl, r = _get_scores(s)
        if c is None or cl is None or r is None:
            continue
        # Scores should theoretically be 1-5; accept any value that can be parsed as int.
        sum_c += c
        sum_cl += cl
        sum_r += r
        cnt += 1

    print(f"\n== {title} ==")
    if cnt == 0:
        print("No valid evaluations parsed. (cnt=0)")
        return
    print(f"Evaluated: {cnt}/{len(samples)}")
    print(f"Avg Complexity: {sum_c / cnt:.4f}")
    print(f"Avg Clarity: {sum_cl / cnt:.4f}")
    print(f"Avg Reasonableness: {sum_r / cnt:.4f}")


def refine_tasks_file(
    input_path: str,
    output_path: str,
    *,
    threshold: float = 0.8,
    max_per_app: int = 100,
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    sort_strategy: str = "both5_complexity_sum",
    w_c: float = 2.0,
    w_cl: float = 1.0,
    w_r: float = 1.0,
) -> list[dict]:
    """
    Filter, deduplicate, and truncate judge results by app, then write the final task JSON.
    This function reuses the logic from `main()` so it can be called by the pipeline.
    """
    data: Any = json.load(open(input_path, "r"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")

    # Basic cleanup: keep only dict items with non-empty task text.
    samples = []
    for x in data:
        if not isinstance(x, dict):
            continue
        t = (x.get("task") or "").strip()
        if not t:
            continue
        samples.append(x)

    print_app_stats("Raw (after non-empty task cleaning)", samples)

    # Step 2: score-based filtering (clarity/reasonableness >= 4)
    samples = filter_by_eval(samples, min_clarity=4, min_reasonableness=4)
    print_app_stats("After eval filter (clarity>=4 & reasonableness>=4)", samples)

    # First sort globally by quality, then apply greedy embedding-based deduplication.
    if sort_strategy == "weighted":
        samples_sorted = sorted(
            samples,
            key=lambda s: weighted_quality_sort_key(s, w_c, w_cl, w_r),
            reverse=True,
        )
    else:
        samples_sorted = sorted(
            samples,
            key=lambda s: quality_sort_key(s, strategy=sort_strategy),
            reverse=True,
        )

    deduped = greedy_dedup_by_embedding(
        samples_sorted,
        threshold=float(threshold),
        model_name=embedding_model,
        batch_size=int(batch_size),
    )
    print_app_stats(f"After embedding de-dup (threshold<{threshold})", deduped)

    # Step 3: keep at most `max_per_app` items per app.
    final = per_app_topk(
        deduped,
        k=int(max_per_app),
        sort_strategy=str(sort_strategy),
        w_c=float(w_c),
        w_cl=float(w_cl),
        w_r=float(w_r),
    )
    print_app_stats(f"Final (per-app max={max_per_app})", final)

    # Apply a final keyword filter for selected apps only; keep all others unchanged.
    # Rules (case-insensitive):
    # - Broccoli - Recipe App: task must contain either "broccoli" or "recipe"
    # - Markor: must contain "markor"
    # - Simple Calendar Pro: must contain "simple calendar pro"
    # - Pro Expense: must contain "expense"
    # - Joplin: must contain "joplin"
    # - Tasks: must contain "tasks"
    # - OpenTracks: must contain "opentracks"
    # - Files: must contain "files"
    # - Retro Music: must contain "retro music"
    # - OsmAnd: must contain "osmand"
    # - VLC: must contain "vlc"
    # - Simple Draw Pro: must contain "simple draw pro"
    _APP_TASK_KEYWORDS_ANY = {
        "Broccoli - Recipe App": ["broccoli", "recipe"],
    }
    _APP_TASK_KEYWORDS_ALL = {
        "Markor": ["markor"],
        "Simple Calendar Pro": ["simple calendar pro"],
        "Pro Expense": ["expense"],
        "Joplin": ["joplin"],
        "Tasks": ["tasks"],
        "OpenTracks": ["opentracks"],
        "Files": ["files"],
        "Retro Music": ["retro music"],
        "OsmAnd": ["osmand"],
        "VLC": ["vlc"],
        "Simple Draw Pro": ["simple draw pro"],
    }

    def _pass_app_keyword_filter(s: dict) -> bool:
        app = (s.get("app") or "").strip()
        task = (s.get("task") or "").strip()
        if not app or not task:
            return True  # Avoid over-constraining here; earlier cleaning already enforces non-empty tasks.
        t = task.lower()
        if app in _APP_TASK_KEYWORDS_ANY:
            kws = _APP_TASK_KEYWORDS_ANY[app]
            return any(kw.lower() in t for kw in kws)
        if app in _APP_TASK_KEYWORDS_ALL:
            kws = _APP_TASK_KEYWORDS_ALL[app]
            return all(kw.lower() in t for kw in kws)
        return True

    final = [s for s in final if _pass_app_keyword_filter(s)]
    random.shuffle(final)
    # Reformat `final` into the rollout schema:
    # rename "task_aw" -> "base_task_name", "task" -> "instruction",
    # and assign a sequential "sample_id" to each item.
    for i, s in enumerate(final):
        if "task_aw" in s and "task" in s:
            s["base_task_name"] = s.pop("task_aw")
            s["instruction"] = s.pop("task")
        else:
            raise ValueError(f"task_aw or task not found in {s}")
        s["sample_id"] = i

    print_app_stats("Final task", final)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")
    print_avg_scores("Final average scores", final)
    return final


def dedup_new_tasks_against_existing_files(
    new_tasks_path: str,
    existing_tasks_paths: list[str],
    output_path: str,
    *,
    threshold: float = 0.8,
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> list[dict]:
    """
    Deduplicate tasks from `new_tasks_path` against the existing task set using
    embedding cosine similarity:
    - discard a new task if its maximum similarity to any existing task is >= threshold
    - otherwise keep it

    This only filters "new vs existing" tasks. It does not sort by quality and
    does not deduplicate within the new file itself.
    """

    def _extract_text(item: dict) -> str:
        return item["instruction"]

    # load new
    new_data: Any = json.load(open(new_tasks_path, "r"))
    if not isinstance(new_data, list):
        raise ValueError("new_tasks_path JSON must be a list.")
    new_items = [x for x in new_data if isinstance(x, dict) and _extract_text(x)]

    # load existing
    existing_texts: list[str] = []
    seen = set()
    for p in existing_tasks_paths:
        if not p:
            continue
        data: Any = json.load(open(p, "r"))
        if not isinstance(data, list):
            continue
        for x in data:
            if not isinstance(x, dict):
                continue
            t = _extract_text(x)
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            existing_texts.append(t)

    # If there are no existing tasks, write the new items unchanged.
    if not existing_texts:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(new_items, f, indent=2, ensure_ascii=False)
        return new_items

    model = get_embedding_model(embedding_model)

    # embed existing (normalized)
    existing_emb_chunks = []
    for st in tqdm(range(0, len(existing_texts), int(batch_size)), desc="Embedding existing tasks"):
        chunk = existing_texts[st : st + int(batch_size)]
        emb = model.encode(chunk, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        existing_emb_chunks.append(emb)
    existing_emb = np.concatenate(existing_emb_chunks, axis=0)
    existing_emb = existing_emb / (np.linalg.norm(existing_emb, axis=1, keepdims=True) + 1e-9)

    # filter new in batches
    kept: list[dict] = []
    new_texts = [_extract_text(x) for x in new_items]
    for st in tqdm(range(0, len(new_texts), int(batch_size)), desc="Filtering new tasks"):
        chunk_texts = new_texts[st : st + int(batch_size)]
        chunk_items = new_items[st : st + int(batch_size)]
        emb = model.encode(chunk_texts, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

        sims = emb @ existing_emb.T  # (b, m)
        max_sims = np.max(sims, axis=1)
        for item, mx in zip(chunk_items, max_sims):
            if float(mx) < float(threshold):
                kept.append(item)
    
    print(f"Kept {len(kept)} tasks out of {len(new_items)} new tasks")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON path (list[dict]).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument("--threshold", type=float, default=0.8, help="Deduplication similarity threshold (`>=` counts as duplicate).")
    parser.add_argument("--max_per_app", type=int, default=100, help="Maximum number of tasks to keep per app.")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Sentence-transformers model name.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for embedding encoding.")
    parser.add_argument(
        "--sort_strategy",
        default="both5_complexity_sum",
        choices=["both5_complexity_sum", "sum", "weighted"],
        help="Sorting strategy used before deduplication and per-app truncation.",
    )
    parser.add_argument("--w_c", type=float, default=2.0, help="Weight for complexity under the `weighted` strategy.")
    parser.add_argument("--w_cl", type=float, default=1.0, help="Weight for clarity under the `weighted` strategy.")
    parser.add_argument("--w_r", type=float, default=1.0, help="Weight for reasonableness under the `weighted` strategy.")
    args = parser.parse_args()

    refine_tasks_file(
        input_path=args.input,
        output_path=args.output,
        threshold=float(args.threshold),
        max_per_app=int(args.max_per_app),
        embedding_model=args.embedding_model,
        batch_size=int(args.batch_size),
        sort_strategy=str(args.sort_strategy),
        w_c=float(args.w_c),
        w_cl=float(args.w_cl),
        w_r=float(args.w_r),
    )


if __name__ == "__main__":
    main()

