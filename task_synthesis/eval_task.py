"""
eval_task.py - evaluate synthesized GUI tasks with a strong model API.

Input: a JSON file containing list[dict], with each sample shaped like:
  {"app": "xxx", "task": "xxx"}

Output:
  - prints average scores for the three evaluation dimensions
  - optionally writes per-sample eval results to JSON
"""

import os
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import sys

from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from utils import get_client


# ============ Configuration ============
DEFAULT_MODEL = "gemini-3.1-pro-preview"
OPENAI_EMBEDDING_BASE_URL = (
    os.environ.get("OPENAI_EMBEDDING_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or "http://127.0.0.1:8000/v1"
)

_embedding_model = None




def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load the sentence-transformer model once and reuse it.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    print("Loading sentence transformer model...")
    _embedding_model = SentenceTransformer(model_name)
    print("Sentence transformer model loaded.")

    return _embedding_model


def encode_with_openai(texts: list[str], model_name: str, max_workers: int = 32) -> np.ndarray:
    """
    Encode texts with an OpenAI-compatible embeddings endpoint for diversity
    computation.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    batch_size = 16
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    batch_results = [None] * len(batches)

    def embed_one_batch(batch_idx: int, batch: list[str]):
        client = OpenAI(api_key=api_key, base_url=OPENAI_EMBEDDING_BASE_URL)
        resp = client.embeddings.create(model=model_name, input=batch)
        return batch_idx, [item.embedding for item in resp.data]

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        futures = [
            executor.submit(embed_one_batch, idx, batch)
            for idx, batch in enumerate(batches)
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Encoding texts with OpenAI ({model_name}, {max_workers} threads)",
        ):
            batch_idx, batch_embeddings = future.result()
            batch_results[batch_idx] = batch_embeddings

    vectors = [emb for batch_embeddings in batch_results for emb in batch_embeddings]
    return np.asarray(vectors, dtype=np.float32)


def encode_texts_for_diversity(texts: list[str], model_name: str) -> np.ndarray:
    """
    Unified text-encoding entry point for diversity computation.
    """
    if model_name == "all-MiniLM-L6-v2":
        model = get_embedding_model(model_name)
        return np.asarray(model.encode(texts, show_progress_bar=False), dtype=np.float32)
    if model_name == "openai/text-embedding-3-large":
        return encode_with_openai(texts, "text-embedding-3-large")
    raise ValueError(
        "diversity_model only supports: all-MiniLM-L6-v2, openai/text-embedding-3-large"
    )


def compute_instruction_diversity(
    samples: list,
    model_name: str = "all-MiniLM-L6-v2",
    max_samples: int | None = None,
    seed: int = 0,
    similarity_threshold: float = 0.8,
) -> dict:
    """
    Estimate instruction diversity with embeddings:
    - embed and normalize task texts
    - greedily treat samples with cosine similarity >= threshold as duplicates
    - diversity = num_unique / n
    """
    tasks = []
    for s in samples:
        if isinstance(s, dict):
            t = (s.get("task") or "").strip()
        else:
            t = str(s).strip()
        if t:
            tasks.append(t)

    if not tasks:
        return {"n": 0, "diversity": None, "mean_cosine_similarity": None}

    if max_samples is not None and len(tasks) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tasks), size=max_samples, replace=False)
        tasks = [tasks[i] for i in idx]

    emb = encode_texts_for_diversity(tasks, model_name)

    # normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb = emb / norms

    n = emb.shape[0]
    if n == 1:
        return {
            "n": 1,
            "num_unique": 1,
            "diversity": 1.0,
            "threshold": float(similarity_threshold),
        }

    # Greedy representative set.
    representatives = [0]
    for i in range(1, n):
        rep_emb = emb[representatives]  # (k, d)
        sims = rep_emb @ emb[i]         # (k,)
        if float(np.max(sims)) < similarity_threshold:
            representatives.append(i)

    num_unique = len(representatives)
    diversity = num_unique / n
    return {
        "n": n,
        "num_unique": num_unique,
        "diversity": float(diversity),
        "threshold": float(similarity_threshold),
    }


EVAL_SYSTEM_PROMPT = r"""
You are an expert evaluator for synthesized mobile GUI tasks/instructions.

Given an Android app name and a single task/instruction (starting from the phone home screen), evaluate the quality of the instruction on THREE dimensions, each scored as an integer from 1 to 5:

1) Complexity (1-5): The more complex and difficult the instruction is—and the more steps it involves—the higher the score. Single-step instructions are overly simple and should receive a low score. 
- Good example (5): Help me set an alarm for tomorrow morning at 10:00, repeat every weekday, add a description reminding me to clock in, and set the ringtone to "WakaWaka".
- Bad example (1): Open the Camera app.

2) Clarity (1-5): The instruction should clearly specify what needs to be done, rather than being vague or underspecified. For example, “Update an existing entry to include a new time, adjust its details, and save the changes in the app.” is not good; the instruction should explicitly state what content to set. Assume you are an executor starting from the phone home screen—does this instruction let you clearly understand what to do.
- Good example (5): In the Expense app, change the entry "Dinner with my senior" to yesterday, set the expense amount to 100 USD, and add the description "I paid for it".
- Bad example 1 (1): Update an existing entry to include a new time, adjust its details, and save the changes in the app. (Reason: It does not specify the exact entry configuration/details to set.)
- Bad example 2 (3): In OpenTracks, open the 'Sailing Expedition' track, switch the view to the 'By time' speed graph, and then use the share button to export the track data. (Reason: It does not specify how to share/export the track data.)
- Bad example 3 (2): Browse and select a new recipe to view its details from the recipe list. (Reason: It does not specify which recipe, and "view its details" is vague.)
- Bad example 4 (2): Send the currently viewed photo as a message via the SMS Messenger app. (Reason: It does not specify how to compose and send the message.)

3) Reasonableness (1-5): The instruction should be logically coherent, not just a stack of operations. More reasonable and realistic instructions should receive higher scores.
- Good example (5): Open my photo gallery, take the total amount shown on the receipt in the first photo, and record it in the Expense app with the name "Christmas dinner" and the date set to yesterday.
- Bad example (1): In the TODO app, mark all of today's todos as completed, take a photo, and then go to System Settings.

Important constraints for your evaluation:
- The environment has NO network connection and tasks should NOT require login or internet.
- Do NOT invent extra context. Judge only based on the task text itself.

Return ONLY a valid JSON object with this exact schema (no markdown, no extra text):
{
  "complexity": <int 1-5>,
  "clarity": <int 1-5>,
  "reasonableness": <int 1-5>,
}
""".strip()


def _extract_json_obj(text: str):
    """Extract a JSON object from model output. Return None on failure."""
    try:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and s < e:
            text = text[s : e + 1]
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def _clamp_score(x):
    try:
        x = int(x)
    except Exception:
        return None
    if 1 <= x <= 5:
        return x
    return None


def eval_one(sample: dict, model: str):
    app = (sample.get("app") or "").strip()
    task = (sample.get("task") or "").strip()
    if not task:
        return {
            **sample,
            "eval": None,
            "raw_response": None,
            "error": "empty task",
        }

    user_prompt = f"App: {app or 'UNKNOWN'}\nTask: {task}"

    # client = get_client_openrouter()
    client = get_client()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        task_preview = task[:200].replace("\n", "\\n")
        print(f"[API_ERROR] {e} | app={app} | task={task_preview}", file=sys.stderr)
        return {
            **sample,
            "eval": None,
            "raw_response": None,
            "error": f"api_error: {e}",
        }

    raw = resp.choices[0].message.content or ""
    # print(raw)

    obj = _extract_json_obj(raw)
    if obj is None:
        task_preview = task[:200].replace("\n", "\\n")
        print(f"[PARSE_ERROR] failed to parse json | app={app} | task={task_preview} | raw={raw[:200]}", file=sys.stderr)
        return {
            **sample,
            "eval": None,
            "raw_response": raw,
            "error": "failed to parse json",
        }

    scores = {
        "complexity": _clamp_score(obj.get("complexity")),
        "clarity": _clamp_score(obj.get("clarity")),
        "reasonableness": _clamp_score(obj.get("reasonableness")),
    }
    if scores["complexity"] is None or scores["clarity"] is None or scores["reasonableness"] is None:
        return {
            **sample,
            "eval": None,
            "raw_response": raw,
            "error": "invalid score range/type",
        }

    return {
        **sample,
        "eval": scores,
        "raw_response": raw,
    }


def evaluate_tasks_file(
    input_path: str,
    output_path: str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    max_workers: int = 16,
    stall_timeout: float = 180.0,
    diversity_max_samples: int = 1000,
    diversity_model: str = "all-MiniLM-L6-v2",
    diversity_threshold: float = 0.8,
    diversity_only: bool = False,
) -> list[dict]:
    """
    Evaluate a task JSON file (list[dict]) and optionally save per-sample
    results to output_path.

    Design goals:
    - keep eval_task.py usable as a standalone CLI
    - also expose a function interface for pipeline reuse
    """
    data = json.load(open(input_path, "r"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples like {'app': 'xxx', 'task': 'xxx'}.")

    # Compute diversity for the input task set first.
    max_n = None if diversity_max_samples <= 0 else int(diversity_max_samples)
    div = compute_instruction_diversity(
        data,
        model_name=diversity_model,
        max_samples=max_n,
        similarity_threshold=float(diversity_threshold),
    )
    if div.get("diversity") is None:
        print("Diversity: N/A (no valid tasks)")
    else:
        print(
            f"Diversity (unique_ratio, embedding-based): {div['diversity']:.6f}  "
            f"|  num_unique={div['num_unique']}  |  n={div['n']}  |  threshold={div['threshold']:.2f}"
        )

    if diversity_only:
        return []

    results = []
    sum_c = sum_cl = sum_r = 0
    cnt = 0

    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futures = [ex.submit(eval_one, s if isinstance(s, dict) else {"task": str(s)}, model) for s in data]
        pending = set(futures)
        pbar = tqdm(total=len(futures))

        # Stall watchdog: wait at most stall_timeout seconds for the next future.
        while pending:
            it = as_completed(pending, timeout=float(stall_timeout))
            try:
                fu = next(it)
            except TimeoutError:
                # Treat this as stalled: cancel remaining futures and record errors.
                for pf in list(pending):
                    pf.cancel()
                    results.append(
                        {
                            "eval": None,
                            "raw_response": None,
                            "error": f"stalled_timeout_no_completion_within_{stall_timeout}s",
                        }
                    )
                break

            pending.discard(fu)
            pbar.update(1)
            try:
                r = fu.result()
            except Exception as e:
                r = {
                    "eval": None,
                    "raw_response": None,
                    "error": f"unhandled_future_exception: {e}",
                }

            results.append(r)
            ev = r.get("eval") or {}
            if ev.get("complexity") and ev.get("clarity") and ev.get("reasonableness"):
                sum_c += ev["complexity"]
                sum_cl += ev["clarity"]
                sum_r += ev["reasonableness"]
                cnt += 1

    if cnt == 0:
        print("No valid evaluations parsed. (cnt=0)")
    else:
        print(f"Evaluated: {cnt}/{len(results)}")
        print(f"Avg Complexity: {sum_c / cnt:.4f}")
        print(f"Avg Clarity: {sum_cl / cnt:.4f}")
        print(f"Avg Reasonableness: {sum_r / cnt:.4f}")

    if div.get("diversity") is None:
        print("Diversity: N/A (no valid tasks)")
    else:
        print(
            f"Diversity (unique_ratio, embedding-based): {div['diversity']:.6f}  "
            f"|  num_unique={div['num_unique']}  |  n={div['n']}  |  threshold={div['threshold']:.2f}"
        )

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved per-sample results to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesized tasks via a strong LLM API.")
    parser.add_argument("--input", required=True, help="Path to input JSON (list of {app, task}).")
    parser.add_argument("--output", default=None, help="Optional: path to save per-sample eval results.")
    parser.add_argument("--model", default=os.getenv("EVAL_MODEL") or DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--max_workers", type=int, default=16, help="Concurrency for API calls.")
    parser.add_argument("--stall_timeout", type=float, default=180.0, help="Stall watchdog: if no task finishes within this time, cancel remaining to avoid hanging forever.")
    parser.add_argument("--diversity_max_samples", type=int, default=1000, help="Max samples used to compute diversity (None for all).")
    parser.add_argument(
        "--diversity_model",
        default="all-MiniLM-L6-v2",
        help="Diversity embedding model: all-MiniLM-L6-v2 or openai/text-embedding-3-large.",
    )
    parser.add_argument("--diversity_threshold", type=float, default=0.8, help="Cosine similarity threshold for de-duplication (>= threshold considered similar).")
    parser.add_argument("--diversity_only", action="store_true", help="Only compute and print diversity; skip judge scoring and do not save output.")
    args = parser.parse_args()

    evaluate_tasks_file(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        max_workers=int(args.max_workers),
        stall_timeout=float(args.stall_timeout),
        diversity_max_samples=int(args.diversity_max_samples),
        diversity_model=args.diversity_model,
        diversity_threshold=float(args.diversity_threshold),
        diversity_only=bool(args.diversity_only),
    )


if __name__ == "__main__":
    main()

