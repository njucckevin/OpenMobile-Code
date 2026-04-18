"""
process_context.py - prepare retrieval context for GUI task synthesis.

For each element in unique_screen_with_elements.json, gather:
1. the element itself
2. N outgoing transitions where the current screen is screen_before
3. M semantically related functionalities within the same app
"""

import os
import time
# Set env vars before importing libraries that depend on them.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Do not force offline mode in code. If needed, export these before running:
#   export HF_HUB_OFFLINE=1
#   export TRANSFORMERS_OFFLINE=1

import json
import pickle
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============ SentenceTransformer loading ============
_embedding_model = None
OPENAI_EMBEDDING_BASE_URL = (
    os.environ.get("OPENAI_EMBEDDING_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or "http://127.0.0.1:8000/v1"
)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load the sentence-transformers model once and reuse it.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    print("Loading sentence transformer model...")
    _embedding_model = SentenceTransformer(model_name)
    print("Sentence transformer model loaded.")

    return _embedding_model


def encode_with_openai(texts: list[str], model_name: str) -> np.ndarray:
    """
    Encode texts through an OpenAI-compatible embeddings endpoint for retrieval.
    """
    if not texts:
        return np.asarray([], dtype=np.float32)

    api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    batch_size = 16
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    batch_results = [None] * len(batches)

    def embed_one_batch(batch_idx: int, batch: list[str]):
        client = OpenAI(api_key=api_key, base_url=OPENAI_EMBEDDING_BASE_URL)
        resp = client.embeddings.create(model=model_name, input=batch)
        return batch_idx, [item.embedding for item in resp.data]

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(embed_one_batch, idx, batch)
            for idx, batch in enumerate(batches)
        ]
        future_iter = as_completed(futures)
        if len(futures) > 1:
            future_iter = tqdm(
                future_iter,
                total=len(futures),
                desc=f"Encoding texts with OpenAI ({model_name}, 32 threads)",
                leave=False,
            )
        for future in future_iter:
            batch_idx, batch_embeddings = future.result()
            batch_results[batch_idx] = batch_embeddings

    vectors = [emb for batch_embeddings in batch_results for emb in batch_embeddings]
    return np.asarray(vectors, dtype=np.float32)


def encode_texts_for_retrieval(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """
    Unified text-encoding entry point for retrieval.
    """
    model_name = model_name or EMBEDDING_MODEL_NAME
    if model_name == "all-MiniLM-L6-v2":
        model = get_embedding_model(model_name)
        return np.asarray(model.encode(texts, show_progress_bar=False), dtype=np.float32)
    if model_name == "openai/text-embedding-3-large":
        return encode_with_openai(texts, "text-embedding-3-large")
    raise ValueError(
        "retrieval embedding model only supports: all-MiniLM-L6-v2, openai/text-embedding-3-large"
    )

# ============ Configurable initialization ============
# process_context is responsible for full-dataset retrieval and indexing, so it
# should always load the full unique_screen data rather than the filtered view.

# No default paths: callers must explicitly pass the full knowledge base.
UNIQUE_SCREEN_PATH = None
STATE_TRANSFER_PATH = None

# Embeddings are relatively fast to compute; disable cache by default to avoid
# mismatch across datasets.
USE_EMBEDDING_CACHE = False
EMBEDDING_CACHE_PATH = None

# Runtime resources populated by init_context().
unique_screens = None
state_transfers = None
image_to_unique_idx = None
app_to_screen_indices = None
app_embeddings = None
app_func_list = None

_initialized_cfg = None  # (unique_screen_path, state_transfer_path, use_embedding_cache, embedding_model_name)


def init_context(
    unique_screen_path: str,
    state_transfer_path: str,
    use_embedding_cache: bool = False,
    embedding_model_name: str = "all-MiniLM-L6-v2",
):
    """
    Initialize or reset the full retrieval knowledge base and embedding index.
    The pipeline should call this explicitly once with the full
    unique_screen_with_elements.json.
    """
    global UNIQUE_SCREEN_PATH, STATE_TRANSFER_PATH, USE_EMBEDDING_CACHE, EMBEDDING_CACHE_PATH
    global EMBEDDING_MODEL_NAME
    global unique_screens, state_transfers, image_to_unique_idx, app_to_screen_indices
    global app_embeddings, app_func_list, _initialized_cfg

    if not unique_screen_path or not state_transfer_path:
        raise ValueError("init_context() requires unique_screen_path and state_transfer_path")

    cfg = (
        unique_screen_path,
        state_transfer_path,
        bool(use_embedding_cache),
        str(embedding_model_name),
    )
    if _initialized_cfg == cfg and unique_screens is not None and state_transfers is not None:
        return

    UNIQUE_SCREEN_PATH = unique_screen_path
    STATE_TRANSFER_PATH = state_transfer_path
    USE_EMBEDDING_CACHE = bool(use_embedding_cache)
    EMBEDDING_MODEL_NAME = str(embedding_model_name)
    _unique_tag = os.path.splitext(os.path.basename(UNIQUE_SCREEN_PATH))[0]
    EMBEDDING_CACHE_PATH = f"functionality_embeddings_cache_{_unique_tag}.pkl"

    print("Loading data...")
    unique_screens = json.load(open(UNIQUE_SCREEN_PATH, "r"))
    state_transfers = json.load(open(STATE_TRANSFER_PATH, "r"))
    print(f"Loaded {len(unique_screens)} unique screens, {len(state_transfers)} state transfers")

    print("Building basic indices...")
    image_to_unique_idx = {}
    for idx, screen in enumerate(unique_screens):
        for img_name in screen.get("similar_images", [screen["screen"]]):
            image_to_unique_idx[img_name] = idx

    app_to_screen_indices = defaultdict(list)
    for idx, screen in enumerate(unique_screens):
        app_to_screen_indices[screen["app"]].append(idx)

    app_embeddings = {}
    app_func_list = {}

    if USE_EMBEDDING_CACHE and EMBEDDING_CACHE_PATH and os.path.exists(EMBEDDING_CACHE_PATH):
        print(f"Loading embeddings from cache: {EMBEDDING_CACHE_PATH}")
        with open(EMBEDDING_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
            cached_tag = cache.get("unique_tag")
            cached_screen_count = cache.get("unique_screen_count")
            cached_model_name = cache.get("embedding_model_name")
            cache_ok = (
                cached_tag == _unique_tag
                and cached_screen_count == len(unique_screens)
                and cached_model_name == EMBEDDING_MODEL_NAME
            )
            if cache_ok:
                app_embeddings = cache["app_embeddings"]
                app_func_list = cache["app_func_list"]
                print(f"Loaded embeddings for {len(app_embeddings)} apps from cache.")
            else:
                print("Embedding cache is not compatible; will recompute.")

    if not app_embeddings:
        print("Computing embeddings (cache disabled or unavailable)...")

        app_functionalities = defaultdict(list)
        print("Collecting functionalities...")
        for screen_idx, screen in enumerate(tqdm(unique_screens, desc="Processing screens")):
            app = screen["app"]
            elements = screen.get("elements") or []
            for elem_idx, elem in enumerate(elements):
                if elem.get("type") == "functionality" and elem.get("description"):
                    app_functionalities[app].append(
                        {
                            "screen_idx": screen_idx,
                            "element_idx": elem_idx,
                            "element": elem,
                            "description": elem["description"],
                        }
                    )

        print("Computing embeddings for all functionalities...")
        for app, func_list in tqdm(app_functionalities.items(), desc="Computing embeddings"):
            if not func_list:
                continue
            descriptions = [f["description"] for f in func_list]
            embeddings = encode_texts_for_retrieval(descriptions, EMBEDDING_MODEL_NAME)
            app_embeddings[app] = np.array(embeddings)
            app_func_list[app] = func_list

        # Do not save cache here to avoid mismatches after dataset changes.

    print(f"Indices built. Apps: {len(app_to_screen_indices)}")
    _initialized_cfg = cfg

    # Print one summary after initialization for a quick sanity check.
    print_functionality_stats(similarity_threshold=0.8)


def _ensure_initialized():
    if unique_screens is None or state_transfers is None or image_to_unique_idx is None:
        raise ValueError("process_context not initialized")
        # init_context(UNIQUE_SCREEN_PATH, STATE_TRANSFER_PATH, use_embedding_cache=USE_EMBEDDING_CACHE)

# Count unique functionalities after deduplication (similarity > threshold counts as duplicate).
def count_unique_functionalities(embeddings, threshold=0.9):
    """Count unique functionalities with greedy clustering."""
    if len(embeddings) == 0:
        return 0
    
    # Normalize embeddings.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    normalized = embeddings / norms
    
    # Greedy clustering: pick representative elements.
    representatives = [0]  # Use the first item as the first representative.
    
    for i in range(1, len(embeddings)):
        # Compute similarity against all representatives.
        rep_embeddings = normalized[representatives]
        similarities = np.dot(rep_embeddings, normalized[i])
        
        # Add a new representative if similarity to all existing ones is below the threshold.
        if np.max(similarities) < threshold:
            representatives.append(i)
    
    return len(representatives)


def print_functionality_stats(similarity_threshold: float = 0.8):
    """
    Print summary statistics for functionality elements.
    Note: `init_context(...)` must be called first.
    """
    _ensure_initialized()

    print(f"\nFunctionality statistics (with deduplication at similarity >= {similarity_threshold}):")
    print("-" * 70)
    print(f"{'App':<30} {'Screens':>10} {'Total Funcs':>12} {'Unique Funcs':>12}")
    print("-" * 70)

    total_funcs = 0
    total_unique = 0
    for app, indices in app_to_screen_indices.items():
        func_count = len(app_func_list.get(app, []))
        total_funcs += func_count

        if app in app_embeddings and len(app_embeddings[app]) > 0:
            unique_count = count_unique_functionalities(
                app_embeddings[app], threshold=float(similarity_threshold)
            )
        else:
            unique_count = 0
        total_unique += unique_count

        print(f"  {app:<28} {len(indices):>10} {func_count:>12} {unique_count:>12}")

    print("-" * 70)
    print(f"  {'TOTAL':<28} {len(unique_screens):>10} {total_funcs:>12} {total_unique:>12}")
    if total_funcs:
        print(f"\nDeduplication ratio: {total_unique}/{total_funcs} = {total_unique/total_funcs*100:.1f}%")

def get_transition_targets(screen_element, n=3):
    """
    Get up to `n` transition targets where the current screen acts as `screen_before`.
    
    Returns a list of dicts, each containing:
        - target_screen: the destination unique_screen element
        - action: action metadata that caused the transition (action_type, bbox)
        - original_screen_after: the original screen_after image name
    """
    as_before_indices = screen_element.get("as_screen_before", [])
    if not as_before_indices:
        return []
    _ensure_initialized()
    
    app = screen_element["app"]
    results = []
    seen_target_screens = set()  # Avoid duplicate target screens.
    
    for sample_idx in as_before_indices:
        if len(results) >= n:
            break
            
        transfer = state_transfers[sample_idx]
        screen_after_name = transfer["screen_after"]
        
        # Find the unique_screen corresponding to screen_after.
        target_unique_idx = image_to_unique_idx.get(screen_after_name)
        if target_unique_idx is None:
            continue
        
        # Keep only targets from the same app.
        target_screen = unique_screens[target_unique_idx]
        if target_screen["app"] != app:
            continue
        
        # Skip duplicates.
        if target_unique_idx in seen_target_screens:
            continue
        seen_target_screens.add(target_unique_idx)
        
        results.append({
            "target_screen": target_screen,
            "action": {
                "action_type": transfer["action_type"],
                "bbox": transfer.get("bbox")
            },
            "original_screen_after": screen_after_name
        })
    
    return results


def get_transition_sources(screen_element, n=1):
    """
    Get up to `n` predecessor screens where the current screen acts as `screen_after`.

    Returns a list of dicts, each containing:
        - source_screen: the predecessor unique_screen element
        - action: action metadata that led to the current screen (action_type, bbox)
        - original_screen_before: the original screen_before image name
    """
    as_after_indices = screen_element.get("as_screen_after", [])
    if not as_after_indices or int(n) <= 0:
        return []
    _ensure_initialized()

    app = screen_element["app"]
    results = []
    seen_source_screens = set()

    for sample_idx in as_after_indices:
        if len(results) >= n:
            break

        transfer = state_transfers[sample_idx]
        screen_before_name = transfer["screen_before"]

        source_unique_idx = image_to_unique_idx.get(screen_before_name)
        if source_unique_idx is None:
            continue

        source_screen = unique_screens[source_unique_idx]
        if source_screen["app"] != app:
            continue

        if source_unique_idx in seen_source_screens:
            continue
        seen_source_screens.add(source_unique_idx)

        results.append(
            {
                "source_screen": source_screen,
                "action": {
                    "action_type": transfer["action_type"],
                    "bbox": transfer.get("bbox"),
                },
                "original_screen_before": screen_before_name,
            }
        )

    return results


def get_related_functionalities(screen_element, m=30, similarity_threshold=0.8):
    """
    Get the top `M` semantically related functionalities within the same app.
    
    Strategy: retrieve several related items for each functionality description
    on the current screen, then deduplicate and return the top M items while
    keeping pairwise similarity within the pool below `threshold`.
    
    Returns a list of dicts containing:
        - element: the functionality element (type, label, description)
        - screen: the source unique_screen element
        - similarity: similarity score
    """
    app = screen_element["app"]
    _ensure_initialized()
    current_screen_idx = None
    
    # Locate the current screen index.
    for idx, screen in enumerate(unique_screens):
        if screen["screen"] == screen_element["screen"]:
            current_screen_idx = idx
            break
    
    # Collect functionality descriptions from the current screen.
    current_elements = screen_element.get("elements") or []
    current_descriptions = []
    for elem in current_elements:
        if elem.get("type") == "functionality" and elem.get("description"):
            current_descriptions.append(elem["description"])
    
    if not current_descriptions:
        return []
    
    # Check whether this app has precomputed embeddings.
    if app not in app_embeddings or len(app_embeddings[app]) == 0:
        return []
    
    all_embeddings = app_embeddings[app]
    all_funcs = app_func_list[app]
    
    # Compute similarities from each current functionality to the full app pool.
    current_embeddings = encode_texts_for_retrieval(current_descriptions, EMBEDDING_MODEL_NAME)
    
    # Compute the similarity matrix: (num_current, num_all), using cosine similarity.
    current_norm = current_embeddings / (np.linalg.norm(current_embeddings, axis=1, keepdims=True) + 1e-9)
    all_norm = all_embeddings / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-9)
    similarity_matrix = np.dot(current_norm, all_norm.T)
    
    # Collect top-k candidates for each current description.
    k_per_desc = max(1, (m * 3) // len(current_descriptions))  # Retrieve extra items because later steps deduplicate them.
    
    candidates = {}  # func_key -> (similarity, func_info, embedding_idx)
    
    for i, desc in enumerate(current_descriptions):
        similarities = similarity_matrix[i]
        top_indices = np.argsort(similarities)[::-1]
        
        count = 0
        for j in top_indices:
            if count >= k_per_desc:
                break
            
            sim_score = float(similarities[j])
            
            # Skip candidates too similar to the current screen.
            if sim_score >= similarity_threshold:
                continue
            
            func_info = all_funcs[j]
            
            # Exclude functionalities from the current screen itself.
            if func_info["screen_idx"] == current_screen_idx:
                continue
            
            # Use (screen_idx, element_idx) as a unique identifier.
            func_key = (func_info["screen_idx"], func_info["element_idx"])
            
            # Keep the highest similarity and record its embedding index.
            if func_key not in candidates or candidates[func_key][0] < sim_score:
                candidates[func_key] = (sim_score, func_info, j)
            
            count += 1
    
    # Sort by similarity.
    sorted_candidates = sorted(candidates.values(), key=lambda x: x[0], reverse=True)
    
    # Build the pool greedily while enforcing pairwise similarity < threshold.
    results = []
    selected_indices = []  # Track indices of selected elements in `all_embeddings`.
    
    for sim_score, func_info, emb_idx in sorted_candidates:
        if len(results) >= m:
            break

        # Check similarity against already selected elements.
        is_diverse = True
        if selected_indices:
            candidate_emb = all_norm[emb_idx]
            for sel_idx in selected_indices:
                sel_emb = all_norm[sel_idx]
                inter_sim = float(np.dot(candidate_emb, sel_emb))
                if inter_sim >= similarity_threshold:
                    is_diverse = False
                    break

        if is_diverse:
            source_screen = unique_screens[func_info["screen_idx"]]
            results.append({
                "element": func_info["element"],
                "screen": source_screen,
                "similarity": sim_score
            })
            selected_indices.append(emb_idx)
    
    return results


def _filter_target_screen_elements(target_screen, current_embeddings, current_norm, similarity_threshold=0.8):
    """
    Filter target-screen elements, keeping only functionalities whose similarity
    to the current screen is below `threshold`.
    
    Returns a new `target_screen` dict (shallow copy) whose `elements` field is filtered.
    """
    target_elements = target_screen.get("elements") or []
    if not target_elements or current_embeddings is None:
        return target_screen
    _ensure_initialized()
    
    # Collect functionality descriptions from the target screen.
    target_funcs = []
    for i, elem in enumerate(target_elements):
        if elem.get("type") == "functionality" and elem.get("description"):
            target_funcs.append((i, elem["description"]))
    
    if not target_funcs:
        return target_screen
    
    # Compute embeddings for target functionalities.
    target_descriptions = [desc for _, desc in target_funcs]
    target_embeddings = encode_texts_for_retrieval(target_descriptions, EMBEDDING_MODEL_NAME)
    target_norm = target_embeddings / (np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-9)
    
    # Compute similarity: (num_target, num_current)
    similarity_matrix = np.dot(target_norm, current_norm.T)
    max_similarities = np.max(similarity_matrix, axis=1)  # Maximum similarity between each target element and the current screen.
    
    # Find element indices to keep.
    filtered_indices = set()
    for j, (orig_idx, _) in enumerate(target_funcs):
        if max_similarities[j] < similarity_threshold:
            filtered_indices.add(orig_idx)
    
    # Keep non-functionality elements and low-similarity functionality elements.
    filtered_elements = []
    for i, elem in enumerate(target_elements):
        if elem.get("type") != "functionality":
            filtered_elements.append(elem)
        elif i in filtered_indices:
            filtered_elements.append(elem)
    
    # Return a new dict without modifying the original object.
    new_target = dict(target_screen)
    new_target["elements"] = filtered_elements
    return new_target


def get_context(screen_element, num_predecessors=1, n=3, m=30, similarity_threshold=0.8):
    """
    Main entry point for retrieving the full context of a screen element.
    
    Inputs:
        screen_element: one element from `unique_screen_with_elements.json`
        num_predecessors: maximum number of predecessor screens to return (default 1)
        n: maximum number of transition targets to return (default 3)
        m: maximum number of related functionalities to return (default 30)
        similarity_threshold: keep functionalities only when similarity is below this threshold (default 0.8)
    
    Output: a dict containing:
        1. "current": the current element itself
        2. "predecessors": predecessor screens with filtered `source_screen["elements"]`
        3. "successors": up to `n` transition targets with filtered `target_screen["elements"]`
        4. "related_functionalities": up to `m` semantically related functionalities whose pairwise similarity stays below the threshold
    """
    _ensure_initialized()
    # Precompute current-screen functionality embeddings for transition filtering.
    current_elements = screen_element.get("elements") or []
    current_descriptions = [
        elem["description"] for elem in current_elements
        if elem.get("type") == "functionality" and elem.get("description")
    ]
    
    current_embeddings = None
    current_norm = None
    if current_descriptions:
        current_embeddings = encode_texts_for_retrieval(current_descriptions, EMBEDDING_MODEL_NAME)
        current_norm = current_embeddings / (np.linalg.norm(current_embeddings, axis=1, keepdims=True) + 1e-9)
    
    # Get predecessor screens and filter their elements.
    predecessors = get_transition_sources(screen_element, n=int(num_predecessors))
    filtered_predecessors = []
    for trans in predecessors:
        new_trans = dict(trans)
        new_trans["source_screen"] = _filter_target_screen_elements(
            trans["source_screen"], current_embeddings, current_norm, similarity_threshold
        )
        filtered_predecessors.append(new_trans)

    # Get successor screens and filter their elements.
    successors = get_transition_targets(screen_element, n=n)
    filtered_successors = []
    for trans in successors:
        new_trans = dict(trans)
        new_trans["target_screen"] = _filter_target_screen_elements(
            trans["target_screen"], current_embeddings, current_norm, similarity_threshold
        )
        filtered_successors.append(new_trans)
    
    return {
        "current": screen_element,
        "predecessors": filtered_predecessors,
        "successors": filtered_successors,
        "transitions": filtered_successors,  # backward compatibility for old callers
        "related_functionalities": get_related_functionalities(
            screen_element, m=m, similarity_threshold=similarity_threshold
        ),
    }


# ============ Test entry point ============
if __name__ == "__main__":
    raise ValueError(
        "process_context requires explicit init_context(unique_screen_path, state_transfer_path). "
        "Please call init_context(...) from pipeline or your own script."
    )
