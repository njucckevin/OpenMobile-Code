"""
Screen transition index utilities.

This module organizes transition relationships between Android screenshots and
supports similarity matching with perceptual hashes.
"""

import os
import json
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from PIL import Image
import imagehash
from tqdm import tqdm
import numpy as np


class ScreenTransitionIndex:
    """
    Screen transition index.

    Features:
    1. map image filenames to sample ids
    2. detect similar images with perceptual hashes
    3. query related screen_before / screen_after sample ids
    """
    
    def __init__(
        self,
        data_path: str = "state_transfer_explore.json",
        screenshots_dir: str = "screenshots_transfer",
        hash_size: int = 16,  # Larger values are more precise but slower.
        similarity_threshold: float = 0.95,  # Similarity threshold.
        cache_path: Optional[str] = None  # Hash cache path.
    ):
        self.data_path = data_path
        self.screenshots_dir = screenshots_dir
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold
        
        # Default cache path.
        if cache_path is None:
            self.cache_path = os.path.join(
                os.path.dirname(data_path), 
                "screen_transition_cache.pkl"
            )
        else:
            self.cache_path = cache_path
        
        # Load data.
        print("Loading data...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Index structures.
        # filename -> sample indices where this image is screen_before
        self.screen_before_index: Dict[str, List[int]] = defaultdict(list)
        # filename -> sample indices where this image is screen_after
        self.screen_after_index: Dict[str, List[int]] = defaultdict(list)
        # filename -> perceptual hash
        self.image_hashes: Dict[str, imagehash.ImageHash] = {}
        # all unique image filenames
        self.all_images: Set[str] = set()
        
        # Build the basic indices.
        self._build_basic_index()
        
    def _build_basic_index(self):
        """Build basic filename-to-sample-id indices."""
        print("Building basic indices...")
        for idx, item in enumerate(tqdm(self.data, desc="Building indices")):
            screen_before = item["screen_before"]
            screen_after = item["screen_after"]
            
            self.screen_before_index[screen_before].append(idx)
            self.screen_after_index[screen_after].append(idx)
            
            self.all_images.add(screen_before)
            self.all_images.add(screen_after)
        
        print(f"Found {len(self.all_images)} unique images")
        print(f"Found {len(self.data)} samples")
        
    def compute_all_hashes(self, force_recompute: bool = False):
        """
        Compute perceptual hashes for all images.
        
        Args:
            force_recompute: Whether to ignore cache and recompute hashes.
        """
        # Try to load cache first.
        if not force_recompute and os.path.exists(self.cache_path):
            print(f"Loading hashes from cache: {self.cache_path}")
            try:
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data.get('hash_size') == self.hash_size:
                        self.image_hashes = cache_data['hashes']
                        print(f"Loaded {len(self.image_hashes)} image hashes")
                        
                        # Compute hashes for newly added images if needed.
                        missing = self.all_images - set(self.image_hashes.keys())
                        if missing:
                            print(f"Found {len(missing)} new images; computing hashes...")
                            self._compute_hashes_for_images(missing)
                            self._save_cache()
                        return
            except Exception as e:
                print(f"Failed to load cache: {e}")
        
        # Compute hashes for all images.
        print("Computing perceptual hashes for all images...")
        self._compute_hashes_for_images(self.all_images)
        self._save_cache()
        
    def _compute_hashes_for_images(self, images: Set[str]):
        """Compute hashes for the specified images."""
        for img_name in tqdm(images, desc="Computing hashes"):
            img_path = os.path.join(self.screenshots_dir, img_name)
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    # Use perceptual hash (pHash).
                    self.image_hashes[img_name] = imagehash.phash(img, hash_size=self.hash_size)
                except Exception as e:
                    print(f"Failed to process image {img_name}: {e}")
            else:
                # Skip missing files.
                pass
                
    def _save_cache(self):
        """Save the hash cache."""
        print(f"Saving cache to: {self.cache_path}")
        cache_data = {
            'hash_size': self.hash_size,
            'hashes': self.image_hashes
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
            
    def compute_hash_distance(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
        """
        Compute similarity between two hashes.
        
        Returns:
            Similarity in [0, 1], where 1 means identical.
        """
        # Hamming distance
        hamming_distance = hash1 - hash2
        # Maximum possible distance
        max_distance = self.hash_size * self.hash_size
        # Convert distance to similarity
        similarity = 1 - (hamming_distance / max_distance)
        return similarity
    
    def find_similar_images(self, image_name: str) -> List[Tuple[str, float]]:
        """
        Find all images similar to the given image.
        
        Args:
            image_name: Image filename.
            
        Returns:
            A list of similar images, where each item is `(filename, similarity)`.
        """
        if image_name not in self.image_hashes:
            # If the hash was not precomputed, try computing it on demand.
            img_path = os.path.join(self.screenshots_dir, image_name)
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    self.image_hashes[image_name] = imagehash.phash(img, hash_size=self.hash_size)
                except Exception as e:
                    print(f"Failed to compute image hash for {image_name}: {e}")
                    return [(image_name, 1.0)]
            else:
                print(f"Image does not exist: {img_path}")
                return [(image_name, 1.0)]
        
        target_hash = self.image_hashes[image_name]
        similar_images = []
        
        for img_name, img_hash in self.image_hashes.items():
            similarity = self.compute_hash_distance(target_hash, img_hash)
            if similarity >= self.similarity_threshold:
                similar_images.append((img_name, similarity))
        
        # Sort by similarity descending.
        similar_images.sort(key=lambda x: x[1], reverse=True)
        return similar_images
    
    def query_transitions(
        self, 
        image_name: str, 
        use_similarity: bool = True
    ) -> Dict[str, List[int]]:
        """
        Query transitions associated with the given image.
        
        Args:
            image_name: Image filename.
            use_similarity: Whether to use similarity matching and include similar images.
            
        Returns:
            A dict containing:
            - "as_screen_before": sample indices where this image appears as screen_before
            - "as_screen_after": sample indices where this image appears as screen_after
            - "similar_images": similar image list when `use_similarity=True`
        """
        result = {
            "as_screen_before": [],  # All transitions leaving this screen.
            "as_screen_after": [],   # All transitions entering this screen.
            "similar_images": []
        }
        
        if use_similarity and self.image_hashes:
            # Use similarity-based matching.
            similar_images = self.find_similar_images(image_name)
            result["similar_images"] = similar_images
            
            # Collect sample IDs for all similar images.
            seen_before = set()
            seen_after = set()
            
            for img_name, similarity in similar_images:
                for idx in self.screen_before_index.get(img_name, []):
                    if idx not in seen_before:
                        seen_before.add(idx)
                        result["as_screen_before"].append(idx)
                        
                for idx in self.screen_after_index.get(img_name, []):
                    if idx not in seen_after:
                        seen_after.add(idx)
                        result["as_screen_after"].append(idx)
        else:
            # Exact match only.
            result["as_screen_before"] = self.screen_before_index.get(image_name, []).copy()
            result["as_screen_after"] = self.screen_after_index.get(image_name, []).copy()
            result["similar_images"] = [(image_name, 1.0)] if image_name in self.all_images else []
        
        return result
    
    def get_sample_by_index(self, idx: int) -> dict:
        """Return sample details for a single index."""
        if 0 <= idx < len(self.data):
            return self.data[idx]
        return None
    
    def get_samples_by_indices(self, indices: List[int]) -> List[dict]:
        """Return sample details for multiple indices."""
        return [self.data[idx] for idx in indices if 0 <= idx < len(self.data)]


def create_index(
    data_path: str = "state_transfer_explore.json",
    screenshots_dir: str = "screenshots_transfer",
    cache_path: str = "screen_transition_cache.pkl",
    similarity_threshold: float = 0.95,
    compute_hashes: bool = True
) -> ScreenTransitionIndex:
    """
    Convenience helper for creating a screen-transition index.
    
    Args:
        data_path: Path to the JSON data file.
        screenshots_dir: Path to the screenshots directory.
        similarity_threshold: Similarity threshold in [0, 1]. The default 0.95 means 95% similarity.
        compute_hashes: Whether to compute image hashes. Needed on first use; later runs can load from cache.
        
    Returns:
        A `ScreenTransitionIndex` instance.
    """
    index = ScreenTransitionIndex(
        data_path=data_path,
        screenshots_dir=screenshots_dir,
        similarity_threshold=similarity_threshold,
        cache_path=cache_path
    )
    
    if compute_hashes:
        index.compute_all_hashes()
    
    return index


def query_screen_transitions(
    image_name: str,
    index: Optional[ScreenTransitionIndex] = None,
    use_similarity: bool = True
) -> Tuple[List[int], List[int]]:
    """
    Query transitions associated with the given image.
    
    Args:
        image_name: Image filename.
        index: Existing `ScreenTransitionIndex` instance. If None, a new one is created.
        use_similarity: Whether to use similarity matching.
        
    Returns:
        (as_screen_before_ids, as_screen_after_ids)
        - as_screen_before_ids: sample indices where this image is screen_before
        - as_screen_after_ids: sample indices where this image is screen_after
    """
    if index is None:
        index = create_index()
    
    result = index.query_transitions(image_name, use_similarity=use_similarity)
    return result["as_screen_before"], result["as_screen_after"]


# ==================== Example usage ====================

if __name__ == "__main__":
    # Create the index. The first run computes image hashes; later runs load them from cache.
    print("=" * 60)
    print("Create screen transition index")
    print("=" * 60)
    
    index = create_index(similarity_threshold=0.95)
    
    # Pick one example image.
    sample_image = list(index.all_images)[0]
    print(f"\nQuery example image: {sample_image}")
    print("-" * 60)
    
    # Query transition relations.
    result = index.query_transitions(sample_image, use_similarity=True)
    
    print(f"\nNumber of similar images: {len(result['similar_images'])}")
    if result['similar_images']:
        print("Similar images (top 5):")
        for img, sim in result['similar_images'][:5]:
            print(f"  - {img}: {sim:.4f}")
    
    print(f"\nSamples with this image as screen_before: {len(result['as_screen_before'])}")
    print(f"Samples with this image as screen_after: {len(result['as_screen_after'])}")
    
    # Show a few sample details.
    if result['as_screen_before']:
        print("\nExample sample where this image is screen_before:")
        sample = index.get_sample_by_index(result['as_screen_before'][0])
        print(f"  - task_id: {sample['task_id']}")
        print(f"  - app: {sample['app']}")
        print(f"  - action_type: {sample['action_type']}")
        print(f"  - screen_after: {sample['screen_after']}")
    
    if result['as_screen_after']:
        print("\nExample sample where this image is screen_after:")
        sample = index.get_sample_by_index(result['as_screen_after'][0])
        print(f"  - task_id: {sample['task_id']}")
        print(f"  - app: {sample['app']}")
        print(f"  - action_type: {sample['action_type']}")
        print(f"  - screen_before: {sample['screen_before']}")
    
    print("\n" + "=" * 60)
    print("Usage examples:")
    print("=" * 60)
