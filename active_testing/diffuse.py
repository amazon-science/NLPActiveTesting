import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from active_testing.active import NLPActiveTesting
from typing import List, Tuple


def safe_softmax(x: np.ndarray, temperature: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """
    Numerically stable softmax that ensures strictly positive outputs.
    temperature > 0 controls sharpness (smaller -> sharper).
    eps is added to avoid exact zeros under pathological underflow.
    """
    # subtract max for numeric stability
    x_scaled = x / max(temperature, 1e-12)
    x_shift = x_scaled - np.max(x_scaled)
    exp_x = np.exp(x_shift)
    probs = exp_x / (np.sum(exp_x) + eps)
    # clip tiny values up to eps to avoid absolute zeros
    probs = np.clip(probs, eps, None)
    # re-normalize so sum == 1
    probs = probs / probs.sum()
    return probs


class Diffuse(NLPActiveTesting):
    """
    Text-only DiffUse / IRT-inspired active testing.

    - Difficulty: distance from dataset centroid (rarity)
    - Discrimination: local variance of neighbor distances (diversity)
    - Utility = Difficulty × Discrimination (rescaled)
    This version guarantees no-zero utilities and returns a robust probability-like score
    via a numerically stable softmax.
    """

    def __init__(self, texts: List[str], labels: List[int], budget: int, classes,
                 model_name: str = "all-MiniLM-L6-v2", device: str = None, **kwargs):
        self.name = 'Diffuse'
        super().__init__(texts=texts, labels=labels, budget=budget, classes=classes,
                         model_name=model_name, device=device, **kwargs)

        self.encoder = SentenceTransformer(model_name, device=device)

    def compute_item_parameters(self, k_neighbors: int = 10):
        """
        Compute embeddings and pseudo-IRT parameters:
          - diff: normalized distance from global centroid (range 0..1)
          - disc: normalized local variance in neighbor distances (range 0..1)
        """
        embeddings = self.get_embeddings(self.texts)#encoder.encode(self.texts, show_progress_bar=True)

        # centroid distance -> difficulty signal
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        distances = cosine_distances(embeddings, centroid).flatten()
        diff = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

        # local neighbor variance -> discrimination signal
        nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(self.texts))).fit(embeddings)
        distances_nn, _ = nn.kneighbors(embeddings)
        # discard self-distance (col 0)
        if distances_nn.shape[1] > 1:
            local_var = distances_nn[:, 1:].var(axis=1)
        else:
            # fallback: if no neighbors, set small constant var
            local_var = np.zeros(len(self.texts))

        disc = (local_var - local_var.min()) / (local_var.max() - local_var.min() + 1e-8)

        # Optional non-linear scaling to increase contrast (tunable)
        disc = np.power(disc, 0.5)  # gamma = 0.5 reduces compression near zero

        self.item_params = {"diff": diff, "disc": disc}
        return diff, disc

    def compute_utility(self, mode: str = "diffdisc", eps_floor: float = 1e-6):
        """
        Combine diff and disc into a utility. Ensure a positive floor (eps_floor)
        so utilities are never exactly zero.
        """
        diff = self.item_params["diff"]
        disc = self.item_params["disc"]

        if mode == "diff":
            utility = diff
        elif mode == "disc":
            utility = disc
        elif mode == "diffdisc":
            utility = diff * disc
        else:
            utility = np.sqrt(diff**2 + disc**2)

        utility = np.maximum(utility, eps_floor)

        utility = (utility - utility.min()) / (utility.max() - utility.min() + 1e-12)

        return utility

    def select_next_test_case(self, num_samples: int = -1, temperature: float = 1.0,
                              k_neighbors: int = 10, sharpen: float = 1.0) -> Tuple[List, List[int], np.ndarray]:
        """
        Main selection method.

        Args:
            num_samples: how many examples to select; -1 means self.budget
            temperature: softmax temperature (smaller -> sharper distribution)
            k_neighbors: neighbors for discrimination calculation
            sharpen: exponent to further emphasize top utilities (>=1.0)
        Returns:
            predictions, selected_indices, selection_scores (probabilities)
        """
        k = num_samples if num_samples > 0 else min(self.budget, len(self.texts))

        # 1) compute item-level parameters
        self.compute_item_parameters(k_neighbors=k_neighbors)

        # 2) compute utility and ensure no-zero values
        utilities = self.compute_utility(mode="diffdisc", eps_floor=1e-8)

        # 3) optional sharpening (emphasize top items)
        if sharpen != 1.0:
            utilities = np.power(utilities, sharpen)

        # 4) convert utilities to a stable positive distribution using softmax
        scores = safe_softmax(utilities, temperature=temperature, eps=1e-12)

        # 5) select top-k by score
        selected_indices = list(np.argsort(-scores)[:k])

        # 6) get model predictions (your existing extraction)
        predictions = self.extract_predictions(selected_indices)

        return predictions, selected_indices, scores
