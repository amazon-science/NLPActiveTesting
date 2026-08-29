from typing import List, Tuple
import random
import numpy as np
from active_testing.active import NLPActiveTesting

from data import get_class_distribution

class Stratified(NLPActiveTesting):
    """
    Random test case selection strategy.
    
    This strategy randomly selects test cases from the available pool.
    It ensures that the same index is never selected more than once.
    """

    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 budget: int,
                 classes: dict,
                 **kwargs):
        """
        Initialize Random testing strategy.

        Args:
            texts: List of input texts to select from
            labels: Corresponding labels for texts
            budget: Ratio for splitting data
            classes: Dictionary mapping class names to indices
            **kwargs: Additional arguments passed to parent class
        """
        self.name = 'Stratified'
        super().__init__(texts=texts,
                         labels=labels,
                         budget=budget,
                         classes=classes,
                         **kwargs)
        if isinstance(self.labels[0], (list, np.ndarray)):
            self.is_multilabel = True
        else:
            self.is_multilabel = False

    def select_next_test_case(
    self,
    num_samples: int = -1,
    lang_prior: Tuple[float, float] = None,
) -> Tuple[List, List[int], np.ndarray]:
        """
        Returns:
            predictions: model predictions
            selected_indices: chosen indices
            scores: global sampling distribution over ALL samples (sum = 1),
                    where ALL entries have positive probability,
                    even those already selected.
        """

        try:
            labels = np.array(self.labels)
        except ValueError:
            labels = self.labels
        n_texts = len(self.texts)

        available = np.array([i for i in range(n_texts) if i not in self.selected_indices])

        if len(available) == 0:
            print("Warning! No available indices.")
            scores = np.ones(n_texts) / n_texts
            return [], [], scores

        n_samples = len(available) if num_samples == -1 else min(num_samples, len(available))

        if lang_prior is not None:
            half = n_texts // 2
            p1, p2 = lang_prior

            # Global scores (everyone has non-zero probability)
            scores = np.array([p1 if i < half else p2 for i in range(n_texts)], dtype=float)
            scores /= scores.sum()

            # Sampling distribution restricted to available
            masked_probs = scores[available]
            masked_probs /= masked_probs.sum()

            selected = np.random.choice(
                available, size=n_samples, replace=False, p=masked_probs
            )

            predictions = self.extract_predictions(selected)
            return predictions, list(map(int, selected)), scores


        if not self.is_multilabel:
            class_distribution = get_class_distribution(self.pipeline_name[1])
            
            # Convert class_distribution to "mass" per class (percentage / 100)
            class_mass = {
                cls: class_distribution[cls]["percentage"] / 100.0
                for cls in class_distribution
            }
            
            # Group indices by class (ALL indices, not only available)
            class_to_indices = {
                cls: np.where(labels == cls)[0]
                for cls in class_distribution.keys()
            }
            
            scores = np.zeros(n_texts)

            for cls, idxs in class_to_indices.items():
                if len(idxs) == 0:
                    continue

                # each item in class gets its class mass divided equally
                per_item_prob = class_mass[cls] / len(idxs)
                scores[idxs] = per_item_prob

            # Ensure normalization (should already sum to 1)
            scores /= scores.sum()

            masked_probs = scores[available]
            masked_probs /= masked_probs.sum()
            
            selected_indices = np.random.choice(
                available, size=n_samples, replace=False, p=masked_probs
            )

            predictions = self.extract_predictions(selected_indices)

            return predictions, list(map(int, selected_indices)), scores
        else:
            class_distribution = get_class_distribution(self.pipeline_name[1])
    
            class_mass = {
                int(cls): class_distribution[cls]["percentage"] / 100.0
                for cls in class_distribution
            }
            
            dominant_labels = []
            for sentence_labels in self.labels:
                if len(sentence_labels) == 0:
                    dominant_labels.append(-1)
                else:
                    tag_counts = Counter([int(tag) for tag in sentence_labels])
                    dominant_tag = tag_counts.most_common(1)[0][0]
                    dominant_labels.append(dominant_tag)
            
            dominant_labels = np.array(dominant_labels)
            
            class_to_indices = {
                int(cls): np.where(dominant_labels == int(cls))[0]
                for cls in class_distribution.keys()
            }
            
            scores = np.zeros(n_texts)
            
            for cls, idxs in class_to_indices.items():
                if len(idxs) == 0:
                    continue
                per_item_prob = class_mass[cls] / len(idxs)
                scores[idxs] = per_item_prob

            if scores.sum() == 0:
                scores = np.ones(n_texts) / n_texts
            else:
                scores /= scores.sum()

            masked_probs = scores[available]
            if masked_probs.sum() == 0:
                masked_probs = np.ones(len(available)) / len(available)
            else:
                masked_probs /= masked_probs.sum()

            selected_indices = np.random.choice(
                available, size=n_samples, replace=False, p=masked_probs
            )

            predictions = self.extract_predictions(selected_indices)

            return predictions, list(map(int, selected_indices)), scores
