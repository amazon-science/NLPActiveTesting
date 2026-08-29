from typing import List
import numpy as np
from active_testing.active import NLPActiveTesting

class Distance(NLPActiveTesting):
    """
    Distance-based test case selection strategy.
    Selects test cases based on their distances in the embedding space.
    """

    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 budget: int,
                 classes: dict,
                 **kwargs):
        self.name = 'Distance'
        super().__init__(texts=texts,
                         labels=labels,
                         budget=budget,
                         classes=classes,
                         **kwargs)
        self.selected_indices = []

    def select_next_test_case(
        self,
        num_samples: int = -1,
    ) -> tuple:
        """
        Select next batch of test cases based on distance metrics,
        excluding already-selected indices.
        """
        # Determine number of samples to select
        n_samples = min(len(self.texts) if num_samples == -1 else num_samples,
                        len(self.texts))

        # Get embeddings for all texts
        embeddings = self.get_embeddings(self.texts)

        # Randomly select initial points (just one for simplicity)
        initial_indices = np.random.choice(n_samples, size=1, replace=False)

        # Calculate distances between all points and initial points
        d = embeddings[:, np.newaxis, :] - embeddings[initial_indices]
        d = np.sqrt(np.sum(d ** 2, axis=-1))
        distances = d.mean(1)

        # Normalize distances
        distances /= distances.max()

        # Convert distances to probabilities via softmax
        pmf = np.exp(distances)
        pmf /= pmf.sum()

        # Exclude already-selected indices
        available_indices = [i for i in range(len(self.texts)) if i not in self.selected_indices]
        if len(available_indices) == 0:
            print(f"Warning! Available indices not found!")
            return [], [], pmf

        n_samples = min(n_samples, len(available_indices))

        masked_pmf = np.array([pmf[i] if i in available_indices else 0.0 for i in range(len(self.texts))])
        masked_pmf /= masked_pmf.sum()  # renormalize

        # Sample points according to probability distribution
        selected_indices = np.random.choice(len(self.texts), size=n_samples, p=masked_pmf, replace=False).tolist()

        # Update selected indices
        self.selected_indices.extend(selected_indices)

        # Get predictions for selected samples
        predictions = self.extract_predictions(selected_indices)

        return predictions, selected_indices, masked_pmf