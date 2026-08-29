from typing import List, Tuple
import random
import numpy as np
from active_testing.active import NLPActiveTesting

class Random(NLPActiveTesting):
    """
    Random test case selection strategy for active testing in NLP tasks.
    
    This strategy implements uniform random sampling from the available pool of
    test cases. It serves as a baseline method for comparison with more sophisticated
    selection strategies and ensures unbiased sampling when no prior information
    about sample informativeness is available.
    
    The strategy supports optional language-based weighting for multilingual datasets,
    allowing controlled sampling proportions across different language segments of
    the data.
    
    Key Features:
        - Uniform random sampling without replacement.
        - Prevents duplicate selection of the same sample.
        - Optional language prior weighting for multilingual datasets.
        - Constant time complexity per selection (O(1) amortized).
    """

    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 budget: int,
                 classes: dict,
                 **kwargs):
        """
        Initialize the Random testing strategy.
        
        Sets up the random selection mechanism with the provided dataset and
        configuration parameters.

        Args:
            texts (List[str]): List of input text samples to select from.
                Each element can be a string or a tuple (for QA tasks).
            labels (List[int]): Corresponding ground truth labels for the texts.
                Length must match the length of texts.
            budget (int): Maximum number of samples to consider from the dataset.
                If None, all samples are used.
            classes (Dict[str, int]): Dictionary mapping class names to integer
                indices. Used for multi-class classification tasks.
            **kwargs: Additional keyword arguments passed to the parent class,
                including:
                - pipeline_name (str or List[str]): Name of the NLP pipeline.
                - batch_size (int): Processing batch size.
                - model_name (str): Pretrained model identifier.
                - max_length (int): Maximum token sequence length.
                - device (str): Computing device specification.
                - predictor_name (str): Name of the predictor model.
                - selected_indices (List[int]): Pre-selected sample indices.
                - all_predictions (List[int]): Pre-computed predictions.
        """
        self.name = 'Random'
        super().__init__(texts=texts,
                         labels=labels,
                         budget=budget,
                         classes=classes,
                         **kwargs)

    def select_next_test_case(
        self,
        num_samples: int = -1,
        lang_prior: Tuple[float, float] = None,
    ) -> Tuple[List, List[int], np.ndarray]:
        """
        Select the next batch of test cases using random sampling.
        
        Performs uniform random sampling from the pool of available (not yet selected)
        samples. Supports optional language-based weighting for multilingual datasets
        where the first half contains one language and the second half contains another.

        Args:
            num_samples (int, optional): Number of samples to select.
                - If -1, selects all available (non-previously-selected) samples.
                - If positive, selects min(num_samples, available_samples) samples.
                Defaults to -1.
            lang_prior (Tuple[float, float], optional): Tuple of probability weights
                (p_lang1, p_lang2) for sampling from the first and second halves
                of the dataset, respectively. Used for language-balanced sampling
                in multilingual settings.
                - If None, uses uniform random sampling (each available sample has
                  equal probability).
                - If provided, samples are selected with probabilities proportional
                  to the language weights.
                Defaults to None.

        Returns:
            Tuple[List[int], List[int], np.ndarray]: A tuple containing:
                - predictions (List[int]): Model predictions for the selected samples.
                    Empty list if no samples are available.
                - selected_indices (List[int]): Indices of the chosen samples in the
                    original dataset. Empty list if no samples are available.
                - scores (np.ndarray): Sampling probability distribution over all
                    samples. Shape: (n_texts,).
                    - For uniform sampling: All values are 1/n_texts.
                    - For language-weighted sampling: Normalized probabilities based
                      on lang_prior weights.
        """
        n_texts = len(self.texts)
        half = n_texts // 2

        available_indices = [i for i in range(n_texts) if i not in self.selected_indices]

        if len(available_indices) == 0:
            print("Warning! Available indices not found!")
            return [], [], np.ones(n_texts) / n_texts

        # Number of samples to select
        n_samples = min(len(available_indices) if num_samples == -1 else num_samples,
                        len(available_indices))

        if lang_prior is None:
            # === Original uniform random selection ===
            selected_indices = random.sample(available_indices, n_samples)
            predictions = self.extract_predictions(selected_indices)
            return predictions, selected_indices, np.ones(n_texts) / n_texts

        # === Language-aware sampling ===
        probs = np.zeros(n_texts)
        for i in available_indices:
            if i < half:  # first language
                probs[i] = lang_prior[0]
            else:         # second language
                probs[i] = lang_prior[1]

        # Normalize to sum to 1 over available indices
        probs = probs / probs.sum()

        # Weighted random selection
        selected_indices = np.random.choice(
            available_indices, size=n_samples, replace=False, p=probs[available_indices]
        )

        predictions = self.extract_predictions(selected_indices)

        return predictions, list(selected_indices), probs
