from typing import List, Tuple
import numpy as np
import torch
import math
import torch.nn.functional as F

from active_testing.active import NLPActiveTesting

class Uncertainty(NLPActiveTesting):
    """
    Uncertainty-based test case selection strategy for active testing in NLP tasks.
    
    This strategy leverages Monte Carlo (MC) Dropout to estimate model uncertainty
    and prioritizes test cases where the model exhibits high uncertainty. High
    uncertainty samples are more likely to reveal model errors, edge cases, or
    distribution shifts, making them valuable for comprehensive model evaluation.
    
    Monte Carlo Dropout works by performing multiple forward passes through the
    model with dropout enabled at inference time. The variation in predictions
    across these passes provides an estimate of model uncertainty without
    requiring ensemble models or Bayesian neural networks.
    
    Acquisition Methods:
        - mutual_information: Measures the difference between the entropy of the
          mean prediction and the mean entropy of individual predictions. High
          mutual information indicates the model is uncertain about which class
          to predict (epistemic uncertainty).
        - gaussian_prior: Computes the standard deviation of embeddings across
          MC Dropout samples. High variance in the embedding space indicates
          the model's internal representations are unstable for that input.
    
    Key Features:
        - Model-agnostic uncertainty estimation via MC Dropout.
        - Two complementary uncertainty metrics for different use cases.
        - Support for both single-label and token-level (NER) tasks.
        - Optional language prior weighting for multilingual datasets.
        - Probabilistic sampling based on uncertainty scores.
    
    Attributes:
        name (str): Strategy identifier, defaults to 'Uncertainty'.
        acquisition (str): Uncertainty estimation method ('mutual_information' or
            'gaussian_prior').    """

    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 budget: int,
                 classes: dict,
                 acquisition: str = 'mutual_information',
                 pipeline_name: str = "sentiment-analysis",
                 batch_size: int = 8,
                 model_name: str = "bert-base-multilingual-uncased",
                 max_length: int = 512,
                 device: str = None,
                 name = None,
                 **kwargs):
        """
        Sets up the uncertainty estimation mechanism using Monte Carlo Dropout
        with the specified acquisition function for computing uncertainty scores.

        Args:
            texts (List[str]): List of input text samples to select from.
                For NER tasks, can be lists of tokens that will be joined.
            labels (List[int]): Corresponding ground truth labels for the texts.
                Length must match the length of texts.
            budget (int): Maximum number of samples to consider from the dataset.
                If None, all samples are used.
            classes (Dict[str, int]): Dictionary mapping class names to integer
                indices. Used for determining the number of output classes.
            acquisition (str, optional): Uncertainty estimation method. Options:
                - 'mutual_information': Information-theoretic uncertainty based on
                  the gap between predictive entropy and expected entropy. Best for
                  identifying samples where the model is confused between classes.
                - 'gaussian_prior': Embedding-space variance across MC samples.
                  Best for identifying samples with unstable internal representations.
                Defaults to 'mutual_information'.
            pipeline_name (str or List[str], optional): Name of the NLP pipeline or
                a two-element list [pipeline_type, dataset_name]. Use "ner" as the
                first element for named entity recognition tasks. Defaults to
                "sentiment-analysis".
            batch_size (int, optional): Number of samples to process simultaneously.
                Note: Current implementation processes all texts at once for MC
                Dropout efficiency. Defaults to 8.
            model_name (str, optional): HuggingFace model identifier for the
                transformer model. Must have dropout layers for MC Dropout to work.
                Defaults to "bert-base-multilingual-uncased".
            max_length (int, optional): Maximum token sequence length for truncation.
                Note: Current implementation uses max_length=16 in predict() for
                efficiency. Defaults to 512.
            device (str, optional): Computing device ('cuda', 'cpu', or specific GPU).
                If None, automatically selects CUDA if available. Defaults to None.
            name (str, optional): Custom name for this strategy instance. If None,
                defaults to 'Uncertainty'. Useful for distinguishing between
                different acquisition functions in experiments. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        if name is None:
            self.name = 'Uncertainty'
        else:
            self.name = name
        self.acquisition = acquisition

        super().__init__(texts=texts,
                         labels=labels,
                         budget=budget,
                         classes=classes,
                         pipeline_name=pipeline_name,
                         batch_size=batch_size,
                         model_name=model_name,
                         max_length=max_length,
                         device=device,
                         **kwargs)

    def predict(self) -> Tuple[np.ndarray, np.ndarray] or np.ndarray:
        """
        Perform predictions using Monte Carlo Dropout for uncertainty estimation.
        
        Executes multiple forward passes (10 by default) through the model with
        dropout enabled to obtain a distribution of predictions. The variation
        in these predictions is used to estimate model uncertainty.
        
        For the 'gaussian_prior' acquisition, extracts embeddings and computes
        their mean and standard deviation across MC samples. For 'mutual_information',
        computes class probabilities and calculates the mutual information between
        the input and the predicted class.

        Returns:
            Tuple[np.ndarray, np.ndarray] or np.ndarray: Depends on acquisition method:
                - If acquisition is 'gaussian_prior':
                    Returns (mean_embeddings, std_embeddings) where:
                    - mean_embeddings: Shape (n_samples, embedding_dim), mean of
                      embeddings across MC samples.
                    - std_embeddings: Shape (n_samples, embedding_dim), standard
                      deviation of embeddings across MC samples.
                - If acquisition is 'mutual_information':
                    Returns mutual_information array of shape (n_samples,) containing
                    the mutual information score for each sample.

        Algorithm:
            1. Tokenize all input texts with padding and truncation.
            2. Set model to training mode (enables dropout).
            3. Perform 10 forward passes with dropout active.
            4. For 'gaussian_prior':
               - Extract [CLS] token or mean-pooled embeddings.
               - Compute mean and std across MC samples.
            5. For 'mutual_information':
               - Compute softmax probabilities for each MC sample.
               - Calculate entropy of mean probabilities: H(E[p]).
               - Calculate mean entropy of individual samples: E[H(p)].
               - Mutual Information = H(E[p]) - E[H(p)].
        """
        predictions = []

        # For NER tasks, join tokens back into string sentences
        if self.pipeline_name[0] == "ner":
            self.texts = [" ".join(sentence) for sentence in self.texts]

        # Tokenize input texts to tensor format, padding and truncating as needed
        inputs = self.tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt"
        ).to(self.device)

        # Get base model (e.g., bert model inside the pipeline)
        if hasattr(self.model, self.model.config.model_type):
            base_model = getattr(self.model, self.model.config.model_type)
        else:
            base_model = self.model.model
            
        base_model.train()
        # Run multiple forward passes with dropout enabled (MC Dropout)
        for _ in range(10):
            with torch.no_grad():  # No gradients, reduces memory usage; dropout still active since model.train()
                outputs = base_model(**inputs)

            if self.acquisition == "gaussian_prior":
                # Extract embeddings by mean pooling last hidden states
                last_hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                emb = (last_hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
                predictions.append(emb)

            elif self.acquisition == "mutual_information":
                # Compute class probabilities via softmax on logits
                logits = getattr(outputs, "logits", self.model(**inputs).logits)
                probs = F.softmax(logits, dim=-1).detach()
                predictions.append(probs)

        predictions = torch.stack(predictions)  # Shape: [num_MC_samples, batch_size, dim or num_classes]
        mean_pred = predictions.mean(dim=0)

        if self.acquisition == "gaussian_prior":
            std_pred = predictions.std(dim=0)
            # Return mean and std embeddings as numpy arrays
            return mean_pred.cpu().numpy(), std_pred.cpu().numpy()

        elif self.acquisition == "mutual_information":
            # Calculate mutual information as difference of entropies
            mean_probs = mean_pred  # shape: [batch_size, num_classes]

            # Entropy of mean distribution
            entropy_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12), dim=1)

            # Average entropy of individual MC sample distributions
            entropy_samples = -torch.sum(
                predictions * torch.log(predictions + 1e-12), dim=2
            ).mean(dim=0)

            mutual_information = entropy_mean - entropy_samples

            # Clamp to zero to avoid negative values from numerical errors
            mutual_information = torch.clamp(mutual_information, min=0)

            # Return mutual information per sample as numpy array
            return mutual_information.cpu().numpy()

        else:
            raise ValueError(f"Acquisition {self.acquisition} not recognized.")

    def expected_loss(self) -> np.ndarray:
        """
        Calculate expected loss (uncertainty scores) for all samples.
        
        Computes uncertainty scores based on the configured acquisition method.
        These scores quantify how uncertain the model is about each sample,
        with higher scores indicating greater uncertainty.

        Returns:
            np.ndarray: Array of uncertainty scores for each sample.
                Shape: (n_samples,). Higher values indicate greater uncertainty.
                - For 'gaussian_prior': Sum of squared standard deviations across
                  embedding dimensions, representing total variance in embedding space.
                - For 'mutual_information': Mutual information between input and
                  predicted class, measuring epistemic uncertainty.

        Raises:
            NotImplementedError: If self.acquisition is not one of the supported
                methods ('gaussian_prior' or 'mutual_information').

        Interpretation:
            - gaussian_prior: High scores indicate the model's internal representation
              of the input is unstable across dropout samples. This may indicate
              out-of-distribution samples or inputs the model hasn't learned well.
            - mutual_information: High scores indicate the model is uncertain about
              which class to predict. This captures epistemic (model) uncertainty
              rather than aleatoric (data) uncertainty.
        """
        if self.acquisition == 'gaussian_prior':
            mu, std = self.predict()
            # Use squared sum of std deviations as uncertainty score
            return std.sum(axis=-1) ** 2
        elif self.acquisition == 'mutual_information':
            # Return mutual information uncertainty scores
            return self.predict()
        else:
            raise NotImplementedError(f"Acquisition method {self.acquisition} not recognized.")

    def select_next_test_case(self, 
                            num_samples: int = -1,
                            lang_prior: Tuple[float, float] = None
                            ) -> Tuple[List[str], List[int], np.ndarray]:
        """
        Select next test cases based on uncertainty scores with optional language weighting.
        
        Computes uncertainty scores for all samples using Monte Carlo Dropout,
        then performs probabilistic sampling weighted by these uncertainty scores.
        Samples with higher uncertainty have higher probability of being selected.
        
        Supports optional language prior weighting for multilingual datasets where
        different languages occupy different portions of the dataset.

        Args:
            num_samples (int, optional): Number of samples to select.
                - If -1, selects all available (non-previously-selected) samples.
                - If positive, selects min(num_samples, available_samples) samples.
                Defaults to -1.
            lang_prior (Tuple[float, float], optional): Tuple of probability weights
                (p_lang1, p_lang2) for the first and second halves of the dataset.
                Used for language-balanced sampling in multilingual settings.
                - If None, uses pure uncertainty-based selection.
                - If provided, uncertainty scores are multiplied by language weights
                  before normalization.
                Defaults to None.

        Returns:
            Tuple[List[int], List[int], np.ndarray]: A tuple containing:
                - predictions (List[int]): Model predictions for the selected samples.
                    Empty list if no samples are available.
                - selected_indices (List[int]): Indices of the chosen samples.
                    Empty list if no samples are available.
                - scores (np.ndarray): Normalized uncertainty scores for all samples
                    (before applying language prior masking). Shape: (n_texts,).
                    These are the base uncertainty scores that can be used for
                    importance weighting in unbiased estimation.
        """

        n_texts = len(self.texts)
        half = n_texts // 2

        # Compute uncertainty scores for all samples
        expected_loss = self.expected_loss()
        expected_loss = np.maximum(expected_loss, 1e-12)  # avoid zeros
        expected_loss /= expected_loss.sum()  # normalize

        # Exclude already selected indices
        available_indices = [i for i in range(n_texts) if i not in self.selected_indices]

        if len(available_indices) == 0:
            print("Warning! Available indices not found!")
            return [], [], expected_loss

        # Number of samples to select
        n_samples = min(len(available_indices) if num_samples == -1 else num_samples,
                        len(available_indices))

        # Apply language prior if provided
        masked_probs = np.zeros_like(expected_loss)
        for i in available_indices:
            weight = 1.0
            if lang_prior is not None:
                weight = lang_prior[0] if i < half else lang_prior[1]
            masked_probs[i] = expected_loss[i] * weight

        # Normalize masked probabilities
        masked_probs /= masked_probs.sum()

        # Sample indices from the weighted available pool
        selected_indices = np.random.choice(
            n_texts,
            size=n_samples,
            p=masked_probs,
            replace=False
        ).tolist()

        # Get predictions for selected samples
        predictions = self.extract_predictions(selected_indices)

        return predictions, selected_indices, expected_loss
