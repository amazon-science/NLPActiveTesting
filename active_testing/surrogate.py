from typing import List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier

from active_testing.active import NLPActiveTesting

class Surrogate(NLPActiveTesting):
    """
    Surrogate model-based test case selection strategy for active testing in NLP tasks.
    
    This strategy trains a lightweight surrogate model (e.g., Random Forest, SVM, or
    Gradient Boosting) on text embeddings to estimate prediction uncertainty. Samples
    with higher uncertainty according to the surrogate model are prioritized for testing,
    as they are more likely to reveal model errors or edge cases.
    
    The surrogate model serves as a proxy for estimating which samples the target model
    is most uncertain about, without requiring direct access to the target model's
    internal confidence scores. This approach is particularly useful for black-box
    model evaluation.
    
    Key Features:
        - Multiple surrogate model options: Random Forest, SVM, Gradient Boosting.
        - Calibrated probability estimates for reliable uncertainty quantification.
        - Support for both single-label and multi-label classification tasks.
        - Entropy-based uncertainty scoring.
        - Optional language prior weighting for multilingual datasets.
        - Optional autoeval prior integration for incorporating external knowledge.
    
    Surrogate Model Options:
        - RF (Random Forest): Ensemble of decision trees with calibrated probabilities.
          Good balance of accuracy and uncertainty estimation.
        - SVM (Support Vector Machine): RBF kernel SVM with Platt scaling for
          probability calibration. Effective for high-dimensional embeddings.
        - GB (Gradient Boosting): Sequential ensemble method with strong predictive
          performance.
    
    Attributes:
        name (str): Strategy identifier, set to 'Surrogate'.
        acquisition (str): Type of surrogate model ('RF', 'SVM', or 'GB').
        surrogate_model: Trained surrogate classifier instance.
        is_multilabel (bool): Flag indicating if the task is multi-label classification.
        mlb (MultiLabelBinarizer): Binarizer for multi-label targets (if applicable).
    """
    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 budget: int,
                 classes: dict,
                 pipeline_name: str = "sentiment-analysis",
                 batch_size: int = 32,
                 model_name: str = "bert-base-multilingual-uncased",
                 max_length: int = 512,
                 device: str = None,
                 acquisition: str = "RF",
                 **kwargs):
        """
        Initialize the Surrogate model-based active testing strategy.
        
        Sets up the surrogate model for uncertainty-based sample selection. The
        surrogate model is trained on text embeddings to estimate prediction
        uncertainty, which guides the selection of informative test cases.

        Args:
            texts (List[str]): List of input text samples to select from.
                Each element can be a string or structured input for specific tasks.
            labels (List[int] or List[List[int]]): Corresponding ground truth labels.
                For single-label tasks: List of integer class indices.
                For multi-label tasks: List of lists containing multiple label indices.
            budget (int): Maximum number of samples to consider from the dataset.
                If None, all samples are used.
            classes (Dict[str, int]): Dictionary mapping class names to integer indices.
            pipeline_name (str or List[str], optional): Name of the NLP pipeline or
                a two-element list [pipeline_type, dataset_name]. Defaults to
                "sentiment-analysis".
            batch_size (int, optional): Number of samples to process simultaneously
                during embedding computation. Defaults to 32.
            model_name (str, optional): HuggingFace model identifier for the
                transformer model used for embeddings. Defaults to
                "bert-base-multilingual-uncased".
            max_length (int, optional): Maximum token sequence length for truncation.
                Defaults to 512.
            device (str, optional): Computing device ('cuda', 'cpu', or specific GPU).
                If None, automatically selects CUDA if available. Defaults to None.
            acquisition (str, optional): Type of surrogate model to use. Options:
                - 'RF': Random Forest classifier (default).
                - 'SVM': Support Vector Machine with RBF kernel.
                - 'GB': Gradient Boosting classifier.
                Defaults to "RF".
            **kwargs: Additional keyword arguments passed to the parent class.
            """
        self.acquisition = acquisition
        self.name = 'Surrogate'
        self.surrogate_model = None

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
        
        if isinstance(self.labels[0], (list, np.ndarray)):
            self.is_multilabel = True
        else:
            self.is_multilabel = False
            
        self.init_surrogate_model()

    def init_surrogate_model(self):
        """Initialize the surrogate model with probability calibration.
        
        Creates and configures the surrogate classifier based on the specified
        acquisition type. Applies probability calibration using cross-validation
        to ensure reliable uncertainty estimates. For multi-label tasks, wraps
        the base model in a MultiOutputClassifier.
        
        Side Effects:
            Sets self.surrogate_model to the configured classifier instance.

        Model Configurations:
            - RF (Random Forest):
                - n_estimators: 300 trees
                - max_depth: None (nodes expanded until all leaves are pure)
                - Wrapped with CalibratedClassifierCV for probability calibration
            - SVM (Support Vector Machine):
                - kernel: RBF (Radial Basis Function)
                - C: 1.0 (regularization parameter)
                - probability: True (enables Platt scaling)
                - Wrapped with CalibratedClassifierCV for improved calibration
            - GB (Gradient Boosting):
                - n_estimators: 300 boosting stages
                - max_depth: 5 (maximum depth of individual trees)
                - Note: GB has built-in probability estimates, no additional calibration
        """
        cv = 2
            
        if self.acquisition == "RF":
            base_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
            )
            self.surrogate_model = CalibratedClassifierCV(base_model, cv=cv)
        elif self.acquisition == "SVM":
            base_model = SVC(kernel='rbf', C=1.0, probability=True)
            self.surrogate_model = CalibratedClassifierCV(base_model, cv=cv)
        elif self.acquisition == "GB":
            base_model = GradientBoostingClassifier(n_estimators=300, max_depth=5)
            
        else:
            raise ValueError(f"acquisition {self.acquisition} not recognized!")
        
        if self.is_multilabel:
            self.surrogate_model = MultiOutputClassifier(base_model)
        else:
            self.surrogate_model = CalibratedClassifierCV(base_model, cv=cv)

    def compute_entropy(self, probs) -> np.ndarray:
        """
        Compute entropy-based uncertainty scores from predicted probabilities.
        
        Calculates the Shannon entropy of the predicted class probability
        distributions. Higher entropy indicates greater uncertainty about the
        predicted class, suggesting the sample may be more informative for testing.
        
        Args:
            probs (np.ndarray or List[np.ndarray]): Predicted class probabilities.
                - For single-label tasks: Array of shape (n_samples, n_classes)
                  where each row sums to 1.
                - For multi-label tasks: List of arrays, one per label, each of
                  shape (n_samples, 2) representing binary probabilities for
                  each label.
            
        Returns:
            np.ndarray: Entropy scores for each sample. Shape: (n_samples,).
                Higher values indicate greater uncertainty.
                - For single-label: Entropy computed over all classes.
                - For multi-label: Average entropy across all labels.

        Formula:
            H(p) = -Σ p_i * log(p_i)
            
            Where p_i is the probability of class i. A small epsilon (1e-6) is
            added to prevent log(0).
        """
        epsilon = 1e-6
        
        if self.is_multilabel:
           
            n_samples = probs[0].shape[0]
            total_entropy = np.zeros(n_samples)
            
            for prob_array in probs:
                prob_array = np.array(prob_array)
                entropy = -np.sum(prob_array * np.log(prob_array + epsilon), axis=1)
                total_entropy += entropy
            
            return total_entropy / len(probs)
        else:
            probs = np.array(probs)
            return -np.sum(probs * np.log(probs + epsilon), axis=1)
        
    def select_next_test_case(
            self,
            num_samples: int = -1,
            autoeval_prior: np.ndarray = None,
            lang_prior: Tuple[float, float] = None
    ) -> Tuple[List[str], List[int], np.ndarray]:
            """
            Select test cases based on uncertainty estimated by the surrogate model.
        
            Trains the surrogate model on text embeddings, computes entropy-based
            uncertainty scores, and selects samples with the highest uncertainty.
            Supports optional priors for incorporating external knowledge or
            language-based weighting.

            Args:
                num_samples (int, optional): Number of samples to select.
                    - If -1, selects all available samples.
                    - If positive, selects the specified number of samples.
                    Defaults to -1.
                autoeval_prior (np.ndarray, optional): Prior knowledge array to influence
                    selection. Shape: (n_samples,). Higher values increase selection
                    probability. Applied multiplicatively to uncertainty scores.
                    Defaults to None.
                lang_prior (Tuple[float, float], optional): Tuple of probability weights
                    (p_lang1, p_lang2) for the first and second halves of the dataset.
                    Used for language-balanced sampling in multilingual settings.
                    Applied multiplicatively to uncertainty scores. Defaults to None.

            Returns:
                Tuple[List[int], List[int], np.ndarray]: A tuple containing:
                    - predictions (List[int]): Model predictions for the selected samples.
                    - selected_indices (np.ndarray): Indices of the chosen samples,
                      sorted by uncertainty score in ascending order (highest uncertainty
                      samples are at the end of the array).
                    - scores (np.ndarray): Normalized uncertainty scores for all samples.
                      Shape: (n_texts,). Incorporates all applied priors and sums to 1.
            """
            num_samples = min(len(self.texts) if num_samples == -1 else num_samples, len(self.texts))
            n_texts = len(self.texts)
            half = n_texts // 2

            if num_samples == len(self.texts):
                selected_indices = np.arange(n_texts)
                predictions = self.extract_predictions(selected_indices)
                return predictions, selected_indices, np.ones(n_texts) / n_texts

            # Extract embeddings
            embeddings = self.get_embeddings(self.texts, plot_and_save=False)
            if self.is_multilabel:
                label_sets = [list(set(seq)) for seq in self.labels]
                self.mlb = MultiLabelBinarizer()
                labels = self.mlb.fit_transform(label_sets)
            else:
                labels = self.labels
                            
            self.surrogate_model.fit(embeddings, labels)

            # Predict probabilities and compute entropy
            probs = self.surrogate_model.predict_proba(embeddings)
            scores = self.compute_entropy(probs)
            
            scores = np.maximum(scores, 1e-3)  # avoid zero scores

            if autoeval_prior is not None:
                scores = self.apply_autoeval_prior(scores, autoeval_prior)

            # Apply language prior if provided
            if lang_prior is not None:
                lang_weights = np.ones_like(scores)
                for i in range(n_texts):
                    if i < half:
                        lang_weights[i] *= lang_prior[0]
                    else:
                        lang_weights[i] *= lang_prior[1]
                scores = scores * lang_weights

            scores /= scores.sum()

            # Select samples with highest uncertainty
            selected_indices = np.argsort(scores)[-num_samples:]
            predictions = self.extract_predictions(selected_indices)

            return predictions, selected_indices, scores
