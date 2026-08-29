import os.path
from typing import List, Union, Tuple, Optional, Dict
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from copy import deepcopy


from utils import plot_embeddings, model_name_map, get_complete_predictions, shuffle_data
from estimator import *

class NLPActiveTesting:
    """
    A class for active testing in NLP tasks.
    
    Implements various strategies for selecting test cases in NLP applications,
    including random sampling, stratified sampling, uncertainty-based selection,
    coverage-based selection, and more. Supports unbiased metric estimation
    using importance weighting techniques.

    Attributes:
        texts (List[str]): Collection of text samples to be processed.
        labels (List[int]): Corresponding labels for the text samples.
        tokenizer: Transformer tokenizer for text processing.
        model: Transformer model for sequence classification.
        embeddings_cache (Dict): Dictionary storing precomputed text embeddings.
        max_length (int): Maximum sequence length for tokenization.
        device (torch.device): Computing device (CPU/GPU).
        classes (Dict): Dictionary mapping class names to indices.
        num_labels (int): Number of unique class labels.
        pipeline_name (List): List containing pipeline type and dataset name.
        batch_size (int): Number of samples to process at once.
        predictor_name (str): Name of the predictor model used for predictions.
        budget (int): Maximum number of samples to consider.
        selected_indices (List[int]): Indices of samples selected for testing.
        all_predictions (np.ndarray): Array of model predictions for all samples.
        total_predictions (np.ndarray): Complete predictions before shuffling.
        embeddings (np.ndarray): Precomputed embeddings for all texts.
        problematic_samples (List): List of known problematic sample indices.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        classes: dict,
        pipeline_name: list,
        batch_size:int = 8,
        budget: int = None,
        model_name: str = "bert-base-multilingual-uncased",
        predictor_name : str = "claude",
        max_length: int = 512,
        device:str=None,
        problematic_samples:list=None,
        selected_indices:list = [],
        all_predictions:list = None,
        total_predictions:list = None,
        **kwargs
    ):
        """
        Initialize the NLP Active Testing system.

        Sets up the tokenizer, model, and prepares the data for active testing.
        Loads or generates embeddings and predictions as needed.

        Args:
            texts (List[str]): List of input text samples.
            labels (List[int]): List of corresponding ground truth labels.
            classes (Dict[str, int]): Dictionary mapping class names to integer indices.
            pipeline_name (List[str]): Two-element list containing [pipeline_type, dataset_name].
            batch_size (int, optional): Number of samples to process in each batch. Defaults to 8.
            budget (int, optional): Maximum number of samples to use. If None, uses all samples.
                Defaults to None.
            model_name (str, optional): HuggingFace model identifier for the transformer model.
                Defaults to "bert-base-multilingual-uncased".
            predictor_name (str, optional): Name of the predictor used for generating predictions.
                Defaults to "claude".
            max_length (int, optional): Maximum token sequence length for truncation.
                Defaults to 512.
            device (str, optional): Device specification ('cuda', 'cpu', or specific GPU like 'cuda:0').
                If None, automatically selects CUDA if available. Defaults to None.
            problematic_samples (List[int], optional): Indices of known problematic samples.
                Defaults to None.
            selected_indices (List[int], optional): Pre-selected sample indices to continue from.
                Defaults to empty list.
            all_predictions (List[int], optional): Pre-computed predictions for all samples.
                If None, predictions are loaded from cache. Defaults to None.
            total_predictions (List[int], optional): Complete predictions before any shuffling.
                Defaults to None.
            **kwargs: Additional keyword arguments for subclass-specific parameters.
        """
        self.model_name = model_name
        self.classes = classes
        self.max_length = max_length
        self.get_device(device=device)
        self.pipeline_name = pipeline_name
        self.batch_size = batch_size
        self.predictor_name = predictor_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_labels = len(classes)
        self.problematic_samples = problematic_samples
        # Initialize model with specified number of labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels
        ).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if all_predictions is None:
            if self.pipeline_name[0] != "summarization":
                all_predictions = np.array(get_complete_predictions(dataset_name=self.pipeline_name[1],
                                                predictor_name=self.predictor_name), dtype=int)
            else:
                all_predictions = np.array(get_complete_predictions(dataset_name=self.pipeline_name[1],
                                            predictor_name=self.predictor_name))
            self.total_predictions = deepcopy(all_predictions)
        else:
            self.total_predictions = total_predictions
            
        texts, labels, all_predictions = shuffle_data(texts=texts, labels=labels, all_predictions=all_predictions)
        self.all_predictions = all_predictions
        self.budget = budget
        self.selected_indices = selected_indices
        self.split_data(texts, labels)
        self.initialize_pred_embeddings(texts)
        
    def initialize_pred_embeddings(self, texts):
        """
        Initialize or load predictions and embeddings from cache.
        If cache files don't exist, generate new predictions and embeddings.
        
        Args:
            texts: Input texts to process
        """
        # Load or generate embeddings
        path = os.path.join('infos', model_name_map(self.model_name), f'all_embeddings_{self.pipeline_name[1]}.npy')
        self.embeddings_cache = {}

        if not os.path.exists(path):
            print(f"File {path} not available. Generating embeddings.")
            self.embeddings = self.get_embeddings(texts)
        else:
            self.embeddings = np.load(path,  allow_pickle=True)
        

    def split_data(self, texts, labels):
        """
        Split the data according to the specified budget.
        
        Truncates the dataset to the budget size if specified, otherwise uses all data.
        Also truncates the corresponding predictions array.
        
        Args:
            texts (List[str]): Input text samples.
            labels (List[int]): Corresponding labels for the texts.
        """
        if self.budget is not None:
            self.texts = texts[0:self.budget]
            self.labels = labels[0:self.budget]
            self.all_predictions = self.all_predictions[0:self.budget]
        else:
            self.texts = texts
            self.labels = labels

    def get_device(self, device):
        """
        Set up the computing device (CPU/GPU).
        
        Args:
            device: Specified device, if None will use CUDA if available
        """
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_embeddings(self, texts: Union[
        str,
        List[str],
        Tuple[str, List[str]],
        List[Tuple[str, List[str]]]],
        plot_and_save:bool=False
    ) -> np.ndarray:
        """
        Generate embeddings for input texts using the transformer model.
        
        Extracts the [CLS] token embedding from the last hidden state of the
        transformer model. Uses caching to avoid recomputing embeddings for
        previously seen texts.
        
        Args:
            texts: Input text(s) in various formats:
                - str: Single text string
                - List[str]: List of text strings
                - Tuple[str, List[str]]: Single (question, context_choices) tuple
                - List[Tuple[str, List[str]]]: List of (question, context_choices) tuples
            plot_and_save (bool, optional): If True, generates and saves PCA/t-SNE
                visualization plots of the embeddings. Defaults to False.
            
        Returns:
            np.ndarray: Array of shape (num_texts, embedding_dim) containing
                the computed embeddings for each input text.
        """
        # Convert single text to list format
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, tuple) and isinstance(texts[1], list):
            texts = [texts]

        embeddings = []

        with torch.no_grad():
            for item in tqdm(texts):
                item = ' '.join(item)
                if item in self.embeddings_cache:
                    embeddings.append(self.embeddings_cache[item])
                else:
                    # Tokenize and get BERT embeddings
                    inputs = self.tokenizer(
                        item,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                    ).to(self.device)
                    if hasattr(self.model, self.model.config.model_type):
                        base_model = getattr(self.model, self.model.config.model_type)
                    else:
                        base_model = self.model.model
                    outputs = base_model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    self.embeddings_cache[item] = embedding[0]
                    embeddings.append(embedding[0])
                    
        filename = f'infos/{model_name_map(self.model_name)}/all_embeddings_{self.pipeline_name[1]}.npy'
        os.makedirs(f'infos/{model_name_map(self.model_name)}', exist_ok=True)
        embeddings = np.array(embeddings)
        with open(filename, 'wb') as f:
            np.save(f, embeddings)

        # Save embeddings and create visualization if requested
        if plot_and_save:
            if self.pipeline_name[0] != "ner":
                plot_embeddings(
                    embeddings=embeddings,
                    labels=self.labels,
                    methods=['pca', 'tsne'],
                    dataset_name=self.pipeline_name[1],
                    results_dir=f'infos/{model_name_map(self.model_name)}/embeddings_plots'
                )
        return embeddings

    def extract_predictions(self, selected_indices):
        """
        Extract predictions for a subset of selected samples.
        
        Retrieves the model predictions corresponding to the specified indices
        from the full predictions array.
        
        Args:
            selected_indices (List[int]): Indices of samples to extract predictions for.
                Must be valid indices within self.all_predictions.
            
        Returns:
            List[int]: Predictions for the selected samples in the same order
                as the input indices.
        """
        return [self.all_predictions[i] for i in selected_indices]

    def select_next_test_case(self, num_samples:int=-1) -> int:
        """
        Abstract method to be implemented by subclasses.
        Should select the next test case based on specific strategy.
        
        Args:
            num_samples: Number of samples to select
            
        Returns:
            int: Index of selected test case
        """
        pass

    def estimate_unbiased_metrics(
        self, 
        selected_indices: List[int], 
        acquisition_weights: Optional[np.ndarray] = None,
        dict_to_compare: Dict = None,
        threshold_to_stop: float = 0.05,
    ) -> Dict[str, float]:
        """
        Estimate unbiased performance metrics using importance weighting.
        
        Computes unbiased estimates of accuracy, precision, recall, and F1 score
        using the provided acquisition weights to correct for selection bias in
        active testing. Compares estimates against ground truth metrics.
        
        Args:
            selected_indices (List[int]): Indices of samples selected for testing.
            acquisition_weights (np.ndarray, optional): Selection probability weights
                for each selected sample. Used for importance weighting correction.
                If None, uniform weights are used. Shape: (len(selected_indices),).
                Defaults to None.
            dict_to_compare (Dict[str, float]): Dictionary containing ground truth
                metrics to compare against. Expected keys: 'accuracy', 'precision',
                'recall'. Used to compute estimation errors.
            threshold_to_stop (float, optional): Threshold for early stopping.
                If accuracy estimation error falls below this value, records the
                stopping point. Defaults to 0.05.
            
        Returns:
            Dict[str, float]: Dictionary containing absolute estimation errors:
                - 'unbiased_accuracy': Absolute error in accuracy estimation
                - 'unbiased_precision': Absolute error in precision estimation
                - 'unbiased_recall': Absolute error in recall estimation
                - 'stopped_M': Number of samples when threshold was met, or -1 if not met
        """
        N = len(self.texts)
        M = len(selected_indices)
        
        if acquisition_weights is None:
            acquisition_weights = np.ones(M)
                        
        
        unbiased_accuracy = compute_accuracy_estimator(predictions=self.all_predictions,
                                                       true_labels=self.labels,
                                                       scores=acquisition_weights,
                                                       selected_indices=selected_indices)
        
        unbiased_metrics = compute_unbiased_precision_recall(predictions=self.all_predictions,
                                                             true_labels=self.labels,
                                                             scores=acquisition_weights,
                                                             selected_indices=selected_indices,
                                                             num_classes=self.num_labels
                                                             )

        if abs(dict_to_compare['accuracy'] - unbiased_accuracy) < threshold_to_stop:
            stopped_M = M
        else:
            stopped_M = -1
            
        results = {'unbiased_accuracy' : abs(dict_to_compare['accuracy'] - unbiased_accuracy).item(),
                   'unbiased_precision' : abs(dict_to_compare['precision'] - unbiased_metrics['unbiased_precision']).item(),
                   'unbiased_recall' : abs(dict_to_compare['recall'] - unbiased_metrics['unbiased_recall']).item(),
                   'stopped_M': stopped_M }
        
        return results


    @staticmethod
    def get_class(class_name):
        """
        Factory method to get the appropriate testing strategy class.
        
        Args:
            class_name: Name of the strategy to use
            
        Returns:
            class: The corresponding strategy class
            
        Raises:
            ValueError: If strategy name is not supported
        """
        if class_name == "Random":
            from active_testing.random import Random
            return Random
        elif class_name == "Stratified":
            from active_testing.stratified import Stratified
            return Stratified
        elif class_name == "Coverage":
            from active_testing.coverage import Coverage
            return Coverage
        elif class_name == "Distance":
            from active_testing.distance import Distance
            return Distance
        elif class_name == 'Uncertainty':
            from active_testing.uncertainty import Uncertainty
            return Uncertainty
        elif class_name == 'Agreement':
            from active_testing.agreement import Agreement
            return Agreement
        elif class_name == 'Surrogate':
            from active_testing.surrogate import Surrogate
            return Surrogate
        elif class_name == 'Diversity':
            from active_testing.diversity import Diversity
            return Diversity
        elif class_name == 'Diffuse':
            from active_testing.diffuse import Diffuse
            return Diffuse
        else:
            raise ValueError(f"Class {class_name} not supported!")

    def create_instance(self, class_name, **optional_params):
        """
        Create an instance of the specified testing strategy.
        
        Args:
            class_name: Name of the strategy to instantiate
            **optional_params: Additional parameters for specific strategies
            
        Returns:
            object: Instance of the specified testing strategy
        """
        # Get the appropriate class
        class_type = self.get_class(class_name)

        # Prepare base parameters
        current_attrs = {
            'texts': self.texts,
            'labels': self.labels,
            'budget': self.budget,
            'classes': self.classes,
            'pipeline_name': self.pipeline_name,
            'batch_size': self.batch_size,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'device': self.device,
            'problematic_samples': self.problematic_samples,
            'predictor_name' : self.predictor_name,
            'selected_indices' : self.selected_indices,
            'all_predictions' : self.all_predictions,
            'total_predictions' : self.total_predictions
        }
        

        # Add strategy-specific parameters
        if class_name == "Coverage":
            current_attrs.update(optional_params)
        if class_name == "Uncertainty" or class_name == "Surrogate":
            current_attrs.update({'acquisition': optional_params['method']})
            
        # Create and return instance
        return class_type(**current_attrs)