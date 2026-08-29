from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from active_testing.active import NLPActiveTesting

class AttentionClassifier(nn.Module):
    """
    Neural network module that combines multi-head attention with binary classification.
    
    This module processes input sequences using a self-attention mechanism and outputs
    classification predictions. It is designed to work with transformer-based embeddings
    and provides attention weights that can be used for interpretability and sample
    selection in active testing scenarios.
    
    Architecture:
        1. Multi-head self-attention layer (8 heads)
        2. Mean pooling across sequence length
        3. Two-layer MLP classifier with ReLU activation and dropout
    
    Attributes:
        attention (nn.MultiheadAttention): Multi-head attention layer with 8 heads.
        classifier (nn.Sequential): Classification head consisting of:
            - Linear layer (hidden_size -> 256)
            - ReLU activation
            - Dropout (p=0.1)
            - Linear layer (256 -> 2) for binary classification
    """
    def __init__(self, hidden_size):
        super().__init__()
        # Multi-head attention layer with 8 attention heads
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),  # First dense layer
            nn.ReLU(),                    # Activation function
            nn.Dropout(0.1),              # Dropout for regularization
            nn.Linear(256, 2)             # Output layer for binary classification
        )

        
    def forward(self, x):
        """
        Forward pass of the attention classifier network.
        
        Applies self-attention to the input sequence, pools the attention outputs
        by averaging across the sequence dimension, and passes through the
        classification head.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, hidden_size)
                containing the hidden states from a transformer encoder.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - logits (torch.Tensor): Classification logits of shape (batch_size, 2)
                    for binary classification.
                - attention_weights (torch.Tensor): Attention weight matrix of shape
                    (batch_size, seq_len, seq_len) representing the attention
                    distribution over input positions.
        """
        # Apply self-attention
        attn_output, attention_weights = self.attention(x, x, x)
        # Average attention outputs across sequence length
        pooled = torch.mean(attn_output, dim=0)
        return self.classifier(pooled), attention_weights

class Agreement(NLPActiveTesting):
    """
    Implementation of agreement-based active testing strategy.
    Uses attention patterns to select test cases.
    """
    
    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 budget: int,
                 classes: dict,
                 attention_threshold: float = .5,
                 pipeline_name:str = "sentiment-analysis",
                 batch_size: int = 32,
                 model_name: str = "bert-base-multilingual-uncased",
                 max_length: int = 512,
                 device: str = None,
                 **kwargs):
        """
        Implementation of agreement-based active testing strategy using attention patterns.
    
        This strategy leverages attention mechanisms to identify samples where the model
        shows high uncertainty or disagreement in its attention patterns. Samples with
        higher attention variance are considered more informative for testing, as they
        may represent edge cases or ambiguous inputs.
        
        The strategy computes attention-based importance scores by:
            1. Passing texts through a BERT model to get hidden states
            2. Applying an attention classifier to compute attention weights
            3. Using the variance of attention weights as an uncertainty measure
            4. Selecting samples with highest attention variance scores
        
        Attributes:
            name (str): Strategy identifier, set to 'Agreement'.
            attention_threshold (float): Threshold for attention-based selection filtering.
            bert (AutoModel): Pretrained BERT model for computing hidden states.
            attention_classifier (AttentionClassifier): Attention-based classifier module.
            is_multilabel (bool): Flag indicating if the task is multi-label classification.
        """
        self.name = 'Agreement'
        self.attention_threshold = attention_threshold

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

        self.init_nn_variables()
        if isinstance(self.labels[0], (list, np.ndarray)):
            self.is_multilabel = True
        else:
            self.is_multilabel = False
    
    def init_nn_variables(self) -> None:
        """Initialize the BERT model and attention classifier neural network components.
        
        Loads the pretrained BERT model and creates the AttentionClassifier with
        matching hidden size. Both models are moved to the specified computing device.
        """
        self.bert = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.attention_classifier = AttentionClassifier(self.bert.config.hidden_size).to(self.device)
        
    def compute_attention_scores(self, texts: List[str]) -> List[float]:
        """
        Compute attention-based importance scores for a list of text samples.
        
        Processes texts through BERT and the attention classifier to compute
        uncertainty scores based on attention weight variance. Higher variance
        indicates greater model uncertainty about which parts of the input to focus on.
        
        Args:
            texts (List[str]): List of text samples to evaluate. For NER tasks,
                texts can be lists of tokens that will be joined with spaces.
            
        Returns:
            List[float]: Importance scores for each text sample based on attention
                patterns. Higher scores indicate higher uncertainty/disagreement
                in attention patterns.
        """
        # Set models to evaluation mode
        self.bert.eval()
        self.attention_classifier.eval()
        
        attention_scores = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                if self.pipeline_name[0] == "ner":
                    modified_sentences = []
                    for text in batch_texts:
                        modified_sentences.append(" ".join(text))
                    batch_texts = modified_sentences
                # Tokenize input texts
                inputs = self.tokenizer(batch_texts,
                                     padding=True,
                                     truncation=True,
                                     max_length=64,
                                     return_tensors="pt").to(self.device)
                
                # Get BERT hidden states
                hidden_states = self.bert(**inputs).last_hidden_state
                
                # Get attention weights from classifier
                _, attention_weights = self.attention_classifier(hidden_states.transpose(0,1))
                    
                # Compute variance of attention weights as uncertainty measure
                if self.is_multilabel:
                    batch_scores = self._compute_token_level_scores(
                        attention_weights, 
                        inputs['attention_mask']
                    )
                else:
                    batch_scores = attention_weights.var(dim=-1).mean(dim=-1)
                    batch_scores = batch_scores.cpu().numpy()
                
                attention_scores.extend(batch_scores)
        
        return attention_scores
    
    def select_next_test_case(
        self,
        num_samples: int = -1,
        lang_prior: Tuple[float, float] = None
    ) -> Tuple[List[str], List[int], np.ndarray]:
        """
        Select next test cases based on attention scores with optional language prior.
        
        Identifies the most informative samples for testing based on attention-based
        uncertainty scores. Supports optional language-based weighting for multilingual
        datasets where samples from different languages occupy different halves of
        the dataset.

        Args:
            num_samples (int, optional): Number of samples to select. If -1, selects
                all available (non-previously-selected) samples. Defaults to -1.
            lang_prior (Tuple[float, float], optional): Tuple of probability weights
                (p_lang1, p_lang2) for the first and second halves of the dataset,
                respectively. Used for language-balanced sampling in multilingual
                settings. If None, uses pure attention-based selection without
                language weighting. Defaults to None.

        Returns:
            Tuple[List[int], List[int], np.ndarray]: A tuple containing:
                - predictions (List[int]): Model predictions for the selected samples.
                - selected_indices (List[int]): Indices of the chosen samples in the
                    original dataset.
                - normalized_scores (np.ndarray): Attention scores normalized to sum
                    to 1 over all samples, representing selection probabilities.
        """
        n_texts = len(self.texts)
        half = n_texts // 2

        # Determine number of samples to select
        num_samples = min(n_texts if num_samples == -1 else num_samples, n_texts)

        # Compute attention scores
        attention_scores = np.exp(self.compute_attention_scores(self.texts))

        # Filter out already selected indices
        available_indices = [i for i in range(n_texts) if i not in self.selected_indices]

        if len(available_indices) == 0:
            print("Warning! Available indices not found!")
            return [], [], np.ones(n_texts) / n_texts

        if lang_prior is None:
            # === Original attention-based selection ===
            sorted_indices = sorted(available_indices,
                                    key=lambda i: attention_scores[i],
                                    reverse=True)
            selected_indices = sorted_indices[:num_samples]
        else:
            probs = np.zeros(n_texts)
            for i in available_indices:
                lang_weight = lang_prior[0] if i < half else lang_prior[1]
                probs[i] = attention_scores[i] * lang_weight

            probs = probs / probs.sum()

            selected_indices = np.random.choice(
                available_indices, size=num_samples, replace=False, p=probs[available_indices]
            )

        # Get predictions for selected samples
        predictions = self.extract_predictions(selected_indices)

        # Normalize scores for output
        normalized_scores = attention_scores / np.sum(attention_scores)

        return predictions, list(selected_indices), normalized_scores
    
    def _compute_token_level_scores(
        self, 
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[float]:
        scores = []
      
        for b in range(attention_weights.shape[0]):
            mask = attention_mask[b].bool()
            seq_len = mask.sum().item()
            
            if seq_len == 0:
                scores.append(0.0)
                continue
            
            attn = attention_weights[b]
            
            eps = 1e-8
            attn_probs = attn + eps
            entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)
            avg_entropy = entropy.mean().item()
            
            variance = attn.var().item()
            normalized_variance = variance * np.log1p(seq_len)  
            
            score = 0.5 * avg_entropy + 0.5 * normalized_variance
            scores.append(score)
        
        return scores
