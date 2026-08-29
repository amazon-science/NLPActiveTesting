from typing import List, Tuple
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans, DBSCAN, MeanShift
import os
import matplotlib.pyplot as plt
from active_testing.active import NLPActiveTesting
import seaborn as sns

class Coverage(NLPActiveTesting):
    """
    Coverage-based test selection strategy for active testing in NLP tasks.
    
    Implements multiple strategies to select test cases that maximize coverage
    of the embedding space. This approach aims to ensure diverse and representative
    test samples by leveraging clustering and distance-based selection methods.
    
    Selection Methods:
        - max_dist: Iteratively selects points that maximize distance from previously
          selected points, ensuring broad coverage of the embedding space.
        - centroids: Uses clustering to identify representative regions and selects
          points closest to cluster centroids.
        - clusters: Distributes sample selection proportionally across clusters based
          on cluster sizes, ensuring representation from all regions.
    
    Clustering Algorithms:
        - kmeans: K-Means clustering with specified number of clusters.
        - dbscan: Density-based clustering that automatically determines cluster count.
        - shift: Mean Shift clustering for automatic cluster detection.
    
    Attributes:
        name (str): Strategy identifier, set to 'Coverage'.
        method (str): Selection strategy ('max_dist', 'centroids', or 'clusters').
        clustering (str): Clustering algorithm ('kmeans', 'dbscan', or 'shift').
        n_clusters (int): Number of clusters for K-Means algorithm.
        scores (np.ndarray): Selection probability scores for each sample.
    """
    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 budget:float,
                 classes:dict,
                 pipeline_name:str = "sentiment-analysis",
                 batch_size:int=32,
                 model_name: str = "bert-base-multilingual-uncased",
                 max_length: int = 512,
                 device: str = None,
                 method: str = "max_dist",
                 clustering: str = "kmeans",
                 n_clusters: int = 10,
                 **kwargs):
        """
        Initialize the Coverage-based active testing strategy.
        
        Sets up the coverage-based selection mechanism with configurable selection
        method and clustering algorithm for embedding space exploration.
        
        Args:
            texts (List[str]): List of input text samples to be tested.
            labels (List[int]): Corresponding ground truth labels for the texts.
            budget (float): Maximum number of samples to consider from the dataset.
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
            method (str, optional): Selection strategy to use. Options:
                - 'max_dist': Maximize distance between selected points (greedy).
                - 'centroids': Select points closest to cluster centroids.
                - 'clusters': Proportional selection from each cluster.
                Defaults to "max_dist".
            clustering (str, optional): Clustering algorithm for centroid/cluster methods.
                Options: 'kmeans', 'dbscan', 'shift'. Defaults to "kmeans".
            n_clusters (int, optional): Number of clusters for K-Means algorithm.
                Ignored for DBSCAN and Mean Shift. Defaults to 10.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        self.method = method
        self.clustering = clustering
        self.n_clusters = n_clusters
        self.name = 'Coverage'

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
        
    def plot_distance_distribution(self, distances_history, selected_indices):
        """
        Plot PDF and CDF of distances between consecutive selected points.
        
        Creates a visualization showing the distribution of distances in the
        selection sequence, distinguishing between problematic and normal samples.
        Useful for analyzing whether the selection strategy effectively identifies
        edge cases.
        
        Args:
            distances_history (np.ndarray): Array of distances between consecutive
                selected points. Length should be len(selected_indices) - 1.
            selected_indices (List[int]): List of indices of selected points in
                selection order.
        """
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Separate distances based on whether they lead to problematic samples
        problematic_distances = []
        normal_distances = []
        
        for i in range(len(distances_history)):
            if selected_indices[i+1] in self.problematic_samples:
                problematic_distances.append(distances_history[i])
            else:
                normal_distances.append(distances_history[i])
        
        # Plot Probability Density Function (PDF)
        if problematic_distances:
            sns.kdeplot(data=problematic_distances, ax=ax1, color='green', label='Problematic')
        if normal_distances:
            sns.kdeplot(data=normal_distances, ax=ax1, color='red', label='Normal')
        
        ax1.set_title('Probability Density Function')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        if problematic_distances:
            sorted_prob_distances = np.sort(problematic_distances)
            cumulative_prob = np.arange(1, len(sorted_prob_distances) + 1) / len(distances_history)
            ax2.plot(sorted_prob_distances, cumulative_prob, 'g-', label='Problematic')
        
        if normal_distances:
            sorted_norm_distances = np.sort(normal_distances)
            cumulative_norm = np.arange(1, len(sorted_norm_distances) + 1) / len(distances_history)
            ax2.plot(sorted_norm_distances, cumulative_norm, 'r-', label='Normal')
        
        ax2.set_title('Cumulative Distribution Function')
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('Cumulative Probability')
        ax2.legend()
        
        plt.tight_layout()
        
        os.makedirs(f'results/evolution/{self.pipeline_name[1]}', exist_ok=True)
        plt.savefig(f'results/evolution/{self.pipeline_name[1]}/distance_distribution_{len(distances_history)+1}.png')

    def plot_selection_evolution(self, all_embeddings, selected_indices, num_samples):
        """
        Visualize the sequential selection of points in the embedding space.
        
        Creates a 2D scatter plot showing how points are selected over time,
        with arrows indicating the selection order and colors distinguishing
        between problematic and normal samples.
        
        Args:
            all_embeddings (np.ndarray): Matrix of embeddings for all points.
                Shape: (n_samples, embedding_dim). Only first 2 dimensions are plotted.
            selected_indices (List[int]): List of indices of selected points in
                the order they were selected.
            num_samples (int): Total number of samples selected (used in title).
        """
        plt.figure(figsize=(15, 10))
        
        # Plot background points (not selected) in gray
        plt.scatter(all_embeddings[:, 0], all_embeddings[:, 1], 
                   c='gray', alpha=0.5, label='Not selected')
        
        # Plot first point with special handling
        first_point_color = 'green' if selected_indices[0] in self.problematic_samples else 'red'
        plt.scatter(all_embeddings[selected_indices[0], 0],
                   all_embeddings[selected_indices[0], 1],
                   c=first_point_color, s=100)
        plt.annotate(f'{selected_indices[0]} (1)', 
                    (all_embeddings[selected_indices[0], 0], 
                     all_embeddings[selected_indices[0], 1]),
                    xytext=(5, 5), textcoords='offset points')
        
        # Plot connections between consecutive points
        for i in range(len(selected_indices)-1):
            current_idx = selected_indices[i]
            next_idx = selected_indices[i+1]
            
            # Determine line color based on point types
            line_color = 'green' if (current_idx in self.problematic_samples and 
                                   next_idx in self.problematic_samples) else 'black'
            
            # Draw connection line and arrow
            plt.plot([all_embeddings[current_idx][0], all_embeddings[next_idx][0]],
                    [all_embeddings[current_idx][1], all_embeddings[next_idx][1]],
                    color=line_color, linestyle='--', alpha=0.5)
            
            plt.annotate('',
                        xy=(all_embeddings[next_idx][0], all_embeddings[next_idx][1]),
                        xytext=(all_embeddings[current_idx][0], all_embeddings[current_idx][1]),
                        arrowprops=dict(arrowstyle='->', color=line_color, lw=2))
            
            # Plot point with appropriate color
            point_color = 'green' if next_idx in self.problematic_samples else 'red'
            plt.scatter(all_embeddings[next_idx, 0], 
                       all_embeddings[next_idx, 1], 
                       c=point_color, s=100)
            
            # Add label with index and order number
            plt.annotate(f'{next_idx} ({i+2})', 
                        (all_embeddings[next_idx, 0], all_embeddings[next_idx, 1]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'Sequential Selection of Most Distant Points ({num_samples})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(['Not selected', 'Problematic', 'Normal'])
        
        # Save the plot
        os.makedirs(f'results/evolution/{self.pipeline_name[1]}', exist_ok=True)
        plt.savefig(f'results/evolution/{self.pipeline_name[1]}/evolution_{num_samples}.png')
        
    def plot_distances(self, distances_history, selected_indices):
        """
        Create a line plot showing distances between consecutive selected points.
        
        Visualizes how the selection distance changes throughout the selection
        process, with different colors for problematic vs normal samples.
        
        Args:
            distances_history (List[float]): List of distances between consecutive
                points in selection order.
            selected_indices (List[int]): List of indices of selected points.
        """
        plt.figure(figsize=(10, 5))
        
        # Plot points with different colors based on sample type
        for i in range(len(distances_history)):
            if selected_indices[i+1] in self.problematic_samples:
                plt.plot(i+1, distances_history[i], 'go')  # Green for problematic
            else:
                plt.plot(i+1, distances_history[i], 'ro')  # Red for normal
        
        # Connect points with lines
        plt.plot(range(1, len(distances_history) + 1), distances_history, 'k-', alpha=0.5)
        
        plt.title(f'Distances Between Consecutive Points ({len(distances_history)+1})')
        plt.xlabel('Point Number')
        plt.ylabel('Distance from Previous Point')
        plt.grid(True)
        
        # Save the plot
        os.makedirs(f'results/evolution/{self.pipeline_name[1]}', exist_ok=True)
        plt.savefig(f'results/evolution/{self.pipeline_name[1]}/distances_{len(distances_history)+1}.png')

    def select_next_test_case(
            self,
            num_samples: int = -1,
            plot: bool = False,
            lang_prior = None
    ) -> Tuple[List[str], List[int]]:
        """
        Select the next batch of test cases using the specified coverage strategy.
        
        Implements the main selection logic for coverage-based active testing.
        Supports three selection methods and optional language-based weighting
        for multilingual datasets.
        
        Args:
            num_samples (int, optional): Number of samples to select. If -1, selects
                all available samples. Defaults to -1.
            plot (bool, optional): If True and problematic_samples is set, generates
                visualization plots showing selection evolution, distance history,
                and distance distributions. Only applicable for 'max_dist' method.
                Defaults to False.
            lang_prior (Tuple[float, float], optional): Tuple of probability weights
                (p_lang1, p_lang2) for the first and second halves of the dataset.
                Used for language-balanced sampling in multilingual settings.
                Only applicable for 'max_dist' method. Defaults to None.
            
        Returns:
            Tuple[List[int], List[int], np.ndarray]: A tuple containing:
                - predictions (List[int]): Model predictions for the selected samples.
                - selected_indices (List[int]): Indices of the chosen samples.
                - scores (np.ndarray): Selection probability scores, normalized to
                  sum to 1. For 'max_dist', based on distances; for 'centroids',
                  based on proximity to centroids.
        """
        # Determine number of samples to select
        num_samples = min(int(len(self.texts)) if num_samples == -1 else num_samples, len(self.texts))
        all_embeddings = self.get_embeddings(self.texts)
        n_texts = len(self.texts)
        half = n_texts // 2

        # If selecting all samples, return everything
        if num_samples == len(self.texts):
            selected_indices = np.arange(len(self.texts))
            predictions = self.extract_predictions(selected_indices)
            self.scores = np.ones(len(self.texts)) / len(self.texts)
            return predictions, selected_indices, self.scores

        if self.method == "max_dist":
            # Start with a random point
            selected_indices = [np.random.choice(len(all_embeddings))]
            distances_history = []

            for _ in range(num_samples - 1):
                last_point = all_embeddings[selected_indices[-1]].reshape(1, -1)
                distances = distance.cdist(all_embeddings, last_point).flatten()
                distances[selected_indices] = -1  # Exclude already selected points

                # Apply language prior if provided
                if lang_prior is not None:
                    lang_weights = np.ones_like(distances)
                    for i in range(n_texts):
                        if i in selected_indices:
                            lang_weights[i] = 0
                        elif i < half:
                            lang_weights[i] *= lang_prior[0]
                        else:
                            lang_weights[i] *= lang_prior[1]
                    # Weighted distances
                    distances = distances * lang_weights

                next_point_idx = np.argmax(distances)
                distances_history.append(distances[next_point_idx])
                selected_indices.append(next_point_idx)

            distances_history = np.array(distances_history)
            distances_history = np.insert(distances_history,0, max(distances_history))
            if isinstance(self.labels[0], (list, np.ndarray)):
                for idx, orig_idx in enumerate(selected_indices):
                    self.scores[orig_idx] = distances_history[idx]
                total = np.sum(self.scores)
                if total > 0:
                    self.scores = self.scores / total
            
            # Generate plots if requested
            if plot and self.problematic_samples is not None:
                self.plot_selection_evolution(all_embeddings, selected_indices, num_samples)
                self.plot_distances(distances_history, selected_indices)
                self.plot_distance_distribution(distances_history, selected_indices)
        # Centroids-based strategy
        elif self.method == "centroids":
            clusterer = self._get_clusterer(num_samples=num_samples)
            cluster_labels = clusterer.fit_predict(all_embeddings)
            selected_indices = self._select_by_centroids(
                cluster_labels,
                all_embeddings,
                self.texts,
                num_samples,
            )

        # Clusters-based strategy
        elif self.method == "clusters":
            clusterer = self._get_clusterer(num_samples=num_samples)
            cluster_labels = clusterer.fit_predict(all_embeddings)
            selected_indices = self._select_by_clusters(
                cluster_labels,
                all_embeddings,
                num_samples,
            )

        else:
            raise ValueError(f"method {self.method} not recognized!")
            
        predictions = self.extract_predictions(selected_indices)
        
        return predictions, selected_indices, self.scores
    
    def _select_by_centroids(
            self,
            cluster_labels: np.ndarray,
            embeddings: np.ndarray,
            texts: List[str],
            num_samples: int,
    ) -> List[int]:
        """
        Select test cases based on their proximity to cluster centroids.
        
        Computes a score for each point based on its inverse distance to the
        centroid of its cluster. Points closer to centroids receive higher scores
        and are more likely to be selected. Selection is performed probabilistically
        using these scores.
        
        Args:
            cluster_labels (np.ndarray): Array of cluster assignments for each point.
                Shape: (n_samples,). Value -1 indicates noise points (excluded).
            embeddings (np.ndarray): Matrix of point embeddings.
                Shape: (n_samples, embedding_dim).
            texts (List[str]): List of input texts (used for determining array size).
            num_samples (int): Number of samples to select.
            
        Returns:
            List[int]: Indices of selected samples.

        """
        # Get unique cluster labels, excluding noise points (-1)
        unique_labels = np.unique(cluster_labels)
        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]

        # Calculate scores for each point
        if isinstance(self.labels[0], (list, np.ndarray)):
            self.scores = np.zeros(n_texts)
        else:
            self.scores = np.zeros(len(texts))
        for cluster_id in unique_labels:
            # Get points in current cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_points = embeddings[cluster_mask]

            if len(cluster_points) == 0:
                continue

            # Calculate centroid and distances to it
            centroid = np.mean(cluster_points, axis=0)
            distances_to_centroid = distance.cdist([centroid], cluster_points)[0]
            # Convert distances to scores (closer points get higher scores)
            cluster_scores = 1 / (distances_to_centroid + 1e-10)
            self.scores[cluster_mask] = cluster_scores

        # Normalize scores and apply prior if available
        if np.sum(self.scores) > 0:
            self.scores = self.scores / np.sum(self.scores)
                
            selected_indices = np.random.choice(
                len(texts),
                size=num_samples,
                p=self.scores,
                replace=False
            )
        else:
            # If all scores are zero, select randomly
            selected_indices = np.random.choice(
                len(texts),
                size=num_samples,
                replace=False
            )
        return selected_indices

    def _select_by_clusters(
            self,
            cluster_labels: np.ndarray,
            embeddings: np.ndarray,
            num_samples: int
            ) -> List[int]:
        """
        Select test cases proportionally from different clusters.
        
        Distributes the selection budget across clusters proportionally to their
        sizes. Within each cluster, selects points closest to the cluster centroid.
        Ensures representation from all identified clusters.
        
        Args:
            cluster_labels (np.ndarray): Array of cluster assignments for each point.
                Shape: (n_samples,).
            embeddings (np.ndarray): Matrix of point embeddings.
                Shape: (n_samples, embedding_dim).
            num_samples (int): Total number of samples to select across all clusters.
            
        Returns:
            List[int]: Indices of selected samples, truncated to exactly num_samples.
        """
        unique_labels = np.unique(cluster_labels)
        selected_indices = []

        # Calculate number of samples to select from each cluster
        samples_per_cluster = {}
        total_points = len(cluster_labels)
        remaining_samples = num_samples

        # Distribute samples proportionally to cluster sizes
        for label in unique_labels:
            cluster_size = np.sum(cluster_labels == label)
            cluster_proportion = cluster_size / total_points
            samples = max(1, int(cluster_proportion * num_samples))
            if remaining_samples >= samples:
                samples_per_cluster[label] = samples
                remaining_samples -= samples

        # Select samples from each cluster
        for label, n_samples in samples_per_cluster.items():
            cluster_mask = cluster_labels == label
            cluster_points = embeddings[cluster_mask]

            if len(cluster_points) > 0:
                # Calculate centroid and distances
                centroid = np.mean(cluster_points, axis=0)
                distances_to_centroid = distance.cdist([centroid], cluster_points)[0]
                
                cluster_indices = np.where(cluster_mask)[0]
                selected = cluster_indices[np.argsort(distances_to_centroid)[:n_samples]]
                selected_indices.extend(selected)

        # Fill remaining slots if needed
        if len(selected_indices) < num_samples:
            remaining = num_samples - len(selected_indices)
            available_indices = list(set(range(len(embeddings))) - set(selected_indices))
            if available_indices:
                    # Random selection if no prior
                additional = np.random.choice(
                    available_indices,
                    size=min(remaining, len(available_indices)),
                    replace=False
                )
            selected_indices.extend(additional)

        return selected_indices[:num_samples]

    def _get_clusterer(self, num_samples: int = -1):
        """
        Create and return appropriate clustering algorithm instance.
        
        Args:
            num_samples: Number of samples to consider for clustering
            
        Returns:
            Clustering algorithm instance
            
        Raises:
            ValueError: If clustering method is not recognized
        """
        if self.clustering == "kmeans":
            return KMeans(n_clusters=min(self.n_clusters, num_samples), n_init="auto")
        elif self.clustering == "dbscan":
            return DBSCAN(eps=0.5, min_samples=5)
        elif self.clustering == "shift":
            return MeanShift()
        else:
            raise ValueError(f"Clustering method {self.clustering} not recognized!")