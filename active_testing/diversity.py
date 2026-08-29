from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP
import pandas as pd

from active_testing.active import NLPActiveTesting

class Diversity(NLPActiveTesting):
    """
    Pure diversity-based selection using cluster entropy maximization.
    
    Selects samples to maximize Shannon entropy H(S) across clusters:
    H(S) = -∑ p(c) log p(c) where p(c) is the proportion of samples in cluster c.
    
    No uncertainty computation needed - much faster than Uncertainty-based methods.
    """
    
    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 budget: int,
                 classes: dict,
                 num_clusters: int = 5,
                 pipeline_name: str = "sentiment-analysis",
                 batch_size: int = 32,
                 model_name: str = "bert-base-multilingual-uncased",
                 max_length: int = 512,
                 device: str = None,
                 **kwargs):
        """
        Initialize pure diversity strategy.
        
        Args:
            texts: List of input texts
            labels: Corresponding labels
            budget: Maximum selection size k
            classes: Dictionary of class mappings
            num_clusters: Number of clusters c for K-means
            pipeline_name: Name of the NLP pipeline
            batch_size: Size of processing batches
            model_name: Name of the pretrained model
            max_length: Maximum sequence length
            device: Computing device (CPU/GPU)
        """
        self.name = 'Diversity'
        self.num_clusters = num_clusters
        
        super().__init__(
            texts=texts,
            labels=labels,
            budget=budget,
            classes=classes,
            pipeline_name=pipeline_name,
            batch_size=batch_size,
            model_name=model_name,
            max_length=max_length,
            device=device,
            **kwargs
        )
        
        self.init_clustering()
    
    def init_clustering(self) -> None:
        """
        Encode texts into embedding space and apply K-means clustering.
        """
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        embeddings = self.get_embeddings(self.texts, plot_and_save=False)
        self.cluster_labels = kmeans.fit_predict(embeddings)
             
    def compute_entropy_diversity(self, selected_indices: List[int]) -> float:
        """
        Compute entropy-based diversity score H(S) from cluster distribution.
        
        Args:
            selected_indices: List of selected sample indices
            
        Returns:
            float: Normalized Shannon entropy [0, 1]
        """
        if len(selected_indices) == 0:
            return 0.0

        selected_clusters = self.cluster_labels[selected_indices]
        
        # Compute cluster distribution
        cluster_counts = np.bincount(selected_clusters, minlength=self.num_clusters)
        cluster_probs = cluster_counts / cluster_counts.sum()
        
        # Shannon entropy: H(S) = -∑ p(c) log p(c)
        entropy = -np.sum([p * np.log(p + 1e-10) for p in cluster_probs if p > 0])
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(self.num_clusters)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def compute_diversity_contribution(self, idx: int, current_selection: List[int]) -> float:
        """
        Compute marginal diversity contribution when adding a sample.
        
        Args:
            idx: Index of the sample to evaluate
            current_selection: List of currently selected sample indices
            
        Returns:
            float: Marginal diversity gain ΔH
        """
        if len(current_selection) == 0:
            # First sample: uniform contribution
            return 1.0 / self.num_clusters
        
        # Marginal diversity: H(S ∪ {i}) - H(S)
        diversity_before = self.compute_entropy_diversity(current_selection)
        diversity_after = self.compute_entropy_diversity(current_selection + [idx])
        
        contribution = diversity_after - diversity_before
        
        return max(0.0, contribution)

    
    def greedy_selection(self, k: int, available_indices: List[int]) -> Tuple[List[int], np.ndarray]:
        """
        Greedy algorithm to maximize cluster diversity (efficient, incremental).
        Returns (selected_indices_in_order, diversity_scores_array).
        """
        n = len(self.texts)
        if n == 0:
            return [], np.array([])

        # Ensure cluster_labels is numpy array of ints
        cluster_labels = np.asarray(self.cluster_labels, dtype=int)
        num_clusters = int(self.num_clusters)

        selected: List[int] = []
        available_pool = set(available_indices)

        # Scores for all items (kept across iterations)
        diversity_scores = np.zeros(n, dtype=float)

        # Current cluster counts and current entropy
        current_counts = np.zeros(num_clusters, dtype=int)
        current_total = 0  # number of selected so far

        def entropy_from_counts(counts, total):
            # if total == 0: entropy = 0
            if total == 0:
                return 0.0
            probs = counts.astype(float) / float(total)
            mask = probs > 0.0
            probs = probs[mask]
            return -float(np.sum(probs * np.log(probs)))

        current_entropy = 0.0

        # Precompute mapping from cluster -> list of available indices in that cluster
        cluster_to_available = {c: [] for c in range(num_clusters)}
        for idx in available_pool:
            c = int(cluster_labels[idx])
            cluster_to_available[c].append(idx)

        max_iters = min(k, len(available_indices))
        for iteration in tqdm(range(max_iters)):
            # Compute marginal gain for adding one element to each cluster:
            # H(counts + 1_on_cluster) - current_entropy
            gains_per_cluster = np.full(num_clusters, -np.inf, dtype=float)
            if current_total == 0:
                # If no elements selected, adding first sample to any cluster:
                # new entropy will be 0 because distribution will be one cluster with prob=1 => entropy 0.
                # But we want to encourage spread: we can treat initial gain as small positive
                # Alternative: set gain to 1/num_clusters (as your original)
                gains_per_cluster[:] = 1.0 / float(num_clusters)
            else:
                # compute for clusters that actually have available candidates
                for c in range(num_clusters):
                    if len(cluster_to_available.get(c, [])) == 0:
                        # if no available items in this cluster, skip
                        continue
                    # new counts if we add one more element in cluster c
                    new_counts = current_counts.copy()
                    new_counts[c] += 1
                    new_total = current_total + 1
                    new_entropy = entropy_from_counts(new_counts, new_total)
                    gains_per_cluster[c] = new_entropy - current_entropy

            # Choose the cluster with maximal gain that has available samples
            # If all -inf (no available candidates), break early
            if np.all(np.isneginf(gains_per_cluster)):
                break

            # pick best cluster; tie-break by cluster size (prefer clusters with more candidates)
            best_cluster = int(np.nanargmax(gains_per_cluster))
            best_gain = float(gains_per_cluster[best_cluster])

            # pick one candidate index from that cluster:
            # choose the candidate with largest current diversity_scores (or arbitrary if all zero)
            candidates = cluster_to_available.get(best_cluster, [])
            if len(candidates) == 0:
                # safety: remove cluster and continue
                gains_per_cluster[best_cluster] = -np.inf
                continue

            # pick candidate with highest score (or first if scores are equal)
            best_idx = max(candidates, key=lambda i: diversity_scores[i])

            # compute decay and update score for this chosen index
            decay = 1.0 / np.sqrt(iteration + 1)
            diversity_scores[best_idx] = max(diversity_scores[best_idx], best_gain * decay)

            # Select it
            selected.append(best_idx)
            available_pool.remove(best_idx)
            cluster_to_available[best_cluster].remove(best_idx)

            # apply selection boost to chosen sample
            selection_boost = (max_iters - iteration) / float(max_iters)
            diversity_scores[best_idx] *= (1.5 + selection_boost)

            # Update counts and entropy
            current_counts[best_cluster] += 1
            current_total += 1
            current_entropy = entropy_from_counts(current_counts, current_total)

            # If cluster_to_available[best_cluster] is empty now, it's fine — next loop will skip it

        return selected, diversity_scores

    
    def select_next_test_case(
        self,
        num_samples: int = -1,
        lang_prior: Tuple[float, float] = None
    ) -> Tuple[List, List[int], np.ndarray]:
        """
        Select samples to maximize cluster diversity.
        
        Args:
            num_samples: Number of samples to select (-1 = use budget)
            lang_prior: Optional language prior (p_lang1, p_lang2)
            
        Returns:
            predictions: Model predictions for selected samples
            selected_indices: Indices of selected samples
            selection_scores: Diversity-based scores (normalized)
        """
        n_texts = len(self.texts)
        half = n_texts // 2
        
        # Determine selection size k
        if num_samples == -1:
            k = min(self.budget, n_texts)
        else:
            k = min(num_samples, n_texts)
        
        # Get available indices
        available_indices = [i for i in range(n_texts) 
                           if i not in self.selected_indices]
        
        if len(available_indices) == 0:
            print("Warning! No available indices remaining!")
            return [], [], np.ones(n_texts) / n_texts
        
        
        # Run greedy diversity maximization
        selected_indices, diversity_scores = self.greedy_selection(k, available_indices)
        
        # Apply optional language prior
        if lang_prior is not None:
            for i in range(n_texts):
                lang_weight = lang_prior[0] if i < half else lang_prior[1]
                diversity_scores[i] *= lang_weight
        
        # Normalize to probability distribution
        score_sum = diversity_scores.sum()
        if score_sum > 0:
            selection_scores = diversity_scores / score_sum
        else:
            print("⚠️  WARNING: All scores are zero! Using uniform distribution.")
            selection_scores = np.ones(n_texts) / n_texts
        
        # Get predictions
        predictions = self.extract_predictions(selected_indices)
        self.plot_diversity_analysis(selected_indices)
        return predictions, selected_indices, selection_scores
    

    def plot_diversity_analysis(self, selected_indices: List[int], embeddings: np.ndarray = None):
        """
        Create a comprehensive visualization to verify diversification.
        
        Args:
            selected_indices: Indices of selected samples
            embeddings: Embeddings (optional, will compute if not provided)
        """
        if embeddings is None:
            embeddings = self.get_embeddings(self.texts, plot_and_save=False)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. T-SNE 2D visualization
        ax1 = plt.subplot(2, 3, 1)
        self._plot_tsne_clusters(embeddings, selected_indices, ax1, method='tsne')
        
        # 2. UMAP 2D visualization (often better than t-SNE)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_tsne_clusters(embeddings, selected_indices, ax2, method='umap')
        
        # 3. Cluster distribution
        ax3 = plt.subplot(2, 3, 3)
        self._plot_cluster_distribution(selected_indices, ax3)
        
        # 4. Selection order heatmap
        ax4 = plt.subplot(2, 3, 4)
        self._plot_selection_order(selected_indices, embeddings, ax4)
        
        # 5. Entropy evolution
        ax5 = plt.subplot(2, 3, 5)
        self._plot_entropy_evolution(selected_indices, ax5)
        
        # 6. Distance matrix heatmap
        #ax6 = plt.sublplot(2, 3, 6)
        #self._plot_distance_matrix(selected_indices, embeddings, ax6)
        
        plt.tight_layout()
        plt.savefig('diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        self._print_diversity_stats(selected_indices)

    def _plot_tsne_clusters(self, embeddings, selected_indices, ax, method='tsne'):
        """Visualize embeddings reduced to 2D with clusters and selections"""
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            title = 't-SNE Projection'
        else:  # umap
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings)-1))
            title = 'UMAP Projection'
        
        if isinstance(embeddings, list):
            coords_2d = reducer.fit_transform(np.array(embeddings))
        
        # Create masks
        is_selected = np.zeros(len(embeddings), dtype=bool)
        is_selected[selected_indices] = True
        
        # Plot non-selected (small, transparent)
        for cluster_id in range(self.num_clusters):
            mask = (self.cluster_labels == cluster_id) & (~is_selected)
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                    c=f'C{cluster_id}', alpha=0.3, s=30,
                    label=f'Cluster {cluster_id}' if mask.any() else '')
        
        # Plot selected (large, black border)
        for cluster_id in range(self.num_clusters):
            mask = (self.cluster_labels == cluster_id) & is_selected
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                    c=f'C{cluster_id}', alpha=1.0, s=200,
                    edgecolors='black', linewidths=2, marker='*')
        
        # Number selected points in selection order
        for order, idx in enumerate(selected_indices[:20]):  # first 20
            ax.annotate(f'{order+1}', (coords_2d[idx, 0], coords_2d[idx, 1]),
                    fontsize=8, ha='center', va='center', weight='bold')
        
        ax.set_title(f'{title}\n★ = Selected samples', fontsize=12, weight='bold')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_cluster_distribution(self, selected_indices, ax):
        """Compare cluster distribution: selected vs available"""
        selected_clusters = self.cluster_labels[selected_indices]
        all_clusters = self.cluster_labels
        
        # Count per cluster
        selected_counts = np.bincount(selected_clusters, minlength=self.num_clusters)
        total_counts = np.bincount(all_clusters, minlength=self.num_clusters)
        
        # Proportions
        selected_props = selected_counts / len(selected_indices) * 100
        total_props = total_counts / len(all_clusters) * 100
        
        x = np.arange(self.num_clusters)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, total_props, width, label='Available', alpha=0.7)
        bars2 = ax.bar(x + width/2, selected_props, width, label='Selected', alpha=0.9)
        
        # Ideal uniform line
        ideal = 100 / self.num_clusters
        ax.axhline(ideal, color='red', linestyle='--', linewidth=2, 
                label=f'Uniform ({ideal:.1f}%)')
        
        ax.set_xlabel('Cluster ID', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title('Cluster Coverage Distribution', fontsize=12, weight='bold')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add counts above bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1,
                f'{total_counts[i]}', ha='center', va='bottom', fontsize=8)
            ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1,
                f'{selected_counts[i]}', ha='center', va='bottom', 
                fontsize=8, weight='bold')

    def _plot_entropy_evolution(self, selected_indices, ax):
        """Show how entropy evolves during greedy selection"""
        entropies = []
        cumulative_selection = []
        
        for i in range(1, len(selected_indices) + 1):
            partial_selection = selected_indices[:i]
            entropy = self.compute_entropy_diversity(partial_selection)
            entropies.append(entropy)
            cumulative_selection.append(i)
        
        ax.plot(cumulative_selection, entropies, 'b-', linewidth=2, marker='o', 
                markersize=4, label='Actual entropy')
        
        # Maximum theoretical entropy
        max_entropy = 1.0  # normalized
        ax.axhline(max_entropy, color='red', linestyle='--', linewidth=2,
                label='Maximum entropy')
        
        # Random uniform entropy
        random_entropies = []
        for i in range(1, len(selected_indices) + 1):
            random_selection = np.random.choice(len(self.texts), size=i, replace=False)
            random_entropy = self.compute_entropy_diversity(list(random_selection))
            random_entropies.append(random_entropy)
        ax.plot(cumulative_selection, random_entropies, 'g--', linewidth=1.5, 
                alpha=0.7, label='Random selection')
        
        ax.set_xlabel('Number of selected samples', fontsize=11)
        ax.set_ylabel('Normalized Entropy H(S)', fontsize=11)
        ax.set_title('Diversity Evolution', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])

    def _plot_selection_order(self, selected_indices, embeddings, ax):
        """Heatmap of selection order per cluster"""
        # Matrix: rows = selection order, columns = cluster
        n_selected = len(selected_indices)
        matrix = np.zeros((n_selected, self.num_clusters))
        
        for order, idx in enumerate(selected_indices):
            cluster = self.cluster_labels[idx]
            matrix[order, cluster] = 1
        
        # Cumulative to see progressive coverage
        cumulative = np.cumsum(matrix, axis=0)
        
        sns.heatmap(cumulative, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Cumulative count'},
                    xticklabels=[f'C{i}' for i in range(self.num_clusters)],
                    yticklabels=[f'{i+1}' if i % 5 == 0 else '' for i in range(n_selected)])
        
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Selection order', fontsize=11)
        ax.set_title('Cumulative Cluster Coverage', fontsize=12, weight='bold')

    def _plot_distance_matrix(self, selected_indices, embeddings, ax):
        """Heatmap of distances between selected samples"""
        from scipy.spatial.distance import pdist, squareform
        
        # Take max 50 samples for readability
        display_indices = selected_indices[:50]
        selected_embeddings = embeddings[display_indices]
        
        # Calculate distance matrix
        distances = squareform(pdist(selected_embeddings, metric='cosine'))
        
        sns.heatmap(distances, ax=ax, cmap='viridis', square=True,
                    cbar_kws={'label': 'Cosine distance'},
                    xticklabels=False, yticklabels=False)
        
        ax.set_title(f'Distance Matrix (first {len(display_indices)} selected)\nDarker = more similar',
                    fontsize=12, weight='bold')
        
        # Statistics
        avg_dist = distances[np.triu_indices_from(distances, k=1)].mean()
        ax.text(0.02, 0.98, f'Avg distance: {avg_dist:.3f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _print_diversity_stats(self, selected_indices):
        """Print detailed statistics about diversification"""
        selected_clusters = self.cluster_labels[selected_indices]
        cluster_counts = np.bincount(selected_clusters, minlength=self.num_clusters)
        
        print(f"\n{'='*70}")
        print(f"DIVERSITY ANALYSIS REPORT")
        print(f"{'='*70}")
        
        # Entropy
        final_entropy = self.compute_entropy_diversity(selected_indices)
        print(f"\n📊 Entropy Metrics:")
        print(f"   Final entropy: {final_entropy:.4f} (max = 1.0)")
        print(f"   Diversity score: {final_entropy * 100:.1f}%")
        
        # Cluster coverage
        covered_clusters = np.sum(cluster_counts > 0)
        print(f"\n🎯 Cluster Coverage:")
        print(f"   Covered: {covered_clusters}/{self.num_clusters} clusters ({covered_clusters/self.num_clusters*100:.1f}%)")
        print(f"   Distribution: {cluster_counts}")
        
        # Uniformity
        ideal_per_cluster = len(selected_indices) / self.num_clusters
        deviations = np.abs(cluster_counts - ideal_per_cluster)
        avg_deviation = deviations.mean()
        print(f"\n⚖️  Uniformity:")
        print(f"   Ideal per cluster: {ideal_per_cluster:.1f}")
        print(f"   Average deviation: {avg_deviation:.2f}")
        print(f"   Max deviation: {deviations.max():.0f}")
        
        # Balance score (how close to uniform)
        balance_score = 1 - (deviations.std() / ideal_per_cluster) if ideal_per_cluster > 0 else 0
        print(f"   Balance score: {max(0, balance_score):.3f} (1.0 = perfectly uniform)")
        
        print(f"\n{'='*70}\n")