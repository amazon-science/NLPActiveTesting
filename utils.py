import os
import json
import numpy as np
from typing import List, Dict, Tuple, Union
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.manifold import TSNE
import random
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import argparse
import re
from rouge_score import rouge_scorer, scoring
from pathlib import Path

from data import load_data

def extract_metric_values(json_path: str, metric_name: str = 'accuracy') -> Dict[str, List[float]]:
    """
    Extract metric values for all budgets from a single method's JSON file.
    
    Args:
        json_path: Path to the JSON file.
        metric_name: Name of the metric to extract.
        
    Returns:
        Dict mapping budget (float) to list of values across seeds.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metrics_by_budget = {}
    
    for seed_data in data:
        first_method = list(seed_data.keys())[0]
        for budget, metrics in seed_data[first_method].items():
            budget_float = float(budget)
            if budget_float not in metrics_by_budget:
                metrics_by_budget[budget_float] = []
            if metric_name in metrics:
                metrics_by_budget[budget_float].append(metrics[metric_name])
    
    return metrics_by_budget


def statistical_comparison_methods(
    folder_path: str,
    metric_names: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    methods_to_ignore: List[str] = []
) -> pd.DataFrame:
    """
    Perform pairwise statistical comparisons between all methods and Random baseline.
    
    Uses t-tests, Mann-Whitney U tests, and computes Cohen's d effect sizes.
    Applies Bonferroni correction for multiple comparisons.
    
    Args:
        folder_path: Path to folder containing JSON result files.
        metric_names: List of metrics to compare.
        methods_to_ignore: Method names to exclude from analysis.
        
    Returns:
        DataFrame with columns: method1, method2, metric, budget, means, stds,
        p-values, effect sizes, and significance flags.
    """
    mapping = get_mapping()
    inverse_mapping = {v: k for k, v in mapping.items()}
    
    # Load all methods
    methods_data = {}
    for json_file in os.listdir(folder_path):
        if not json_file.endswith('.json'):
            continue
            
        method_name = decode_filename(json_file.replace('.json', ''), inverse_mapping)
        if method_name in methods_to_ignore:
            continue
            
        json_path = os.path.join(folder_path, json_file)
        methods_data[method_name] = json_path
    
    if len(methods_data) < 2:
        print(f"Not enough methods to compare in {folder_path}")
        return None
    
    # Perform pairwise comparisons
    results = []
    method_names_list = list(methods_data.keys())
    method_names_list.remove('Random')

    for i in range(len(method_names_list)):
        method1_name = method_names_list[i]
        method2_name = 'Random'
        
        for metric_name in metric_names:
            # Extract metrics for both methods
            metrics1 = extract_metric_values(methods_data[method1_name], metric_name)
            metrics2 = extract_metric_values(methods_data[method2_name], metric_name)
            
            # Find common budgets
            common_budgets = set(metrics1.keys()) & set(metrics2.keys())
            
            for budget in sorted(common_budgets):
                values1 = np.array(metrics1[budget])
                values2 = np.array(metrics2[budget])
                
                # Skip if not enough samples
                if len(values1) < 2 or len(values2) < 2:
                    continue
                
                # Calculate statistics
                mean1, std1 = values1.mean(), values1.std()
                mean2, std2 = values2.mean(), values2.std()
                
                # T-test
                if len(values1) == len(values2):
                    t_stat, p_value = stats.ttest_rel(values1, values2)
                    test_type = "paired"
                else:
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    test_type = "unpaired"
                
                # Mann-Whitney U test (non-parametric alternative)
                try:
                    u_stat, u_pvalue = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                except ValueError:
                    u_stat, u_pvalue = np.nan, np.nan
                
                # Cohen's d (effect size)
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                
                # Determine effect size category
                if abs(cohens_d) < 0.2:
                    effect_size = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect_size = "small"
                elif abs(cohens_d) < 0.8:
                    effect_size = "medium"
                else:
                    effect_size = "large"
                
                results.append({
                    'method1': method1_name,
                    'method2': method2_name,
                    'metric': metric_name,
                    'budget': int(budget),
                    'mean1': mean1,
                    'std1': std1,
                    'mean2': mean2,
                    'std2': std2,
                    'difference': mean1 - mean2,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'u_statistic': u_stat,
                    'u_pvalue': u_pvalue,
                    'cohens_d': cohens_d,
                    'effect_size': effect_size,
                    'test_type': test_type,
                    'significant_005': p_value < 0.05,
                    'significant_001': p_value < 0.01,
                })
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    # Apply Bonferroni correction
    n_tests = len(df)
    bonferroni_alpha = 0.05 / n_tests
    df['bonferroni_significant'] = df['p_value'] < bonferroni_alpha
    df['bonferroni_alpha'] = bonferroni_alpha
    
    return df


def save_statistical_results(
    df: pd.DataFrame,
    folder_path: str,
    dataset: str
) -> None:
    """
    Save statistical comparison results to CSV and generate summary report.
    
    Creates statistical_analysis directory with CSV results, text summary,
    and LaTeX table of significant differences.
    
    Args:
        df: DataFrame with statistical results from statistical_comparison_methods.
        folder_path: Path where to save results.
        dataset: Dataset name for labeling outputs.
    """
    if df is None or df.empty:
        return
    
    # Create output directory
    parts = folder_path.split(os.sep)
    if "synthetic" in folder_path:
        distribution = parts[-2]
        num_samples = parts[-1]
        stats_dir = os.path.join(
            folder_path.split('/')[0], "statistical_analysis",
            dataset, distribution, num_samples
        )
    else:
        num_samples = parts[-1]
        stats_dir = os.path.join(
            parts[0], parts[-4], "statistical_analysis",
            dataset, num_samples
        )
    
    os.makedirs(stats_dir, exist_ok=True)
    
    # Save full results
    csv_path = os.path.join(stats_dir, "statistical_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate and save summary
    summary_path = os.path.join(stats_dir, "statistical_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL SIGNIFICANCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Significant differences (p < 0.05)
        sig_results = df[df['significant_005']].sort_values(['metric', 'budget'])
        if not sig_results.empty:
            f.write("SIGNIFICANT DIFFERENCES (p < 0.05):\n")
            f.write("-" * 80 + "\n")
            for _, row in sig_results.iterrows():
                f.write(f"\n{row['metric'].upper()} at budget {row['budget']}:\n")
                f.write(f"  {row['method1']} vs {row['method2']}\n")
                f.write(f"  Mean difference: {row['difference']:.4f}\n")
                f.write(f"  p-value: {row['p_value']:.4f}\n")
                f.write(f"  Cohen's d: {row['cohens_d']:.4f} ({row['effect_size']})\n")
        else:
            f.write("No significant differences found at p < 0.05\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Bonferroni corrected results
        bonf_results = df[df['bonferroni_significant']].sort_values(['metric', 'budget'])
        if not bonf_results.empty:
            f.write(f"BONFERRONI CORRECTED SIGNIFICANT DIFFERENCES (α = {df['bonferroni_alpha'].iloc[0]:.6f}):\n")
            f.write("-" * 80 + "\n")
            for _, row in bonf_results.iterrows():
                f.write(f"\n{row['metric'].upper()} at budget {row['budget']}:\n")
                f.write(f"  {row['method1']} vs {row['method2']}\n")
                f.write(f"  Mean difference: {row['difference']:.4f}\n")
                f.write(f"  p-value: {row['p_value']:.6f}\n")
                f.write(f"  Cohen's d: {row['cohens_d']:.4f} ({row['effect_size']})\n")
        else:
            f.write("No significant differences after Bonferroni correction\n")
    
    
    # Generate LaTeX table for significant results
    generate_statistical_latex_table(df, stats_dir, dataset)


def generate_statistical_latex_table(
    df: pd.DataFrame,
    output_dir: str,
    dataset: str
) -> None:
    """
    Generate LaTeX table showing Bonferroni-corrected significant comparisons.
    
    Args:
        df: DataFrame with statistical results.
        output_dir: Directory to save the .tex file.
        dataset: Dataset name for table caption.
    """
    sig_results = df[df['bonferroni_significant']].sort_values(['metric', 'budget'])
    
    if sig_results.empty:
        return
    
    latex = "\\begin{table}[h]\n\\centering\n\\small\n"
    latex += "\\begin{tabular}{llccccc}\n\\hline\n"
    latex += "Metric & Budget & Method 1 & Method 2 & Diff. & p-value & Effect Size \\\\\n\\hline\n"
    
    for _, row in sig_results.iterrows():
        latex += f"{row['metric']} & {int(row['budget'])} & "
        latex += f"{row['method1']:.10s} & {row['method2']:.10s} & "
        latex += f"{row['difference']:.3f} & {row['p_value']:.4f} & "
        latex += f"{row['effect_size']} \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n"
    latex += f"\\caption{{Statistically significant differences for {dataset} "
    latex += f"(Bonferroni corrected, $\\alpha = {sig_results['bonferroni_alpha'].iloc[0]:.4f}$)}}\n"
    latex += f"\\label{{tab:stats_{dataset}}}\n"
    latex += "\\end{table}"
    
    latex_path = os.path.join(output_dir, "statistical_comparison.tex")
    with open(latex_path, 'w') as f:
        f.write(latex)
    


def compute_number_english_samples(inputs, threshold):
    """
    Count samples below and above threshold for multilingual dataset splits.
    
    Args:
        inputs: List of sample indices.
        threshold: Cutoff index separating languages.
        
    Returns:
        Tuple of (count_below_threshold, count_above_threshold).
    """
    english = 0
    for item in inputs:
        if item < threshold:
            english +=1
            
    return english, len(inputs) - english

def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across multiple libraries.

    Args:
        seed: Integer seed for random number generation
    """
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_complete_predictions(dataset_name, predictor_name, language="spanish"):
    """
    Load pre-computed predictions from cache file.
    
    Args:
        dataset_name: Name of the dataset.
        predictor_name: Name of the predictor model.
        
    Returns:
        Array of predictions, or None if file not found.
    """
    if dataset_name not in ["multilingual", "xlsum"]:
        path = os.path.join("autoeval", predictor_name, f"all_predictions_{dataset_name}.npy")
    else:
        path = os.path.join("autoeval", predictor_name, f"all_predictions_{dataset_name}_{language}.npy")
    if not os.path.exists(path):
        print(f"File {path} not available. You must generate the predictions.")
        return None
    else:
        all_predictions = np.load(path, allow_pickle=True)
        return all_predictions

def shuffle_data(texts, labels, all_predictions):
    """
    Shuffle texts, labels, and predictions together maintaining alignment.
    
    Args:
        texts: List of input texts.
        labels: List of corresponding labels.
        all_predictions: List of model predictions.
        
    Returns:
        Tuple of (shuffled_texts, shuffled_labels, shuffled_predictions).
    """
    combined = list(zip(texts, labels, all_predictions))
    random.shuffle(combined)
    shuffled_texts, shuffled_labels, shuffled_predictions = zip(*combined)
    return list(shuffled_texts), list(shuffled_labels), list(shuffled_predictions)


def compute_class_metrics(
    labels,
    indices,
    pipeline_name: str = None,
) -> Tuple[float, float]:
    """
    Compute minority class precision and recall for selected samples.
    
    For NER/POS tasks, operates at sequence level (sequence contains minority tag).
    For classification, operates at sample level.
    
    Args:
        labels: List of labels (flat for classification, nested for NER).
        indices: Selected sample indices.
        pipeline_name: Task type ('ner', 'pos', or None for classification).
        
    Returns:
        Dict with 'unbalance_precision' and 'unbalance_recall'.
    """
    if pipeline_name in ["ner", "pos"]:
        # Handle NER case: flatten the nested lists of labels
        flat_labels = []
        for label_seq in labels:
            flat_labels.extend(label_seq)

        # Convert to numpy array for efficient counting
        labels_array = np.array(flat_labels)

        # Find the minority class
        unique_classes, counts = np.unique(labels_array, return_counts=True)
        minority_class = unique_classes[np.argmin(counts)]

        # Find sequences containing the minority class
        minority_indices = []
        for i, label_seq in enumerate(labels):
            # If any token in the sequence is of minority class, include the sequence
            if minority_class in label_seq:
                minority_indices.append(i)

        # Find selected sequences that contain minority class
        selected_minority = np.intersect1d(indices, minority_indices)

    else:
        # Handle simple classification case
        labels_array = np.array(labels)

        # Count instances of each class
        class_counts = [
            np.sum(labels_array == i) for i in range(len(np.unique(labels_array)))
        ]
        # Identify minority class and its instances
        minority_class = np.argmin(class_counts)
        minority_indices = np.where(labels_array == minority_class)[0]

        # Find selected instances of minority class
        selected_minority = np.intersect1d(indices, minority_indices)

    # Calculate unbalance metrics
    try:
        unbalance_precision = round(len(selected_minority) / len(indices), 4)
    except ZeroDivisionError:
        unbalance_precision = 0
    # Recall: proportion of minority class samples that were selected
    try:
        unbalance_recall = round(len(selected_minority) / len(minority_indices), 4)
    except ZeroDivisionError:
        unbalance_recall = 0

    return {
        "unbalance_precision": unbalance_precision,
        "unbalance_recall": unbalance_recall,
    }


def save_json(
    file_name: str,
    data: dict,
    dataset_name: str,
    base_folder: str = "results",
    additional_params: dict = None,
) -> None:
    """
    Save results to JSON file with organized directory structure.
    
    Args:
        file_name: Name of the JSON file (without extension).
        data: Dictionary containing results to save.
        dataset_name: Name of the dataset for directory organization.
        base_folder: Base directory for results.
        additional_params: Optional params like num_samples, language_prior.
    """

    # Create base directories
    os.makedirs(f"{base_folder}/json", exist_ok=True)
    os.makedirs(f"{base_folder}/json/{dataset_name}", exist_ok=True)
    path = f"{base_folder}/json/{dataset_name}"
    
    # Handle additional parameters for synthetic data
    if additional_params is not None:
        
        if 'num_samples' in additional_params.keys():

            os.makedirs(
                f"{base_folder}/json/{dataset_name}/{additional_params['num_samples']}",
                exist_ok=True,
            )
        path = f"{base_folder}/json/{dataset_name}/{additional_params['num_samples']}"
        #data[0]['class_distribution'] = additional_params['class_distribution']
        if 'language_prior' in additional_params.keys():
            data[0]['language_prior'] = additional_params['language_prior']
            data[0]['stats_languages'] = additional_params['stats_languages']
    # Save data to JSON file
    with open(os.path.join(path, f"{file_name}.json"), "w") as f:
        json.dump(data, f)


def get_mapping() -> dict:
    """
    Get mapping dictionary for method and parameter names to short codes.

    Returns:
        dict: Mapping of full names to short codes
    """
    return {
        # Testing methods
        "Random": "0",
        "Coverage": "1",
        "Distance": "2",
        "Uncertainty": "3",
        "Agreement": "4",
        "Surrogate": "5",
        "Diversity" : "6",
        "Diffuse" : "7",
        "Stratified" : "8",
        # Clustering methods
        "dbscan": "a",
        "kmeans": "b",
        "shift": "c",
        # Selection strategies
        "clusters": "y",
        "centroids": "x",
        "max_dist": "z",
        # Acquisition functions
        "gaussian_prior": "s",
        "mutual_information": "t",
        # Model types
        "SVM": "r",
        "RF": "p",
    }


def map_methods_to_id(results: dict, **kwargs) -> str:
    """
    Convert method names and parameters to a compact identifier string.

    Args:
        results: Dictionary containing results
        **kwargs: Additional parameters to include in the identifier

    Returns:
        str: Compact identifier string
    """
    mapping = get_mapping()
    output = ""

    # Map method names
    for key in results.keys():
        try:
            output += mapping[key]
        except KeyError:
            continue
    # Handle additional parameters
    if kwargs:
        for value in kwargs.values():
            try:
                if value != "-1":
                    output += f"_{mapping[value]}"
            except (KeyError, TypeError):
                continue
        if kwargs["n_clusters"] != -1:
            output += f"_{kwargs['n_clusters']}"
    return output


def preprocess_text(text: str, length: int = 512) -> str:
    """
    Preprocess text by truncating to specified length.

    Args:
        text: Input text to preprocess
        length: Maximum number of words to keep

    Returns:
        str: Preprocessed text
    """
    return " ".join(text.split()[:length])


def evaluate_metrics(
    y_true: List[int],
    y_pred: List[int],
    results_not_active: dict = None,
    pipeline_name: str = None,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for classification or sequence labeling results.
    
    Args:
        y_true: List of true labels (nested for NER/POS).
        y_pred: List of predicted labels.
        results_not_active: If provided, returns absolute differences from these values.
        pipeline_name: Task type ('ner', 'pos', or None for classification).
        
    Returns:
        Dict with 'accuracy', 'precision', 'recall', 'f1' (or their differences).
    """

    # Calculate base metrics
    if pipeline_name == "ner":
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(y_true)
        y_pred = mlb.transform(y_pred)
    
    if pipeline_name not in ["ner", "pos"]:
        accuracy = accuracy_score(y_true, y_pred)
    else:
        y_true = [label for seq in y_true for label in seq]
        y_pred = [label for seq in y_pred for label in seq]
        accuracy = accuracy_score(y_true, y_pred)
        
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # If reference results provided, compute differences
    if results_not_active is not None:
        return {
            "accuracy": abs(results_not_active["accuracy"] - accuracy),
            "precision": abs(results_not_active["precision"] - precision),
            "recall": abs(results_not_active["recall"] - recall),
            "f1": abs(results_not_active["f1"] - f1),
        }

    # Otherwise return raw metrics
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def format_distribution(distribution: str) -> str:
    """
    Format the distribution string into LaTeX array notation.

    Args:
        distribution (str): Distribution string (e.g., "0.7_0.3" or "0.33_0.33_0.34")

    Returns:
        str: Formatted LaTeX string (e.g., "$[0.7, 0.3]$" or "$[0.33, 0.33, 0.34]$")
    """
    # Split the distribution string and format as array
    classes = distribution.split("_")
    return f"$[{', '.join(classes)}]$"


def read_result_file(file_path: str) -> Dict[int, List[Tuple[float, float]]]:
    """
    Read a single result file and extract the metrics.

    Args:
        file_path (str): Path to the result file

    Returns:
        Dict[int, List[Tuple[float, float]]]: Dictionary mapping n_samples to list of (mean, std) tuples

    Example:
        {
            20: [(0.123, 0.045), (0.234, 0.056), (0.345, 0.067)],  # (precision, recall, f1)
            50: [(0.234, 0.056), (0.345, 0.067), (0.456, 0.078)],
            ...
        }
    """
    results = {}
    with open(file_path, "r") as f:
        for line in f:
            # Remove trailing \\ and split by &
            parts = line.strip().rstrip("\\").split("&")
            n_samples = float(parts[0].strip())

            # Extract numerical values using regex
            values = []
            for part in parts[1:]:
                match = re.search(r"(-?\d+\.\d+)\s*\$\\pm\$\s*(-?\d+\.\d+)", part)
                if match:
                    mean, std = float(match.group(1)), float(match.group(2))
                    values.append((mean, std))
            results[n_samples] = values
    return results


def find_best_methods(
    results_dict: Dict[str, Dict[int, List[Tuple[float, float]]]],
    n_samples: int,
    metric_idx: int,
) -> Tuple[str, str]:
    """
    Find the best and second best methods for a given metric at a specific n_samples.

    Args:
        results_dict: Dictionary containing results for all methods
        n_samples: Number of samples to compare
        metric_idx: Index of the metric (0: precision, 1: recall, 2: f1)

    Returns:
        Tuple[str, str]: Names of the best and second best performing methods
    """
    performances = []

    for method, results in results_dict.items():
        mean, _ = results[n_samples][metric_idx]
        if mean != -1:  # Ignore -1 values
            performances.append((mean, method))

    # Sort by performance (descending)
    performances.sort(reverse=True)

    if len(performances) == 0:
        return None, None
    elif len(performances) == 1:
        return performances[0][1], None
    else:
        return performances[0][1], performances[1][1]


def generate_latex_table(
    results_dict: Dict[str, Dict[int, List[Tuple[float, float]]]],
    dataset: str,
    distribution: str = None,
    num_samples: str = None,
) -> str:
    """
    Generate LaTeX table from results with bold/underline formatting for best values.
    
    Args:
        results_dict: Dict mapping method names to their results.
        dataset: Dataset name for caption.
        distribution: Optional class distribution string.
        num_samples: Optional sample count string.
        
    Returns:
        Complete LaTeX table string.
    """

    latex = "\\begin{table}[h]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{l|"

    num_methods = len(results_dict)
    latex += "ccc|" * num_methods
    latex = latex.rstrip("|") + "}\n\\hline\n"

    latex += "N samples"
    for method in results_dict.keys():
        method_name = method.replace("_", " ")
        latex += f" & \\multicolumn{{3}}{{c|}}{{{method_name}}}"
    latex = latex.rstrip("|")
    latex += " \\\\\n"

    latex += " & "
    metrics = ["Precision", "F1"] * num_methods#"Recall", "F1"] * num_methods
    latex += " & ".join(metrics)
    latex += " \\\\\n\\hline\n"

    n_samples = sorted(list(next(iter(results_dict.values())).keys()))
    for n in n_samples:
        row = [str(n)]

        # Find best and second best methods for each metric at current n_samples
        best_methods = [
            find_best_methods(results_dict, n, metric_idx) for metric_idx in range(3)
        ]

        for method in results_dict.keys():
            for metric_idx, (mean, std) in enumerate(results_dict[method][n]):
                if mean == -1 and std == -1:
                    row.append("-")
                else:
                    value = f"{mean:.3f} $\\pm$ {std:.3f}"
                    best, second_best = best_methods[metric_idx]

                    if method == best:
                        value = f"\\textbf{{{value}}}"
                    elif method == second_best:
                        value = f"\\underline{{{value}}}"

                    row.append(value)

        latex += " & ".join(row) + " \\\\\n"

    latex += "\\hline\n\\end{tabular}}\n"
    if distribution is not None:
        latex += f"\\caption{{Results for {dataset} dataset with {format_distribution(distribution)} distribution and {num_samples} samples. "
    else:
        latex += f"\\caption{{Results for {dataset} dataset. "
    latex += "Bold values indicate best performance and underlined values indicate second best performance for each metric.}\n"
    if distribution is not None:
        latex += f"\\label{{tab:{dataset}_{distribution}_{num_samples}}}\n"
    else:
        latex += f"\\label{{tab:{dataset}}}\n"

    latex += "\\end{table}"

    return latex


def process_results_structure(base_path: str = "synthetic/overleaf") -> None:
    """
    Process entire results directory structure and generate LaTeX tables.
    
    Iterates through dataset/distribution/samples directories and creates
    tables in a latex_tables output directory.
    
    Args:
        base_path: Path to base directory containing results.
    """
    base_path = Path(base_path)

    # Create output directory for LaTeX tables
    output_dir = base_path / "latex_tables"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all directories
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        for dist_dir in dataset_dir.iterdir():
            if not dist_dir.is_dir():
                continue

            for samples_dir in dist_dir.iterdir():
                if not samples_dir.is_dir():
                    continue

                # Process all txt files in the current directory
                results_dict = {}
                incomplete_methods = []

                for result_file in samples_dir.glob("*.txt"):
                    method_name = result_file.stem
                    try:
                        file_results = read_result_file(str(result_file))
                        if file_results:  # Only add if we got valid results
                            results_dict[method_name] = file_results
                        else:
                            incomplete_methods.append(method_name)
                    except Exception:
                        incomplete_methods.append(method_name)

                if incomplete_methods:
                    print(
                        f"Warning: Skipping incomplete results for methods: {', '.join(incomplete_methods)} "
                        f"in {dataset_dir.name}/{dist_dir.name}/{samples_dir.name}"
                    )

                if results_dict:  # Only proceed if we have some valid results
                    try:
                        # Generate LaTeX table
                        latex_table = generate_latex_table(
                            results_dict,
                            dataset_dir.name,
                            dist_dir.name,
                            samples_dir.name,
                        )

                        # Save the table
                        output_file = (
                            output_dir
                            / f"{dataset_dir.name}_{dist_dir.name}_{samples_dir.name}.tex"
                        )
                        with open(output_file, "w") as f:
                            f.write(latex_table)

                        print(f"Generated table: {output_file}")

                    except Exception as e:
                        print(
                            f"Error generating table for {dataset_dir.name}/{dist_dir.name}/{samples_dir.name}: {str(e)}"
                        )


def save_exp(
    all_results: List[Dict],
    dataset_name: str = "imdb",
    synthetic: bool = False,
    **kwargs,
) -> None:
    """
    Save experimental results to JSON files with appropriate directory structure.

    Args:
        all_results: List of dictionaries containing experimental results
        dataset_name: Name of the dataset
        synthetic: Whether the results are from synthetic data
        **kwargs: Additional parameters for file organization
    """
    os.makedirs(kwargs["base_folder"], exist_ok=True)
    base_folder = kwargs["base_folder"]

    if "language_prior" in kwargs.keys():

        save_json(
        file_name=map_methods_to_id(all_results[0][list(all_results[0].keys())[0]], **kwargs),
        data=all_results,
        dataset_name=dataset_name,
        base_folder=base_folder,
        additional_params=kwargs,
    )
        
    else:
        save_json(
            file_name=map_methods_to_id(all_results[0], **kwargs),
            data=all_results,
            dataset_name=dataset_name,
            base_folder=base_folder,
            additional_params=kwargs,
        )


def model_name_map(original_name, inverse=False) -> str:
    if not inverse:
        mapping = {
            "bert-base-multilingual-cased": "bert",
            "distilbert-base-multilingual-cased": "distilbert",
            "Qwen/Qwen3-Embedding-0.6B": "qwen",
            "NovaSearch/stella_en_1.5B_v5" : "stella",
        }
    else:
        mapping = {
            "bert": "bert-base-multilingual-cased",
            "distilbert": "distilbert-base-multilingual-cased",
            "qwen": "Qwen/Qwen3-Embedding-0.6B",
            "stella" : "NovaSearch/stella_en_1.5B_v5",
        }
    return mapping[original_name]


def compute_rouge(predictions, references, rouge_types=None, use_stemmer=False):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        aggregator.add_scores(score)

    result = aggregator.aggregate()

    final = {}
    for key in result:
        mid = result[key].mid
        final[f"{key}_precision"] = mid.precision.item()
        final[f"{key}_recall"]    = mid.recall.item()

    return final


def decode_filename(filename: str, reverse_mapping: Dict[str, str], separator = '-') -> str:
    """
    Convert encoded filename back to human-readable method name.

    Args:
        filename: Encoded filename to decode
        reverse_mapping: Dictionary mapping codes back to method names

    Returns:
        str: Human-readable method name
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    # Decode main method name
    if parts[0] in reverse_mapping:
        method_name = reverse_mapping[parts[0]]
    else:
        method_name = parts[0]

    # Decode additional parameters
    if len(parts) > 1:
        for i in range(1, len(parts)):
            if parts[i] in reverse_mapping:
                method_name += f" {separator} {reverse_mapping[parts[i]]}"
            else:
                method_name += f" ({parts[i]})"
    return method_name


def extract_metrics(
    data: List[Dict], metric_name: str
) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract metric values and statistics from experimental data.

    Args:
        data: List of dictionaries containing experimental results
        metric_name: Name of the metric to extract

    Returns:
        tuple: (split_sizes, mean_metric_values, std_metric_values)
    """
    all_metrics = defaultdict(list)

    # Collect metrics for each split size
    for item in data:
        for method_name, method_data in item.items():
            for split_size, metrics in method_data.items():
                split_size_float = float(split_size)
                if split_size_float != 1.0:  # Exclude full dataset results
                    all_metrics[split_size_float].append(metrics[metric_name])

    # Calculate statistics
    split_sizes = sorted(all_metrics.keys())
    metric_values = [np.mean(all_metrics[size]) for size in split_sizes]
    std_values = [np.std(all_metrics[size]) for size in split_sizes]

    return split_sizes, metric_values, std_values

def get_colors() -> dict:
    """
    Assign colors to methods for reproducibility.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#17becf', '#e377c2', '#8c564b', '#bcbd22', '#7f7f7f']
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
    return {
        "Random" : (colors[0],markers[0]),
        "Coverage - max_dist" : (colors[4],markers[4]),
        "Surrogate - SVM" : (colors[2],markers[2]),
        "Surrogate - RF" : (colors[5],markers[5]),
        "Agreement" : (colors[1],markers[1]),
        "Stratified" : (colors[3],markers[3]),
        "Uncertainty - gaussian_prior" : (colors[6],markers[6]),
        "Uncertainty - mutual_information" : (colors[7],markers[7]),
        "en" : (colors[0],"+"),
        "italian" : (colors[1],"*"),
        "mixed" : (colors[2], "o"),
    }

def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    methods: list = ["pca"],
    n_components: int = 2,
    perplexity: int = 30,
    dataset_name: str = "imdb",
    results_dir: str = "results/embeddings",
) -> None:
    """
    Create dimensionality reduction visualizations (PCA/t-SNE) of embeddings.
    
    Args:
        embeddings: High-dimensional embeddings array.
        labels: Labels for color-coding points.
        methods: List of reduction methods ('pca', 'tsne').
        n_components: Number of dimensions for reduction.
        perplexity: Perplexity parameter for t-SNE.
        dataset_name: Dataset name for filename.
        results_dir: Directory to save plots.
    """
    for method in methods:
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components, perplexity=perplexity, random_state=42
            )
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        reduced_embeddings = reducer.fit_transform(embeddings)

        plt.figure(figsize=(12, 8))
        plt.scatter(
            reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis"
        )
        plt.colorbar(label="Label")
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.title(f"{method.upper()} Plot of Embeddings")

        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(f"{results_dir}/{method}_{dataset_name}.png")
        plt.close()


def compute_unbalance_results(
    method: str,
    class_distribution: str = None,
    num_samples: int = None,
    dataset: str = "imdb",
    base_folder: str = "synthetic/json",
    save: bool = True,
    original_name=False
) -> dict:
    """
    Compute and optionally save minority class metrics from experimental results.
    
    Args:
        method: Method name or comma-separated methods.
        class_distribution: Class distribution string (for synthetic experiments).
        num_samples: Number of samples used.
        dataset: Dataset name.
        base_folder: Base directory containing results.
        save: Whether to save results to CSV.
        original_name: If True, use method as literal filename.
        
    Returns:
        Dict mapping n_samples to metric statistics.
    """
    # Process method name
    mapping = get_mapping()

    # Handle method name processing
    if not original_name:
        if "," in method:
            approaches = method.split(",")
            file_name = (
                "_".join(
                    mapping.get(approach, approach) if not approach.isdigit() else approach
                    for approach in approaches
                )
                + ".json"
            )
        else:
            file_name = f"{mapping[method]}.json"
    else:
        file_name = method

    
    # Construct file path based on available parameters
    if class_distribution is not None:
        # Original nested structure
        file_path = os.path.join(
            base_folder, dataset, class_distribution, str(num_samples), file_name
        )
    if num_samples is not None:
        file_path = os.path.join(
            base_folder, dataset, str(num_samples), file_name
        )
    
    else:
        # Flat structure
        file_path = os.path.join(base_folder, dataset, file_name)

    # Load and process results
    try:
        with open(file_path, "r") as f:
            
            experiments = json.load(f)

        # Extract sample sizes and initialize results
        n_samples = list(experiments[0][list(experiments[0].keys())[0]].keys())
        unbalance_metrics = {
            n: {"precision": [], "f1": []} for n in n_samples
            #n: {"precision": [], "recall": [], "f1": []} for n in n_samples
        }

        max_n = max(float(x) for x in n_samples)

        # Calculate metrics for each experiment
        for exp in experiments:
            for n in n_samples:
                if float(n) != max_n:
                    recall = exp[list(experiments[0].keys())[0]][n]["unbalance_recall"]
                    precision = exp[list(experiments[0].keys())[0]][n][
                        "unbalance_precision"
                    ]
                    #unbalance_metrics[n]["recall"].append(recall)
                    unbalance_metrics[n]["precision"].append(precision)
                    try:
                        f1 = (2 * precision * recall) / (precision + recall)
                    except ZeroDivisionError:
                        f1 = 0
                    unbalance_metrics[n]["f1"].append(f1)

        # Compute statistics
        results = {}
        for n in n_samples:
            if float(n) != max_n:
                results[n] = {
                    metric: {"mean": np.mean(values), "std": np.std(values)}
                    for metric, values in unbalance_metrics[n].items()
                }
            else:
                results[n] = {
                    metric: {"mean": -1, "std": -1}
                    for metric in ["precision", "f1"]#["recall", "precision", "f1"]
                }

        # Save results if requested
        if save:
            save_results_to_csv(
                results, method, dataset, class_distribution, num_samples, base_folder
            )
        else:
            print_results(results)
    except Exception:
        print(file_path)
        results = None

    return results


def save_results_to_csv(
    results: dict,
    method: str,
    dataset: str,
    class_distribution: str,
    num_samples: str,
    base_folder: str,
) -> None:
    """
    Save computed results to CSV and generate Overleaf-compatible table.
    
    Args:
        results: Dict of computed metrics by sample size.
        method: Method name for filename.
        dataset: Dataset name.
        class_distribution: Class distribution string.
        num_samples: Number of samples.
        base_folder: Base directory for output.
    """
    flat_results = []
    for n, metrics in results.items():
        row = {"n_samples": n}
        for metric, stat in metrics.items():
            row[f"{metric}_mean"] = round(stat["mean"], 4)
            row[f"{metric}_std"] = round(stat["std"], 4)
        flat_results.append(row)

    df = pd.DataFrame(flat_results)
    # Create output directory
    if class_distribution is not None:
        output_dir = os.path.join(
            base_folder.replace("json", "results"),
            dataset,
            class_distribution,
            str(num_samples),
        )
        overleaf_dir = os.path.join(
            base_folder.replace("json", "overleaf"),
            dataset,
            class_distribution,
            str(num_samples),
        )
    else:
        output_dir = os.path.join(base_folder.replace("json", "results"), dataset)
        overleaf_dir = os.path.join(base_folder.replace("json", "overleaf"), dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV file
    output_file = os.path.join(output_dir, f"{method.replace(',', '_')}.csv")
    df.to_csv(output_file, index=False)

    os.makedirs(overleaf_dir, exist_ok=True)
    table_overleaf(
        results=results, base_folder=overleaf_dir, method_name=method.replace(",", "_")
    )


def table_overleaf(results: dict, base_folder: str, method_name: str) -> None:
    """
    Generate Overleaf-compatible table format (mean ± std per row).
    
    Args:
        results: Dict mapping n_samples to metric statistics.
        base_folder: Directory to save output.
        method_name: Method name for filename.
    """

    with open(os.path.join(base_folder, f"{method_name}.txt"), "w") as f:
        for n in sorted(results.keys(), key=float):
            f.write(f"{n}")
            for k, value in results[n].items():
                f.write(f" & {value['mean']:.3f} $\pm$ {value['std']:.3f}")
            f.write(" \\\\\n")


def print_results(results: dict) -> None:
    """
    Print formatted results to console.
    """
    for n in sorted(results.keys(), key=int):
        print(f"------------------ N={n} --------------------")
        for k, value in results[n].items():
            print(f"{k}: {value['mean']:.4f} ± {value['std']:.4f}")


def to_distr(input_distr: str) -> str:
    """
    Convert a distribution string from underscore format to array notation.

    Args:
        input_distr (str): Distribution string with values separated by underscores

    Returns:
        str: Distribution string in array notation [val1, val2, ...]
    """
    input_distr = input_distr.replace("_", ",")
    return f"[{input_distr}]"


def plot_sub_metrics_combined(
    folder_path: str,
    metric_name: str = "sub_accuracy",
    plot_compared_to_zero: bool = False,
    methods_to_ignore: list = [], #["Surrogate - RF", "Coverage - max_dist", "Surrogate - SVM"],
    # methods_to_ignore: list = ["Uncertainty - gaussian_prior", "Uncertainty - mutual_information",
    #                            "Surrogate - SVM", "Surrogate - RF"],
    for_final:bool=False
) -> None:
    """
    Create combined line plot of a metric across all methods in a folder.
    
    Generates plot with mean ± std bands for each method, saved to plots directory.
    
    Args:
        folder_path: Path to folder containing JSON result files.
        metric_name: Metric to plot (e.g., 'sub_accuracy', 'unbiased_accuracy').
        plot_compared_to_zero: If True, center unbiased metrics around zero line.
        methods_to_ignore: List of method names to exclude from plot.
    """
    plt.figure(figsize=(12, 8))

    mapping = get_mapping()
    inverse_mapping = {v: k for k, v in mapping.items()}

    # Define visual elements for different methods
    colors = get_colors()

    # Data structures to store results and track errors
    method_data: Dict[str, Dict[str, List[float]]] = {}

    # Process each JSON file in the folder
    for json_file in os.listdir(folder_path):
        if not json_file.endswith(".json"):
            continue

        method_name = json_file.replace(".json", "")
        json_path = os.path.join(folder_path, json_file)

        try:
            # Load JSON data
            with open(json_path, "r") as f:
                data = json.load(f)
        
            first_method = list(data[0].keys())[0]
            samples: List[int] = []
            values: List[float] = []
            stds: List[float] = []

            # Get sample sizes excluding reference
            maximum = max([int(k) for k in data[0][first_method].keys()])
            sample_sizes = sorted(
                [int(k) for k in data[0][first_method].keys() if k != maximum]
            )

            # Process each sample size
            for n in sample_sizes:
                try:
                    # Collect metric values across experiments
                    method_values = [
                        exp[first_method][str(n)][metric_name]
                        for exp in data
                        if str(n) in exp[first_method]
                        and metric_name in exp[first_method][str(n)]
                    ]

                    if method_values:
                        samples.append(n)
                        values.append(np.mean(method_values))
                        stds.append(np.std(method_values))

                except (KeyError, TypeError):
                    continue
            # Store results if valid data points exist
            if (
                samples
                and decode_filename(method_name, inverse_mapping)
                not in methods_to_ignore
            ):

                method_data[method_name] = {
                    "samples": samples,
                    "values": values,
                    "stds": stds,
                }
                


        except Exception:
            # Silently skip any method that causes errors
            continue
        

    if not method_data:
        print(f"Warning: No valid methods found in {folder_path}")
        return

    # Create plot for all valid methods
    for idx, (method_name, data) in enumerate(method_data.items()):
        try:
            method_label = inverse_mapping[method_name.split("_")[0]]
            if len(method_name.split("_")) > 1:
                method_label += f" - {inverse_mapping[method_name.split('_')[1]]}"
        except KeyError:
            method_label = method_name

        if "unbiased" in metric_name and plot_compared_to_zero:
            # 0 line as reference
            plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)

            values = np.array(data["values"])
            stds = np.array(data["stds"])

            combined_values = np.abs(values) + stds

            # Symmetric centering around 0
            half_combined = combined_values / 2

            plt.plot(
                data["samples"],
                half_combined,
                label=f"{method_label} (upper)",
                color=colors[method_label][0],
                marker=colors[method_label][1],
                linewidth=2,
                markersize=8,
            )

            plt.plot(
                data["samples"],
                -half_combined,
                label=f"{method_label} (lower)",
                color=colors[method_label][0],
                marker=colors[method_label][1],
                linewidth=2,
                markersize=8,
            )

            plt.fill_between(
                data["samples"],
                -half_combined,
                half_combined,
                color=colors[method_label][0],
                alpha=0.2,
            )
        else:
            plt.plot(
                data["samples"],
                data["values"],
                label=method_label,
                color=colors[method_label][0],
                marker=colors[method_label][1],
                linewidth=2,
                markersize=8,
            )

            plt.fill_between(
                data["samples"],
                np.array(data["values"]) - np.array(data["stds"]),
                np.array(data["values"]) + np.array(data["stds"]),
                color=colors[method_label][0],
                alpha=0.2,
            )

    # Configure plot styling
    plt.rcParams.update({'font.size': 20})
    plt.grid(True, linestyle="--", alpha=0.7)
    if not for_final:
        plt.xlabel("Number of Samples")
        plt.ylabel(f"Estimation Error")
        plt.legend()
    
    # Extract metadata from path
    parts = folder_path.split(os.sep)
    
    dataset_names = {
        "xlsum" : "XLSum",
        "cnn" : "CNN",
        "uniner" : "Universal NER",
        "wikineural" : "WikiNeural",
        "ud-ewt" : "UD-EWT",
        "sst2" : "SST2",
        "trec6" : "TREC6",
        "dbpedia" : "DBPedia",
    }

    if "synthetic" in folder_path:
        dataset = parts[-3]
        distribution = parts[-2]
        num_samples = parts[-1]
        plot_dir = os.path.join(
        f"{folder_path.split('/')[0]}/plots", dataset.replace('json',''), distribution, num_samples
    )
    else:
        dataset = parts[-2]
        num_samples = parts[-1]  
        
        try:
            d_name = dataset_names[dataset]
            
        except KeyError:
            d_name = dataset
        plt.title(f"{d_name} - B={num_samples}")      
        plot_dir = os.path.join(
        folder_path.split('/')[0], parts[-4], "plots", dataset.replace('json',''), num_samples)

    dataset = load_data(dataset_name=dataset,
                                     language="english")
    
    data_size = len(dataset[0])
    plt.xticks(data["samples"], [f"{i} ({((i/data_size)*100):.1f}%)" for i in data["samples"]], fontsize=16, rotation=45)
    
    os.makedirs(plot_dir, exist_ok=True)
    plot_name = f"{metric_name}_combined_plot.png"
    if len(methods_to_ignore) > 0:
        plot_name = f"{metric_name}_combined_plot_ignored.png"
    plt.savefig(os.path.join(plot_dir, plot_name), dpi=300, bbox_inches="tight")
    plt.close()


def process_all_folders(
    base_path: str = "synthetic/json",
    metric_type: str = "standard",
    methods_to_ignore: list = ["Uncertainty - gaussian_prior", "Uncertainty - mutual_information",
                                "Coverage - max_dist", "Surrogate - RF", "Diversity"],
    run_statistical_analysis: bool = True,
    for_final:bool=False,
) -> None:
    """
    Process all experiment folders and create combined plots for each metric.
    Optionally perform statistical analysis between methods.

    Args:
        base_path: Root path containing the experimental results structure
        metric_type: Type of metrics to process ("standard" or "unbiased")
        methods_to_ignore: List of method names to exclude from analysis
        run_statistical_analysis: Whether to perform statistical comparison between methods
    """
    
    if metric_type == "standard":
        metrics = ["sub_accuracy", "sub_precision", "sub_recall", "sub_f1"]
    elif metric_type == "unbiased":
        metrics = [
            "unbiased_accuracy",
            "unbiased_precision",
            "unbiased_recall",
            "unbiased_f1",
        ]
    elif metric_type == "summarization":
        metrics = [
            "unbiased_rouge_1_recall",
            "unbiased_rouge_1_precision",
        ]
    else:
        raise ValueError(f"Metric type {metric_type} not recognized.")
    
    mapping = get_mapping()
    inverse_mapping = {v: k for k, v in mapping.items()}
    
    # Walk through directory structure
    for dataset in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        if 'synthetic' in base_path:
            for dist in os.listdir(dataset_path):
                dist_path = os.path.join(dataset_path, dist)
                if not os.path.isdir(dist_path):
                    continue

                for n_samples in os.listdir(dist_path):
                    samples_path = os.path.join(dist_path, n_samples)
                    if not os.path.isdir(samples_path):
                        continue

                    # Create plots for each metric
                    for metric in metrics:
                        try:
                            plot_sub_metrics_combined(samples_path, metric, methods_to_ignore=methods_to_ignore,
                                                      for_final=for_final)
                            print(f"Created combined plot for {metric} in {samples_path}")
                        except Exception as e:
                            print(f"Error creating plot for {samples_path} and {metric}: {str(e)}")
                            continue
                    
                    if run_statistical_analysis:
                        try:
                            stat_df = statistical_comparison_methods(
                                samples_path,
                                metric_names=[m.replace('sub_', '').replace('unbiased_', '') 
                                            for m in metrics],
                                methods_to_ignore=methods_to_ignore
                            )
                            if stat_df is not None:
                                save_statistical_results(stat_df, samples_path, dataset)
                        except Exception as e:
                            print(f"Error in statistical analysis for {samples_path}: {str(e)}")
                            
        else:
            for n_samples in os.listdir(dataset_path):
                samples_path = os.path.join(dataset_path, n_samples)
                if not os.path.isdir(samples_path):
                    continue
                
                for method in os.listdir(samples_path):
                    if decode_filename(method, inverse_mapping) not in methods_to_ignore:
                        compute_unbalance_results(
                            method=method,
                            dataset=dataset,
                            num_samples=n_samples,
                            base_folder=base_path,
                            original_name=True,
                        )
                        
                # generate_latex_table_from_directory(
                #     os.path.join(base_path, "overleaf", dataset), dataset
                # )
                
                for metric in metrics:
                    try:
                        plot_sub_metrics_combined(samples_path, metric, methods_to_ignore=methods_to_ignore,
                                                  for_final=for_final)
                        if dataset == 'multilingual':
                            plot_multilingual(samples_path)
                        print(f"Created combined plot for {metric} in {dataset_path}")
                    except Exception as e:
                        print(f"Error creating plot for {samples_path} and {metric}: {str(e)}")
                        continue
                
                if run_statistical_analysis:
                    try:
                        stat_df = statistical_comparison_methods(
                            samples_path,
                            metric_names=[m.replace('sub_', '').replace('unbiased_', '') 
                                        for m in metrics],
                            methods_to_ignore=methods_to_ignore
                        )
                        if stat_df is not None:
                            save_statistical_results(stat_df, samples_path, dataset)
                    except Exception as e:
                        print(f"Error in statistical analysis for {samples_path}: {str(e)}")
        
def plot_multilingual(folder_path, exclude_budget=100):
    """
    Create a grouped bar plot with mean ± std of unbiased accuracy across seeds.

    Args:
        json_path (str): Path to the input JSON file.
        output_path (str): Path where the plot will be saved.
        exclude_budget (int): Budget value to exclude from the plot (default=100).
    """
    multilingual = None
    mapping = get_mapping()
    colors = get_colors()
    inverse_mapping = {v: k for k, v in mapping.items()}
    
    for json_file in os.listdir(folder_path):
        if not json_file.endswith(".json"):
            continue

        method_name = json_file.replace(".json", "")
        json_path = os.path.join(folder_path, json_file)

        # Load the JSON file
        with open(json_path, "r") as f:
            data = json.load(f)
        if 'language_prior' in data[0].keys():
                multilingual = data[0]['language_prior']
                del data[0]['language_prior']
                del data[0]['stats_languages']

        # Collect rows from all seeds
        rows = []
        for seed_entry in data:  # each element = one seed
            for method, strategies in seed_entry.items():
                for strategy, budgets in strategies.items():
                    for budget, metrics in budgets.items():
                        if not isinstance(metrics, float):
                            unbiased_accuracy = metrics.get("unbiased_accuracy")
                            # Exclude the specified budget value
                            if unbiased_accuracy is not None and int(budget) != exclude_budget:
                                rows.append([method, strategy, int(budget), unbiased_accuracy])

        # Convert into DataFrame
        df = pd.DataFrame(rows, columns=["method", "strategy", "budget", "unbiased_accuracy"])

        # Compute mean and std across seeds
        df_stats = df.groupby(["method", "strategy", "budget"])["unbiased_accuracy"].agg(["mean", "std"]).reset_index()

        # Prepare bar plot parameters
        methods = df_stats["method"].unique()
        budgets = sorted(df_stats["budget"].unique())
        bar_width = 0.25
        x = np.arange(len(budgets))

        plt.figure(figsize=(14, 9))

        # Draw bars for each method with error bars (std)
        for i, method in enumerate(methods):
            dataset, _, _, _, _ = load_data(dataset_name=folder_path.split(os.sep)[-2].replace('json',''),
                                     language="english")
            data_size = len(dataset)
            subset = df_stats[df_stats["method"] == method].sort_values("budget")
            plt.bar(
                x + i * bar_width,
                subset["mean"],
                yerr=subset["std"],
                width=bar_width,
                capsize=4,
                color=colors[method][0],
                hatch=colors[method][1],
                label=method,
            )

        # Customize axes

        #plt.xlabel("Number of samples")
        #plt.ylabel("Estimation Error")
        
        if multilingual is not None:
            plt.rcParams.update({'font.size': 22})
            name = decode_filename(method_name, inverse_mapping) + r" $- p=(" + str(round(multilingual,2)) + "," + str(round(1-multilingual, 2)) + r")$"
    
            plt.title(name)
        else:
            plt.title(decode_filename(method_name, inverse_mapping))
            #plt.legend()
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.xticks(x + bar_width * (len(methods) - 1) / 2, [f"{i} ({int((i.item()/data_size)*100)}%)" for i in budgets], fontsize=12)
        # Use logarithmic scale if values differ a lot
        plt.yscale("log")        
        
        parts = folder_path.split(os.sep)
        if "synthetic" in folder_path:
            dataset = parts[-3]
            distribution = parts[-2]
            num_samples = parts[-1]
            plot_dir = os.path.join(
            f"{folder_path.split('/')[0]}/plots", dataset.replace('json',''), distribution, num_samples
        )
        else:
            dataset = parts[-2]
            num_samples = parts[-1]        
            plot_dir = os.path.join(
            folder_path.split('/')[0], parts[-4], "plots", dataset.replace('json',''), num_samples)
        
        os.makedirs(plot_dir, exist_ok=True)
        plot_name = f"multilingual_{method_name}.png"
        plt.savefig(os.path.join(plot_dir, plot_name), format="png", dpi=800, bbox_inches="tight")
        plt.close()
        
def get_method_name_from_file(filename: str) -> str:
    """Extract method name from filename."""
    return os.path.splitext(filename)[0]

def compute_accuracy_predictor_only(data_folder: str = 'autoeval', predictor_name:str = ['claude', 'nova_pro', 'qwen']) -> dict:
    """
    Compute accuracy metrics for all predictors against ground truth labels.
    
    Args:
        data_folder: Path to folder containing prediction .npy files.
        predictor_name: List of predictor names to evaluate.
        
    Returns:
        Dict mapping dataset names to metric dictionaries.
    """
    # Initialize output dictionary to store results
    for predictor in predictor_name:
        print(f'-----------{predictor}-----------')
        output = {}
        complete_folder = os.path.join(data_folder, predictor)
        # Get list of all files in the specified folder
        files = os.listdir(complete_folder)
        # Process each prediction file
        for item in files:
            # Load predictions from .npy file
            
            # Extract dataset name from filename
            dataset_name = item.replace('.npy', '').replace('all_predictions_', '')
            if 'multilingual' in item:
                splitted = dataset_name.split('_')
                dataset_name = splitted[0]
                language = splitted[1]
            else:
                
                dataset_name = item.replace('.npy', '').replace('all_predictions_', '')
                language = None
                
            
            # Load corresponding dataset and its true labels
            dataset, _, y_name, _, _ = load_data(dataset_name, language=language)
            predictions = np.load(os.path.join(complete_folder,item))
            
            try:
                # Calculate performance metrics
                results = evaluate_metrics(predictions, dataset[y_name])
                
                # Store results in output dictionary
                output[dataset_name] = results
                
                # Print results in a formatted table row
                print(f"{dataset_name} & {results['accuracy']:.4f} & "
                    f"{results['recall']:.4f} & " #{results['precision']:.4f} & 
                    f"{results['f1']:.4f}")
                
            except ValueError:
                # Handle cases where metric computation fails
                print(f"{dataset_name} has problems. Length of predictions: {len(predictions)}")
                

def find_best_values(
    methods_data: dict, samples_percentage: float, metric_idx: int
) -> tuple:
    """
    Find best and second-best metric values across methods.
    
    Args:
        methods_data: Dictionary containing all methods' results.
        samples_percentage: Sample percentage to evaluate at.
        metric_idx: Index of the metric.
        
    Returns:
        Tuple of (best_value, second_best_value).
    """
    values = []
    for method in methods_data.keys():
        try:
            mean, std = methods_data[method][samples_percentage][metric_idx]
        except KeyError:
            mean, std = -1, -1
        if mean != -1:  # Ignore invalid values
            values.append(mean)

    if not values:
        return None, None

    values.sort(reverse=True)
    best = values[0]
    second_best = values[1] if len(values) > 1 else None

    return best, second_best


def generate_latex_table_from_directory(directory_path: str, dataset: str) -> str:
    """
    Generate LaTeX table from all .txt files in a directory.

    Args:
        directory_path: Path to directory containing the txt files
        dataset: Name of the dataset
    """
    # Read all txt files from directory
    input_files = {}
    if '/json' in directory_path:
        directory_path = directory_path.replace('/json', '')
        
    os.makedirs(directory_path, exist_ok=True)
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            method_name = get_method_name_from_file(filename)
            file_path = os.path.join(directory_path, filename)
            input_files[method_name] = file_path

    if not input_files:
        raise ValueError(f"No .txt files found in {directory_path}")

    # Read and parse input files
    results_dict = {}
    for method, file_path in input_files.items():
        with open(file_path, "r") as f:
            lines = f.readlines()
            results = {}
            for line in lines:
                # Rimuovi il \\ alla fine della riga
                line = line.replace("\\\\", "").strip()
                values = [v.strip() for v in line.split("&")]
                sample_percentage = int(values[0])
                metrics = []
                for val in values[1:]:
                    if "-1.000" in val:
                        metrics.append((-1, -1))
                    else:
                        mean_std = val.split("$\\pm$")
                        mean = float(mean_std[0].strip())
                        std = float(mean_std[1].strip())
                        metrics.append((mean, std))
                results[sample_percentage] = metrics
            results_dict[method] = results

    # Generate LaTeX table
    latex = "\\begin{table}[h]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{l|"

    num_methods = len(results_dict)
    latex += "ccc|" * num_methods
    latex = latex.rstrip("|") + "}\n\\hline\n"

    # Header
    latex += "\% Samples"
    for method in sorted(results_dict.keys()):
        method_name = method.replace("_", " ")
        latex += f" & \\multicolumn{{3}}{{c|}}{{{method_name}}}"
    latex = latex.rstrip("|")
    latex += " \\\\\n"

    latex += " & "
    metrics = ["Precision", "F1"] * num_methods #["Precision", "Recall", "F1"] * num_methods
    latex += " & ".join(metrics)
    latex += " \\\\\n\\hline\n"

    # Data rows
    samples_perc = sorted(list(next(iter(results_dict.values())).keys()))
    for sample in samples_perc:
        row = [f"{int(sample)}"]

        # Find best values for each metric at this sample percentage
        best_values = [find_best_values(results_dict, sample, i) for i in range(2)]

        for method in sorted(results_dict.keys()):
            try:
                result = results_dict[method][sample]
                for metric_idx, (mean, std) in enumerate(result):
                    if mean == -1 and std == -1:
                        row.append("-")
                    else:
                        value = f"{mean:.3f} $\\pm$ {std:.3f}"

                        # Check if this is the best or second best value
                        best, second_best = best_values[metric_idx]
                        if mean == best:
                            value = f"\\textbf{{{value}}}"
                        elif second_best is not None and mean == second_best:
                            value = f"\\underline{{{value}}}"

                    row.append(value)
            except KeyError:
                row.append("-")
            

        latex += " & ".join(row) + " \\\\\n"

    latex += "\\hline\n\\end{tabular}}\n"
    latex += f"\\caption{{Results for {dataset} dataset.}}\n"
    latex += f"\\label{{tab:{dataset}}}\n"
    latex += "\\end{table}"

    dir = os.path.join(Path(directory_path).parent, "latex_tables")
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, f"{dataset}.tex"), "w") as f:
        f.write(latex)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["unbiased","acc_pred_only", "summarization", "final"],
        required=True,
        default="unbiased",
        help="Task to perform",
    )

    parser.add_argument("--method", type=str, nargs="+", help="Method(s) to analyze")

    parser.add_argument(
        "--dataset", type=str, nargs="+", default=["imdb"], help="Dataset name(s)"
    )

    parser.add_argument(
        "--class_distribution",
        type=str,
        nargs="+",
        help="Class distribution(s) (comma-separated)",
    )

    parser.add_argument(
        "--num_samples", type=int, nargs="+", help="Number(s) of samples"
    )

    parser.add_argument(
        "--base_folder", type=str, default="synthetic", help="Base folder for results"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    if args.task == "unbiased":
        process_all_folders(
            os.path.join(args.base_folder, "json"), metric_type="unbiased", for_final=False
        )
        process_results_structure(os.path.join(args.base_folder, "overleaf"))
    elif args.task == "final":
        process_all_folders(
            os.path.join(args.base_folder, "json"), metric_type="unbiased", for_final=True
        )
        process_results_structure(os.path.join(args.base_folder, "overleaf"))
    elif args.task == "summarization":
        process_all_folders(
            os.path.join(args.base_folder, "json"), metric_type="summarization"
        )
        process_results_structure(os.path.join(args.base_folder, "overleaf"))
    elif args.task == 'acc_pred_only':
        print(compute_accuracy_predictor_only())
    else:
        raise ValueError(f"Plotting method {args.task} not recognized.")
