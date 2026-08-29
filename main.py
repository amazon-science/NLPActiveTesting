from tqdm import tqdm
import argparse
import time
from copy import deepcopy

from utils import *
from data import *
from active_testing.active import NLPActiveTesting

def main(
    methods: list,
    budget: int,
    dataset_name: str = "imdb",
    seed: int = 42,
    test_sizes: List[float] = None,
    max_lenght: int = 512,
    model_name: str = "bert-base-multilingual-cased",
    predictor:str = 'qwen',
    **kwargs
) -> None:
    """
    Run the active testing framework with specified strategies and configuration.

    Args:
        methods: List of testing strategy names (e.g., ['Random', 'Coverage', 'Uncertainty']).
        budget: Maximum number of samples to use from the dataset.
        dataset_name: Name of the dataset to load. Defaults to "imdb".
        seed: Random seed for reproducibility. Defaults to 42.
        test_sizes: List of test set size fractions to evaluate (e.g., [0.1, 0.2, 0.5]).
        max_lenght: Maximum sequence length for text preprocessing. Defaults to 512.
        model_name: Transformer model for embeddings. Defaults to "bert-base-multilingual-cased".
        predictor: Name of predictor model for predictions. Defaults to 'qwen'.
        **kwargs: Additional parameters (n_clusters, clustering, method, etc.).

    Returns:
        Dict: Nested results {method_name: {num_samples: metrics_dict}}.
    """
    # Initialization
    main_dict = None
    seed_everything(seed=seed)
    if 1.0 not in test_sizes:
        test_sizes.append(1.0)

    # Load and preprocess dataset
    dataset, x_name, y_name, pipeline_name, classes = load_data(dataset_name=dataset_name)
    pipeline_name = [pipeline_name, dataset_name]

    if pipeline_name[0] == "text-classification" and 'relevant' in classes.values():
        texts = [preprocess_text(dataset[x_name][i], length=max_lenght) for i in range(len(dataset))]
    else:
        texts = [preprocess_text(sample[x_name], length=max_lenght) for sample in dataset]
    labels = dataset[y_name]
    
    full_labels = deepcopy(labels)  
    
    # Initialize testing strategies
    methods = [
        NLPActiveTesting(
            texts=texts, labels=labels, budget=budget,
            pipeline_name=pipeline_name, classes=classes, 
            model_name=model_name,
            predictor_name=predictor
        ).create_instance(x, **kwargs)
        for x in methods
    ]

    # Run tests for each method and size
    results = {}
    for method in methods:
        results[method.name] = {}
        for size in list(reversed(test_sizes)):
            initial_time = time.time()
            num_samples = int(len(method.texts)*size)
            predictions, indices, scores = method.select_next_test_case(num_samples=num_samples)
            labels = [method.labels[i] for i in indices]
            # Evaluate metrics
            if size == 1.0:
                current_result = evaluate_metrics(full_labels, method.total_predictions,
                                                  None)
                main_dict = current_result
            else:
                # Store results
                minority_results = {
                    'time': time.time()-initial_time,
                }
                
                budgeted_result = evaluate_metrics(labels, predictions, None)
                unbiased_metrics = method.estimate_unbiased_metrics(indices, scores, main_dict)
                
                minority_results.update(compute_class_metrics(labels=method.labels,
                                                              indices=indices,
                                                              ))
                current_result = evaluate_metrics(labels, predictions, main_dict)
                current_result.update(minority_results)
                current_result.update(unbiased_metrics)
            results[method.name][num_samples] = current_result
    return results

def run_multiple_seeds(
    methods,
    budget,
    dataset_name: str,
    seeds: List[int],
    test_sizes: List[float],
    model_name: str = "bert-base-multilingual-cased",
    predictor:str = 'qwen',
    **kwargs
) -> Dict:
    """
    Run experiments multiple times with different random seeds.

    Args:
        methods: List of strategy names to evaluate.
        budget: Maximum samples to use from dataset.
        dataset_name: Dataset to use for experiments.
        seeds: List of random seeds for reproducibility.
        test_sizes: Test set size fractions to evaluate.
        model_name: Transformer model for embeddings.
        predictor: Predictor model name.
        **kwargs: Additional parameters passed to main().

    Returns:
        List[Dict]: List of result dictionaries, one per seed.
    """
    all_results = []
    for seed in tqdm(seeds):
        results = main(
            methods=methods,
            budget=budget,
            dataset_name=dataset_name,
            seed=seed,
            test_sizes=test_sizes,
            predictor = predictor,
            model_name=model_name,
            **kwargs
        )
        all_results.append(results)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        type=str,
        default="-1",
        help="Testing strategy to use",
    )
    parser.add_argument(
        "--clustering",
        type=str,
        default="-1",
        choices=["shift", "kmeans", "dbscan"],
        help="Clustering algorithm to use",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=-1,
        help="Number of clusters",
    )

    parser.add_argument(
        "--methods",
        type=str,
        default= ["Coverage", "Random"],
        nargs="*",
        help="Active testing methods",
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Maximum percentage of the full test set which can be used for testing",
    )

    parser.add_argument(
        "--num_seeds", type=int, default=5, help="Number of different seeds to run"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default='qwen',
        help="Name of the model used to have the embeddings.",
    )

    parser.add_argument(
        "--dataset_name", type=str, default="imdb", help="Name of the dataset."
    )
    
    parser.add_argument(
        "--predictor", type=str, default="claude", help="Name of the dataset."
    )

    args = parser.parse_args()
    additional = {
        "n_clusters": args.n_clusters,
        "clustering": args.clustering,
        "method": args.strategy,
        "base_folder" : f"{args.predictor}/results_{args.model_name}",
        "num_samples" : args.budget,
    }
    
    test_sizes = [0.02, 0.05] + [x / 10 for x in range(1, 11)]

    seeds = list(range(args.num_seeds))
    if len(seeds) > 1:
        all_results = run_multiple_seeds(
            methods=args.methods,
            budget=args.budget,
            predictor = args.predictor,
            dataset_name=args.dataset_name,
            seeds=seeds,
            test_sizes=test_sizes,
            model_name=model_name_map(args.model_name, inverse=True),
            **additional
        )
        save_exp(all_results=all_results,
                 dataset_name=args.dataset_name,
                 **additional)
    else:
        raise ValueError("You have to insert more than 1 seed value!")

