from tqdm import tqdm
import argparse
import time
from copy import deepcopy

from utils import *
from data import *
from active_testing.active import NLPActiveTesting


from transformers import pipeline
    
        
def map_model_to_real_name() -> dict:
    """
    Map short model names to HuggingFace model identifiers.
    """
    return {
        "cner" : "Babelscape/cner-base",
        "nuner" : "guishe/nuner-v1_orgs",
        "bert": "vblagoje/bert-english-uncased-finetuned-pos",
        "deberta": "jordigonzm/mdeberta-v3-base-multilingual-pos-tagger"
    }
    
def get_ner_predictions(dataset, task_name: str,
                                  model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
    """
    Generate token-level predictions using a HuggingFace pipeline.

    Args:
        dataset: List of sentences (tokens joined by spaces).
        task_name: Task type ('ner' or 'pos') for pipeline selection.
        model_name: HuggingFace model identifier.

    Returns:
        List[List[int]]: Token-level predictions aligned with original tokens.
    """
    if task_name == "ner":
        my_pipeline = pipeline("ner", model=model_name, aggregation_strategy="none")
    else:
        my_pipeline = pipeline("token-classification", model=model_name, aggregation_strategy="none")
    
    labels_map = get_ner_labels(task_name=task_name)
    output = []
    
    results = my_pipeline(dataset)
    
    for idx, item in enumerate(results):
        original_tokens = dataset[idx].split()
        predictions_aligned = []
        
        current_pos = 0
        token_positions = []
        for token in original_tokens:
            start = dataset[idx].find(token, current_pos)
            end = start + len(token)
            token_positions.append((start, end))
            current_pos = end
        
        for token_start, token_end in token_positions:
            pred_found = False
            for pred in item:
                if pred['start'] < token_end and pred['end'] > token_start:
                    entity = pred['entity']
                    if entity in labels_map:
                        predictions_aligned.append(labels_map[entity])
                    else:
                        predictions_aligned.append(0)
                    pred_found = True
                    break
            
            if not pred_found:
                predictions_aligned.append(0) 
        output.append(predictions_aligned)
    
    return output

def main(
    methods: list,
    budget: int,
    dataset_name: str = "conll-ner",
    seed: int = 42,
    test_sizes: List[float] = None,
    max_lenght: int = 512,
    model_name: str = "bert-base-multilingual-cased",
    predictor:str = 'qwen',
    **kwargs
) -> None:
    """
    Run active testing framework on sequence labeling tasks.

    Args:
        methods: List of testing strategy names.
        budget: Maximum number of sequences to use.
        dataset_name: NER/POS dataset name (e.g., 'wikineural', 'ud-ewt').
        seed: Random seed for reproducibility. Defaults to 42.
        test_sizes: List of test set size fractions.
        max_lenght: Maximum sequence length. Defaults to 512.
        model_name: Transformer model for embeddings.
        predictor: NER/POS predictor model name.
        **kwargs: Additional parameters (n_clusters, clustering, method).

    Returns:
        Dict: Nested results {method_name: {num_samples: metrics_dict}}.

    """
    # Initialization
    main_dict = None
    seed_everything(seed=seed)
    if 1.0 not in test_sizes:
        test_sizes.append(1.0)

    # Load and preprocess dataset
    dataset, labels, task_name = load_data(dataset_name=dataset_name)
    pipeline_name = [task_name, dataset_name]
    
    full_labels = deepcopy(labels)  
    total_predictions = get_ner_predictions(dataset=dataset,
                                            task_name=task_name,
                                            model_name=map_model_to_real_name()[predictor])
    
    for prediction, label in zip(total_predictions, full_labels):
        assert len(prediction) == len(label)
        
        
    all_preds = [p for seq in total_predictions for p in seq]
    all_labels = [t for seq in full_labels for t in seq]
    classes = [i for i in range(max(max(all_preds), max(all_labels)) + 1)]
        
    # Initialize testing strategies
    methods = [
        NLPActiveTesting(
            texts=dataset, labels=labels, budget=budget,
            pipeline_name=pipeline_name, classes=classes, 
            model_name=model_name,
            predictor_name=predictor,
            total_predictions=total_predictions,
            all_predictions=total_predictions
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
                                                  None, task_name)
                main_dict = current_result
            else:
                # Store results
                minority_results = {
                    'time': time.time()-initial_time,
                }
                unbiased_metrics = method.estimate_unbiased_metrics(indices, scores, main_dict)
                minority_results.update(compute_class_metrics(labels=method.labels,
                                                              indices=indices,
                                                              pipeline_name=task_name))
                current_result = evaluate_metrics(labels, predictions, main_dict, task_name)
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
    Run sequence labeling experiments with multiple random seeds.

    Args:
        methods: List of strategy names to evaluate.
        budget: Maximum sequences to use.
        dataset_name: NER/POS dataset name.
        seeds: List of random seeds.
        test_sizes: Test set size fractions.
        model_name: Transformer model for embeddings.
        predictor: NER/POS predictor model name.
        **kwargs: Additional parameters.

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
        "--num_seeds", type=int, default=10, help="Number of different seeds to run"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default='qwen',
        help="Name of the model used to have the embeddings.",
    )

    parser.add_argument(
        "--dataset_name", type=str, default="conll-pos", help="Name of the dataset."
    )
    
    parser.add_argument(
        "--predictor", type=str,
        help="Name of the predictor.",
        choices = list(map_model_to_real_name().keys())
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

