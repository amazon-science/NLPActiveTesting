from tqdm import tqdm
import argparse
import time
from typing import List, Dict
from copy import deepcopy
import numpy as np
from collections import Counter

from transformers import pipeline
import evaluate

from active_testing.active import NLPActiveTesting
from utils import *
from data import *
from estimator import * 

def map_model_to_real_name() -> dict:
    return {
        "bart": "facebook/bart-large-cnn",
        "pegasus": "google/pegasus-xsum",
        "t5": "google/flan-t5-base"
    }


def get_summarization_predictions(
    dataset: List[str],
    model_name: str,
    max_length: int = 128,
):
    summarizer = pipeline("summarization", model=model_name)

    outputs = summarizer(
        dataset,
        max_length=max_length,
        truncation=True
    )

    return [o["summary_text"] for o in outputs]


def main(
    methods: list,
    budget: int,
    dataset_name: str,
    seed: int,
    test_sizes: List[float],
    model_name: str,
    predictor: str,
    **kwargs
):

    seed_everything(seed)

    if 1.0 not in test_sizes:
        test_sizes.append(1.0)

    
    dataset, x_name, y_name, task_name, _ = load_data(dataset_name=dataset_name)
    pipeline_name = [task_name, dataset_name]
    texts = dataset[x_name]
    labels = dataset[y_name]

    methods = [
        NLPActiveTesting(
            texts=texts,
            labels=labels,
            budget=budget,
            pipeline_name=pipeline_name,
            model_name=model_name,
            classes=[],
            predictor_name=predictor
        ).create_instance(x, **kwargs)
        for x in methods
    ]

    results = {}

    for method in methods:
        results[method.name] = {}

        for size in reversed(test_sizes):
            start = time.time()
            num_samples = int(len(method.texts) * size)

            predictions_subset, indices, scores = \
                method.select_next_test_case(num_samples=num_samples)

            references_subset = [labels[i] for i in indices]

            if size == 1.0:
                current_result = compute_rouge(
                    method.total_predictions,
                    labels,
                    ["rouge1"]
                )
                main_dict = current_result
            else:
                current_result = compute_rouge(
                    predictions_subset,
                    references_subset,
                     ["rouge1"]
                )

                unbiased_rouge = compute_unbiased_rouge_n(
                    method.all_predictions,
                    labels,
                    scores,
                    indices,
                    n=1
                )

                current_result.update(unbiased_rouge)
                current_result["time"] = time.time() - start

            results[method.name][num_samples] = current_result

    return results


def run_multiple_seeds(
    methods,
    budget,
    dataset_name,
    seeds,
    test_sizes,
    model_name,
    predictor,
    **kwargs
):

    all_results = []

    for seed in tqdm(seeds):
        results = main(
            methods=methods,
            budget=budget,
            dataset_name=dataset_name,
            seed=seed,
            test_sizes=test_sizes,
            model_name=model_name,
            predictor=predictor,
            **kwargs
        )
        all_results.append(results)

    return all_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--methods", nargs="*", default=["Coverage", "Random"])
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--predictor", type=str, default="claude")
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--num_seeds", type=int, default=5)
    
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

    args = parser.parse_args()

    test_sizes = [0.02, 0.05] + [x / 10 for x in range(1, 11)]
    seeds = list(range(args.num_seeds))

    if len(seeds) <= 1:
        raise ValueError("You must use more than 1 seed.")
    
    additional = {
        "n_clusters": args.n_clusters,
        "clustering": args.clustering,
        "method": args.strategy,
        "base_folder" : f"{args.predictor}/results_{args.model_name}",
        "num_samples" : args.budget,
    }

    all_results = run_multiple_seeds(
        methods=args.methods,
        budget=args.budget,
        dataset_name=args.dataset_name,
        seeds=seeds,
        test_sizes=test_sizes,
        model_name=model_name_map(args.model_name, inverse=True),
        predictor=args.predictor,
        **additional
    )

    save_exp(
        all_results=all_results,
        dataset_name=args.dataset_name,
        **additional
    )