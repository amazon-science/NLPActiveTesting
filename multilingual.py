import numpy as np
from tqdm import tqdm
import argparse
import time

from utils import *
from data import *
from active_testing.active import NLPActiveTesting
    
def main(
    methods: list,
    seed: int = 42,
    test_sizes: List[float] = None,
    max_lenght: int = 512,
    predictor:str = 'nova_pro',
    budget: int = 500,
    model_name:str = "bert-base-multilingual-cased",
    other_language:str='italian',
    **kwargs
) -> None:
    """
    Run the active testing framework on multilingual datasets.

    Evaluates sampling strategies on English-only, other-language-only, and mixed
    datasets. Supports language prior weighting for controlled cross-lingual sampling.

    Args:
        methods: List of testing strategy names to evaluate.
        seed: Random seed for reproducibility. Defaults to 42.
        test_sizes: List of test set size fractions to evaluate.
        max_lenght: Maximum sequence length for preprocessing. Defaults to 512.
        predictor: Predictor model name for loading predictions. Defaults to 'nova_pro'.
        budget: Maximum samples per language to use. Defaults to 500.
        model_name: Transformer model for embeddings. Defaults to "bert-base-multilingual-cased".
        other_language: Second language to compare with English. Defaults to 'italian'.
        **kwargs: Additional parameters including 'language_prior' for mixed sampling.

    Returns:
        Tuple[Dict, Tuple[int, int]]:
            - results: Nested dict {language: {method: {num_samples: metrics}}}.
            - (n_english, n_other): Sample counts per language in mixed selection.
    """
    # Initialize random seed and test sizes
    seed_everything(seed=seed)
    if 1.0 not in test_sizes:
        test_sizes.append(1.0)

    dataset_other_language, x_name, y_name, pipeline_name, classes = load_data(dataset_name="multilingual", language=other_language)
    pipeline_name_other_language = [pipeline_name, f"multilingual_{other_language}"]
    
    dataset_en, x_name, y_name, pipeline_name, classes = load_data(dataset_name="multilingual", language="english")
    
    pipeline_name_en = [pipeline_name, "multilingual_english"]

    texts_en = [preprocess_text(sample[x_name], length=max_lenght) for sample in dataset_en]
    texts_other_language = [preprocess_text(sample[x_name], length=max_lenght) for sample in dataset_other_language]
    
    texts_mixed = texts_en + texts_other_language
    
    labels_en = dataset_en[y_name]
    labels_other_language = dataset_other_language[y_name]
    
    labels_mixed = labels_en + labels_other_language

    
    preds_en = np.load(f'autoeval/{predictor}/all_predictions_multilingual_english.npy')
    preds_other_language = np.load(f'autoeval/{predictor}/all_predictions_multilingual_{other_language}.npy')
    
    try:
        preds_mixed = np.load(f'autoeval/{predictor}/all_predictions_multilingual_mixed.npy')
    except FileNotFoundError:
        preds_mixed =  preds_en + preds_other_language
        preds_mixed = np.concatenate([preds_en, preds_other_language])
        np.save(f'autoeval/{predictor}/all_predictions_multilingual_mixed.npy', preds_mixed)
    
    methods_only_en = [
        NLPActiveTesting(
            texts=texts_en, labels=labels_en, budget=budget,
            pipeline_name=pipeline_name_en, classes=classes,
            model_name = model_name,
            predictor_name=predictor
        ).create_instance(x, **kwargs)
        for x in methods
    ]
    methods_only_other_language = [
        NLPActiveTesting(
            texts=texts_other_language, labels=labels_other_language, budget=budget,
            pipeline_name=pipeline_name_other_language, classes=classes,
            model_name = model_name,
            predictor_name=predictor
        ).create_instance(x, **kwargs)
        for x in methods
    ]
    
    methods_mixed = [
        NLPActiveTesting(
            texts=texts_mixed, labels=labels_mixed, budget=budget,
            pipeline_name=[pipeline_name, "multilingual_mixed"], classes=classes,
            model_name = model_name,
            predictor_name=predictor
        ).create_instance(x, **kwargs)
        for x in methods
    ]
        
    # Evaluate methods
    results = {'en': {},
               other_language : {},
               'mixed' : {},
               }
    assert len(preds_en) == len(labels_en), f"Predictions and labels must have the same lenght! \
        Predictions lenght: {len(preds_en)}, Label lenght: {len(labels_en)}"
    main_dict_en = evaluate_metrics(labels_en, preds_en, None)
    
    assert len(preds_other_language) == len(labels_other_language), f"Predictions and labels must have the same lenght! \
        Predictions lenght: {len(preds_other_language)}, Label lenght: {len(labels_other_language)}"
    main_dict_other_language = evaluate_metrics(labels_other_language, preds_other_language, None)
    
    assert len(preds_mixed) == len(labels_mixed), f"Predictions and labels must have the same lenght! \
        Predictions lenght: {len(preds_mixed)}, Label lenght: {len(labels_mixed)}"
    main_dict_mixed= evaluate_metrics(labels_mixed, preds_mixed, None)
    
    # Test each method with different test sizes
    for method_en, method_other_language, method_mixed in zip(methods_only_en, methods_only_other_language,
                                                            methods_mixed):
        results['en'][method_en.name] = {}
        results[other_language][method_other_language.name] = {}
        results['mixed'][method_mixed.name] = {}
        
        for size in list(reversed(test_sizes)):
            initial_time = time.time()
            num_samples = int(len(texts_en)*size)
            
            
            current_predictions_en, indices_en, scores_en = method_en.select_next_test_case(num_samples=num_samples)
            
            
            current_predictions_other_language, indices_other_language, scores_other_language = method_other_language.select_next_test_case(
                num_samples=num_samples,
            )
            
            current_predictions_mixed, indices_mixed, scores_mixed = method_mixed.select_next_test_case(
                num_samples=num_samples, lang_prior=(kwargs['language_prior'],1-kwargs['language_prior'])
            )
                        
            if size != 1.0:
                minority_results = {
                    'time': time.time()-initial_time,
                    'en' : {},
                    other_language : {},
                    'mixed' : {},
                }
                
                selected_labels_en = [labels_en[i] for i in indices_en]
                selected_labels_other_language = [labels_other_language[i] for i in indices_other_language]
                selected_labels_mixed = [labels_mixed[i] for i in indices_mixed]
            
            
            
                minority_results["en"].update(compute_class_metrics(labels=selected_labels_en,
                                                              indices=indices_en))
                minority_results[other_language].update(compute_class_metrics(labels=selected_labels_other_language,
                                                              indices=indices_other_language))
                minority_results["mixed"].update(compute_class_metrics(labels=selected_labels_mixed,
                                                              indices=indices_mixed))
                
                current_result = {'en': {},
                                  other_language : {},
                                  'mixed' : {},
                                  }
                
                budgeted_result = {'en': {},
                                  other_language : {},
                                  'mixed' : {},
                                  }
                unbiased_metrics = {'en': {},
                                  other_language : {},
                                  'mixed' : {},
                                  }
                
                
                current_result["en"] = evaluate_metrics(selected_labels_en,
                                                             current_predictions_en,
                                                             main_dict_en)
                current_result[other_language] = evaluate_metrics(selected_labels_other_language,
                                                             current_predictions_other_language,
                                                             main_dict_other_language)
                current_result["mixed"] = evaluate_metrics(selected_labels_mixed,
                                                             current_predictions_mixed,
                                                             main_dict_mixed)
                
                budgeted_result["en"] = evaluate_metrics(selected_labels_en,
                                                              current_predictions_en, None)
                budgeted_result[other_language] = evaluate_metrics(selected_labels_other_language,
                                                              current_predictions_other_language, None)
                budgeted_result["mixed"] = evaluate_metrics(selected_labels_mixed,
                                                              current_predictions_mixed, None)
                
                
                unbiased_metrics["en"] = method_en.estimate_unbiased_metrics(indices_en,
                                                                                       scores_en,
                                                                                       budgeted_result["en"])
                unbiased_metrics[other_language] = method_other_language.estimate_unbiased_metrics(indices_other_language,
                                                                                       scores_other_language,
                                                                                       budgeted_result[other_language])
                unbiased_metrics["mixed"] = method_mixed.estimate_unbiased_metrics(indices_mixed,
                                                                                       scores_mixed,
                                                                                       budgeted_result["mixed"])
                
                
                
                current_result["en"].update(minority_results["en"])
                current_result["en"].update(unbiased_metrics["en"])
                current_result[other_language].update(minority_results[other_language])
                current_result[other_language].update(unbiased_metrics[other_language])
                current_result["mixed"].update(minority_results["mixed"])
                current_result["mixed"].update(unbiased_metrics["mixed"])

                results["en"][method_en.name][num_samples] = current_result["en"]
                results[other_language][method_other_language.name][num_samples] = current_result[other_language]
                results["mixed"][method_mixed.name][num_samples] = current_result["mixed"]
            else: 
                results["en"][method_en.name][num_samples] = main_dict_en
                results[other_language][method_other_language.name][num_samples] = main_dict_other_language
                results["mixed"][method_mixed.name][num_samples] = main_dict_mixed
    
    n_samples_english, n_samples_other = compute_number_english_samples(indices_mixed, int(len(methods_mixed[0].texts)/2))
    
    return results, (n_samples_english, n_samples_other)

def run_multiple_seeds(
    methods,
    dataset_name: str,
    seeds: List[int],
    budget: int,
    test_sizes: List[float],
    predictor:str='nova_pro',
    model_name: str = "bert-base-multilingual-cased",
    **kwargs
) -> Dict:
    """
    Run multilingual experiments with multiple random seeds.

    Args:
        methods: List of strategy names to evaluate.
        dataset_name: Dataset name (should be "multilingual").
        seeds: List of random seeds for reproducibility.
        budget: Maximum samples per language.
        test_sizes: Test set size fractions to evaluate.
        predictor: Predictor model name.
        model_name: Transformer model for embeddings.
        **kwargs: Additional parameters including 'language_prior', 'other_language'.

    Returns:
        Tuple[List[Dict], Tuple[float, float]]:
            - all_results: List of result dictionaries, one per seed.
            - (mean_english, mean_other): Average sample counts per language across seeds.
    """
    all_results = []
    eng_samples, other_language_samples = np.array([]), np.array([])
    for seed in tqdm(seeds):
        results, samples_language_distribution = main(
            methods=methods,
            budget = budget,
            dataset_name=dataset_name,
            predictor=predictor,
            seed=seed,
            test_sizes=test_sizes,
            model_name=model_name,
            **kwargs
        )
        eng_samples = np.append(eng_samples,samples_language_distribution[0])
        other_language_samples = np.append(other_language_samples, samples_language_distribution[1])
        all_results.append(results)
    return all_results, (np.mean(eng_samples), np.mean(other_language_samples))

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()

    # Define command line arguments
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
        "--model_name",
        type=str,
        default='qwen',
        help="Name of the model used to have the embeddings.",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default='qwen',
        help="Name of the predictor.",
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
        default=["Coverage", "Random"],  # Alternative options: ["Uncertainty", "Random"], ["Distance"]
        nargs="*",
        help="Active testing methods",
    )

    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of different seeds to run"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="multilingual",
        help="Name of the dataset."
    )
    
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Maximum percentage of the full test set which can be used for testing",
    )
    
    parser.add_argument(
        "--prior",
        type=float,
        default=.6,
        help="Prior on english language",
    )
    

    # Parse arguments
    args = parser.parse_args()
    
    # Additional parameters for methods
    additional = {
        "n_clusters": args.n_clusters,
        "clustering": args.clustering,
        "method": args.strategy,
        "num_samples" : args.budget,
        "language_prior": args.prior,
        #"n_samples": compute_number_english_samples()
    }
    
    # Define test sizes (2%, 5%, and 10% through 100%)
    test_sizes = [0.02, 0.05] + [x / 10 for x in range(1, 11)]

    # Generate list of seeds
    seeds = list(range(args.num_seeds))
    
    # Run experiments
    if len(seeds) > 1:
        # Run with multiple seeds
        all_results, stats_languages = run_multiple_seeds(
            methods=args.methods,
            budget=args.budget,
            dataset_name=args.dataset_name,
            seeds=seeds,
            test_sizes=test_sizes,
            predictor=args.predictor,
            model_name=model_name_map(args.model_name, inverse=True),
            **additional
        )
        
        # Update additional parameters for saving
        additional["base_folder"] = f"results_{args.model_name}"
        additional['stats_languages'] = stats_languages

        # Save experimental results
        save_exp(
            all_results=all_results,
            dataset_name=args.dataset_name,
            synthetic=True,
            **additional
        )
    else:
        # Require multiple seeds for robust results
        raise ValueError("This code only runs with multiple seed values per each experiment!")
