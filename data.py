import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import Counter

def load_data(dataset_name:str='imdb', language:str="spanish"):
    """
    Load a dataset from HuggingFace Hub for active testing experiments.

    Args:
        dataset_name (str, optional): Dataset to load. Supported:
            - Text Classification: 'imdb', 'dbpedia', 'pubmed', 'emotions', 'rotten',
              'agnews', 'banking77', 'fnc1', 'mnli', 'qnli', 'sst2', 'trec6',
              'multilingual'
            - Sequence Labeling: 'uniner', 'wikineural', 'ud-ewt', 'ud-atis'
            Defaults to 'imdb'.
        language (str, optional): Language for multilingual datasets. Defaults to "italian".

    Returns:
        For classification/QA: Tuple[Dataset, text_col, label_col, task_type, classes]
        For sequence labeling: Tuple[List[str], List[List[int]], task_type]
    """
    if dataset_name == 'imdb':
        dataset = load_dataset("imdb", split="test")
        return dataset, "text", "label", "text-classification", get_labels(dataset=dataset)
    elif dataset_name == 'dbpedia':
        dataset = load_dataset("pietrolesci/dbpedia_14_indexed", split="test")
        return dataset, "content", "labels", "text-classification", get_labels(dataset=dataset, label_alias='labels')
    elif dataset_name == "pubmed":
        dataset = load_dataset("pietrolesci/pubmed-20k-rct", split="test")
        return dataset, "text", "labels", "text-classification", get_labels(dataset=dataset, label_alias='labels')
    elif dataset_name == "emotions":
        dataset = load_dataset("dair-ai/emotion", split="test")
        return dataset, "text", "label", "text-classification", get_labels(dataset=dataset)
    elif dataset_name == "rotten":
        dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
        return dataset, "text", "label", "text-classification", get_labels(dataset=dataset)
    elif dataset_name == "agnews":
        dataset = load_dataset("fancyzhx/ag_news", split="test")
        return dataset, "text", "label", "text-classification", get_labels(dataset=dataset)
    elif dataset_name == "banking77":
        dataset = load_dataset("PolyAI/banking77", split="test", trust_remote_code=True)
        return dataset, "text", "label", "text-classification", get_labels(dataset=dataset)
    elif dataset_name == "fnc1":
        dataset = load_dataset("nid989/FNC-1", split="test")
        return dataset, "articleBody", "Stance", "text-classification", get_stances()
    elif dataset_name == "mnli":
        dataset = load_dataset("nyu-mll/glue", "mnli",  split="validation_matched")
        return dataset, "sentence", "label", "text-classification", get_labels(dataset=dataset)
    elif dataset_name == "qnli":
        dataset = load_dataset("nyu-mll/glue", "qnli", split="validation")
        return dataset, "sentence", "label", "text-classification", get_labels(dataset=dataset)
    elif dataset_name == "sst2":
        dataset = load_dataset("stanfordnlp/sst2", split="validation")
        return dataset, "sentence", "label", "text-classification", get_labels(dataset=dataset)
    elif dataset_name == "trec6":
        dataset = load_dataset("OxAISH-AL-LLM/trec6", split="test")
        return dataset, "text", "label", "text-classification", get_labels(dataset=dataset)
    elif dataset_name == "multilingual":
        dataset = load_dataset("tyqiangz/multilingual-sentiments", language, split="test")
        return dataset, "text", "label", "text-classification", get_labels(dataset=dataset)
    elif dataset_name == "uniner":
        dataset = load_dataset("universalner/universal_ner", "en_ewt", split="test")
        return [' '.join(item["tokens"]) for item in dataset], dataset["ner_tags"], "ner"
    elif dataset_name == "ud-ewt":
        dataset = load_dataset("commul/universal_dependencies", "en_ewt", split="test", trust_remote_code=True)
        return [' '.join(item["tokens"]) for item in dataset], dataset["upos"], "pos"
    elif dataset_name == "wikineural":
        dataset = load_dataset("Babelscape/wikineural", split="test_en")
        return [' '.join(item["tokens"]) for item in dataset], dataset["ner_tags"], "ner"
    elif dataset_name == "ud-atis":
        dataset = load_dataset("commul/universal_dependencies", "en_atis", split="test", trust_remote_code=True)
        return [' '.join(item["tokens"]) for item in dataset], dataset["upos"], "pos"
    elif dataset_name == "cnn":
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
        return dataset, "article", "highlights", "summarization", None
    elif dataset_name == "xlsum":
        dataset = load_dataset("csebuetnlp/xlsum", language, split="test")
        return dataset, "text", "summary", "summarization", None
    else:
        raise  ValueError(f"Dataset {dataset_name} not recognized!")


def get_class_distribution(dataset_name: str = 'imdb', language: str = "spanish",
                           debug:bool=False):
    """
    Calculate class distribution for a dataset.
    
    Args:
        dataset_name: name of the dataset
        language: language (for multilingual datasets)
    
    Returns:
        dict: dictionary with class distribution (count and percentage)
    """
    try:
        dataset, _, label_col, _, _ = load_data(dataset_name, language)
        all_labels = dataset[label_col]
    except ValueError:
        _, labels, _ = load_data(dataset_name=dataset_name)
        all_labels = [label for seq in labels for label in seq]
    # Count occurrences
    label_counts = Counter(all_labels)
    total = len(all_labels)
    
    # Create distribution with counts and percentages
    distribution = {}
    for label_id, count in sorted(label_counts.items()):
        distribution[label_id] = {
            'count': count,
            'percentage': (count / total) * 100
        }
    
    if debug:
        # Print results
        print(f"Class distribution for '{dataset_name}':")
        print(f"{'Class':<10} {'Count':<12} {'Percentage':<12}")
        print("-" * 40)
        for label_id, stats in distribution.items():
            print(f"{label_id:<10} {stats['count']:<12} {stats['percentage']:.2f}%")
        print(f"\nTotal: {total} samples")
    
    return distribution

def get_labels(dataset, label_alias='label') -> dict:
    output = {}
    for index, class_value in enumerate(dataset.features[label_alias].names):
        output[index] = class_value
    return output

def get_stances() -> dict:
    return {
        1 : "Agrees",
        2 : "Disagrees",
        3 : "Discusses",
        4 : "Unrelated",
    }
    
    
def get_ner_labels(task_name:str) -> dict:
    if task_name == "ner":
        return {
                'O': 0, 'B-PER': 1, 'I-PER': 2,
                'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5,
                'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8
            }
    else:
        return{
        'ADJ':   0,   # adjective
        'ADP':   1,   # adposition (prepositions, postpositions)
        'ADV':   2,   # adverb
        'AUX':   3,   # auxiliary verb
        'CCONJ': 4,   # coordinating conjunction
        'DET':   5,   # determiner
        'INTJ':  6,   # interjection
        'NOUN':  7,   # noun
        'NUM':   8,   # numeral
        'PART':  9,   # particle
        'PRON':  10,  # pronoun
        'PROPN': 11,  # proper noun
        'PUNCT': 12,  # punctuation
        'SCONJ': 13,  # subordinating conjunction
        'SYM':   14,  # symbol
        'VERB':  15,  # verb
        'X':     16,  # other
    }

if __name__ == "__main__":

    print(get_class_distribution('dbpedia', debug=True))
   
