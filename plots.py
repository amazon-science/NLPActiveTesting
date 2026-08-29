

import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os

from utils import *
from data import load_data

plt.rcParams.update({
    'font.size':        22,    
    'axes.titlesize':   20,    
    'axes.labelsize':   22,    
    'xtick.labelsize':  22,    
    'ytick.labelsize':  22,    
    'legend.fontsize':  18,    
    'legend.title_fontsize': 15
})

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
        "Surrogate" : (colors[2],markers[2]),
    }


def plot_stopping_line(base_folder: str, dataset_name: str = "banking77",
                       threshold: float = 0.02, ax=None):

    colors = get_colors()
    current_budget = None
    mapping = get_mapping()
    inverse_mapping = {v: k for k, v in mapping.items()}

    json_files = []
    for item in os.listdir(os.path.join(base_folder, "json", dataset_name, "1000")):
        json_files.append(item)

    data_loaded, _, _, _, _ = load_data(dataset_name=dataset_name)
    data_length = len(data_loaded)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        standalone = True
    else:
        standalone = False

    for filename in json_files:
        method_key = filename.replace('.json', '')
        method_name = decode_filename(method_key, inverse_mapping)

        with open(os.path.join(base_folder, "json", dataset_name, "1000", filename), "r") as f:
            data = json.load(f)

        budget_values = {}

        for seed in data:
            if method_name == 'Surrogate - SVM':
                method_name = 'Surrogate'
            if method_name not in seed:
                continue
            seed_data = seed[method_name]
            for b_str, metrics in seed_data.items():
                if "unbiased_accuracy" in metrics:
                    b = int(b_str)
                    acc = metrics["unbiased_accuracy"]
                    budget_values.setdefault(b, []).append(acc)

        budgets = sorted(budget_values.keys())
        mean_acc = [np.mean(budget_values[b]) for b in budgets]
        std_acc = [np.std(budget_values[b]) for b in budgets]

        if len(budgets) > 1:
            ax.plot(budgets, mean_acc,
                    marker=colors[method_name][1],
                    color=colors[method_name][0],
                    label=method_name)
            ax.fill_between(budgets,
                            np.array(mean_acc) - np.array(std_acc),
                            np.array(mean_acc) + np.array(std_acc),
                            alpha=0.15)

            collapse_budget = None
            for b, acc in zip(budgets, mean_acc):
                if acc < threshold:
                    collapse_budget = b
                    break

            if collapse_budget is not None:
                if current_budget is not None and current_budget == collapse_budget:
                    collapse_budget += 100
                ax.axvline(x=collapse_budget, linestyle='--', alpha=0.7,
                           color=colors[method_name][0])
                current_budget = collapse_budget

    ax.set_xlim(0, 1000)
    ticks = ax.get_xticks()[1:]
    labels = [f"{int(t)} ({(t / data_length) * 100:.0f}%)" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_title(dataset_name.upper(), fontsize=16)
    ax.tick_params(axis='x')
    ax.grid(True, linestyle='--', alpha=0.4)

    if standalone:
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f"plots/stopping_{dataset_name}.pdf", format="pdf")



def extract_stats(data, method):
    budgets = sorted(map(int, data[0][method].keys()))
    stats = {b: [] for b in budgets}
    for run in data:
        for b_str, metrics in run[method].items():
            b = int(b_str)
            if "unbiased_accuracy" in metrics:
                stats[b].append(metrics["unbiased_accuracy"])
    means = [np.mean(stats[b]) for b in budgets]
    stds = [np.std(stats[b]) for b in budgets]
    return budgets, means, stds

def plot_emedders_difference(method:str, dataset:str, embedders:list, num_samples:int,):
    embedders = ['qwen', 'stella']
    
    mapping = get_mapping()
    inverse_mapping = {v: k for k, v in mapping.items()}
    base_folder = f"complete_folders/nova_pro/results_{embedders[0]}/json/{dataset}/{num_samples}"
    
    for item in os.listdir(base_folder):
        if decode_filename(item,inverse_mapping) == method:
            with open(f"{base_folder}/{item}") as f:
                data1 = json.load(f)
            with open(f"{base_folder.replace(embedders[0], embedders[1])}/{item}") as f:
                data2 = json.load(f)

    budgets1, means1, stds1 = extract_stats(data1, method)
    budgets2, means2, stds2 = extract_stats(data2, method)

    x = np.arange(len(budgets1))
    width = 0.35
    
    plt.bar(x - width/2, means1, width, yerr=stds1, capsize=5, label="Qwen", hatch='*')
    plt.bar(x + width/2, means2, width, yerr=stds2, capsize=5, label="Stella", hatch='+')
    
    plt.xlabel("Number of Samples")
    plt.ylabel("Estimation Error")
    plt.xticks(x, budgets1, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results_embedders.pdf', format='pdf', dpi=600)
    
def plot_initial():
    datasets = ['IMDB', 'Pubmed', 'AgNews', 'QNLI', 'Banking77', 'SST2']
    full_test_set = [500, 600, 150, 110, 60, 40]
    active_test_set = [8, 12, 12, 12, 8, 8]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width/2, full_test_set, width, label='Full Test Set', color='#ff7f0e', hatch='+')
    bars2 = ax.bar(x + width/2, active_test_set, width, label='Active Test Set', color='#1f77b4', hatch='*')

    ax.set_ylabel('Cost')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=16)
    ax.set_ylim(0, 620)
    ax.legend()

    ax.yaxis.grid(False)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/cost_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
def combine_pdfs_to_subplot(nrows, ncols, model_name='qwen', figsize=(9, 9), plot_name='multilingual'):
    os.makedirs('plots', exist_ok=True)
    
    if plot_name == "multilingual":
        img_paths = [
            f"multilingual/results_{model_name}/plots_0_6/multilingual/1000/multilingual_4.png",
            f"multilingual/results_{model_name}/plots_0_9/multilingual/1000/multilingual_4.png",
            f"multilingual/results_{model_name}/plots_0_6/multilingual/1000/multilingual_0.png",
            f"multilingual/results_{model_name}/plots_0_9/multilingual/1000/multilingual_0.png",
            f"multilingual/results_{model_name}/plots_0_6/multilingual/1000/multilingual_5_p.png",
            f"multilingual/results_{model_name}/plots_0_9/multilingual/1000/multilingual_5_p.png",
        ]
        
        save_path = "plots/multilingual.pdf"
    elif plot_name == 'standard_figure':
        img_paths = [
            "qwen/results_qwen/plots/cnn/1000/unbiased_rouge_1_precision_combined_plot_ignored.png",
            "claude/results_bert/plots/xlsum/1000/unbiased_rouge_1_precision_combined_plot_ignored.png",
            "complete_folders/results_bert/plots/uniner/1000/unbiased_accuracy_combined_plot_ignored.png",
            "complete_folders/results_qwen/plots/ud-ewt/1000/unbiased_accuracy_combined_plot_ignored.png",
            "complete_folders/results_bert/plots/dbpedia/1000/unbiased_accuracy_combined_plot_ignored.png",
            "complete_folders/results_stella/plots/trec6/400/unbiased_accuracy_combined_plot_ignored.png",
        ]
        
        save_path = "plots/standard_figure.pdf"
    else:
        raise ValueError(f"Plot name {plot_name} not recognized.")
        

    total_slots = nrows * ncols
    if len(img_paths) > total_slots:
        raise ValueError(f"More images ({len(img_paths)}) than available slots ({total_slots}).")

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)

    axs_flat = axs.flat if hasattr(axs, "flat") else [axs]

    for ax in axs_flat:
        ax.axis('off')  

    for i, img_path in enumerate(img_paths):
        ax = axs_flat[i]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Immagine non trovata: {img_path}")
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_aspect('equal')

        ax.axis('off')

    if plot_name == "multilingual":
        fig.legend(
        handles=[
            plt.Line2D([0], [0], color=get_colors()['en'][0], lw=2, label='Only English', marker=get_colors()['en'][1]),
            plt.Line2D([0], [0], color=get_colors()['italian'][0], lw=2, label='Only Italian', marker=get_colors()['italian'][1]),
            plt.Line2D([0], [0], color=get_colors()['mixed'][0], lw=2, label='Mixed', marker=get_colors()['mixed'][1]),
        ],
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, -0.08) 
    )
    
    elif plot_name == 'standard_figure':
        fig.legend(
        handles=[
            plt.Line2D([0], [0], color=get_colors()['Random'][0], lw=2, label='Random', marker=get_colors()['Random'][1]),
            plt.Line2D([0], [0], color=get_colors()['Agreement'][0], lw=2, label='Agreement', marker=get_colors()['Agreement'][1]),
            plt.Line2D([0], [0], color=get_colors()['Stratified'][0], lw=2, label='Stratified', marker=get_colors()['Stratified'][1]),
            plt.Line2D([0], [0], color=get_colors()['Surrogate - SVM'][0], lw=2, label='Surrogate - SVM', marker=get_colors()['Surrogate - SVM'][1]),
        ],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, -0.13) 
    )
    else:
        raise ValueError(f"Plot name {plot_name} not recognized.")
    
    fig.supxlabel("Number of Samples (% of Test Set)", fontsize=22)
    fig.supylabel("Estimation Error", fontsize=22)

    fig.savefig(save_path, dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plot_type",
        type=str,
        default="-1",
        help="Plot type",
    )
    
    parser.add_argument(
        "--base_folder",
        type=str,
        default="-1",
        help="Base folder",
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="-1",
        help="Dataset name",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="-1",
        nargs='+',
        help="Model name.",
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="-1",
        help="Method name.",
    )
    
    args = parser.parse_args()
    
    if args.plot_type == 'stopping':
        #python3 plots.py --dataset_name imdb --base_folder complete_folders/qwen/results_qwen --plot_type stopping
        assert args.dataset_name != "-1" and args.base_folder != "-1", "Select the dataset and the base folder."        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        plot_stopping_line(base_folder="complete_folders/qwen/results_qwen", dataset_name="imdb",    ax=ax1)
        plot_stopping_line(base_folder="complete_folders/claude/results_qwen", dataset_name="qnli",    ax=ax2)

        fig.supxlabel("Number of Samples (% of Test Set)", fontsize=22)
        fig.supylabel("Estimation Error", fontsize=22)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=22,
                bbox_to_anchor=(0.5, -0.12))

        plt.tight_layout()
        plt.savefig("stopping_new.pdf", format="pdf", bbox_inches='tight')
        plt.show()
    elif args.plot_type == "standard_figure":
        combine_pdfs_to_subplot(3,2, plot_name="standard_figure")
    elif args.plot_type == "multilingual":
        combine_pdfs_to_subplot(3,2, plot_name="multilingual")
    elif args.plot_type == "embedders":
        #python3 plots.py --dataset_name agnews --method Agreement --plot_type embedders --model boh
        assert args.method != "-1" and args.dataset_name!= "-1" and args.model_name!= "-1", "Select name of the model."
        plot_emedders_difference(args.method, args.dataset_name ,args.model_name,1000)
    elif args.plot_type == "initial":
        plot_initial()
    else:
        raise ValueError(f"Plot type not recognized.")