# Select, Label, Evaluate: Active Testing in NLP
Additional plots of the paper can be found at [this link](https://drive.google.com/drive/folders/1iLsUNRwcFyHzakvfZC0BohcJKt6mrR7D?usp=sharing)!

Human annotation cost and time remain significant bottlenecks in Natural Language Processing (NLP), with test data annotation being particularly expensive due to the stringent requirement for low-error and high-quality labels necessary for reliable model evaluation. Traditional approaches require annotating entire test sets, leading to substantial resource requirements. Active Testing is a framework that selects the most informative test samples for annotation. Given a labeling budget, it aims to choose the subset that best estimates model performance while minimizing cost and human effort. In this work, we formalize Active Testing in NLP and we conduct an extensive benchmarking of existing approaches across 18 datasets and 4 embedding strategies spanning 4 different NLP tasks. The experiments show annotation reductions of up to 95%, with performance estimation accuracy difference from the full test set within 1%. Our analysis reveals variations in method effectiveness across different data characteristics and task types, with no single approach emerging as universally superior. Lastly, to address the limitation of requiring a predefined annotation budget in existing sample selection strategies, we introduce an adaptive stopping criterion that automatically determines the optimal number of samples.

If you want to launch an experiment:

Install the required libraries:

```
pip install -r requirements.txt
```

You can simply use the bash files with the pre-defined experiments. 

For experiments on real data:

```
bash runs/total/run_{dataset_name}.sh
```

- If you want to run a specific experiment:
    ```
    python3 main.py --predictor PREDICTOR_NAME --dataset_name DATASET_NAME --model_name EMBEDDER_NAME --budget B --methods METHOD_NAME --strategy STRATEGY
    ```

    Example:
    
    ```
    python3 main.py --num_seeds 10 --dataset_name agnews --budget 1000 --methods Agreement
    ```

- If you want to run a multilingual experiment:

    ```
    python3 multilingual.py --strategy STRATEGY --model_name EMBEDDER_NAME --methods METHOD_NAME --budget B --prior PRIOR_VALUE
    ```

    Example:
    
    ```
    python3 multilingual.py --predictor nova_pro --model_name bert --prior 0.6 --num_seeds 10 --dataset_name multilingual --budget 1000 --methods Agreement
    ```

- If you want to run an experiment on NER/POS tagging:
    ```
    python3 pos_ner.py --strategy STRATEGY --model_name EMBEDDER_NAME --methods METHOD_NAME --budget B --predictor PREDICTOR_NAME --dataset_name DATASET_NAME
    ```

    Example:
    
    ```
    python3 pos_ner.py --model_name distilbert --predictor nuner --num_seeds 10 --dataset_name uniner --budget 1000 --methods Agreement
    ```

- If you want to run an experiment on summatization:
    ```
    python3 summarization.py --strategy STRATEGY --model_name EMBEDDER_NAME --methods METHOD_NAME --budget B --predictor PREDICTOR_NAME --dataset_name DATASET_NAME
    ```

    Example:
    
    ```
    python3 summarization.py --predictor claude --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Agreement
    ```

    Due to space limits on Github, in order to download the LLM annotations for summarization, you have to go to the `autoeval` folder [here](https://drive.google.com/drive/folders/1iLsUNRwcFyHzakvfZC0BohcJKt6mrR7D?usp=sharing).

If you want to see some plots:

```
python3 utils.py --task unbiased --base_folder PREDICTOR/results_{EMBEDDER_NAME}
```

With:
- B: Value of budget (int);
- PRIOR_VALUE value of the prior on english (float);
- EMBEDDER_NAME: qwen,stella,bert,distilbert;
- METHOD_NAME: Random,Coverage,Surrogate,Uncertainty;
- STRATEGY: SVM or RF for Surrogate, max_dist for Coverage, gaussian_prior or mutual_information for Uncertainty.
- PREDICTOR_NAME: claude,nova_pro
- DATASET_NAME: list of all the available datasets can be found in `data.py`;

## Citation
If you use this code in your research or project, please cite us:
```bibtex
@article{purificato2026select,
  title={Select, Label, Evaluate: Active Testing in NLP},
  author={Purificato, Antonio and Bucarelli, Maria Sofia and Bacciu, Andrea and Mantrach, Amin and Silvestri, Fabrizio},
  journal={arXiv preprint arXiv:2603.21840},
  year={2026}
}
```
For doubts or errors feel free to ping purificato@diag.uniroma1.it!

## Acknowledgments

The implementation of competitor methods draws from the [Active Testing Repository](http://github.com/jlko/active-testing) and the paper [Active Testing: Sample-Efficient Model Evaluation](https://proceedings.mlr.press/v139/kossen21a.html). We gratefully acknowledge the authors for making their code available.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

## License

This library is licensed under the CC-BY-NC-4.0 License.
