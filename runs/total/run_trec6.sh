#!/bin/bash

#Coverage & Others
CUDA_VISIBLE_DEVICES=0 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 200 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 200 --methods Random &
CUDA_VISIBLE_DEVICES=3 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 200 --methods Stratified &
CUDA_VISIBLE_DEVICES=2 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 200 --methods Agreement

#Uncertainty and Surrogate
CUDA_VISIBLE_DEVICES=0 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 200 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 200 --methods Uncertainty --strategy gaussian_prior &
CUDA_VISIBLE_DEVICES=2 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 200 --methods Surrogate --strategy SVM &
CUDA_VISIBLE_DEVICES=3 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 200 --methods Surrogate --strategy RF 


#Coverage & Others
CUDA_VISIBLE_DEVICES=0 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 400 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 400 --methods Random &
CUDA_VISIBLE_DEVICES=3 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 400 --methods Stratified &
CUDA_VISIBLE_DEVICES=2 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 400 --methods Agreement

#Uncertainty and Surrogate
CUDA_VISIBLE_DEVICES=0 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 400 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 400 --methods Uncertainty --strategy gaussian_prior &
CUDA_VISIBLE_DEVICES=2 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 400 --methods Surrogate --strategy SVM &
CUDA_VISIBLE_DEVICES=3 python3 main.py --num_seeds 10 --dataset_name trec6 --budget 400 --methods Surrogate --strategy RF 

