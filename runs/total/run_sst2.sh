#!/bin/bash

#Coverage & Others
CUDA_VISIBLE_DEVICES=0 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 300 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 300 --methods Random &
CUDA_VISIBLE_DEVICES=3 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 300 --methods Stratified &
CUDA_VISIBLE_DEVICES=2 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 300 --methods Agreement

#Uncertainty and Surrogate
CUDA_VISIBLE_DEVICES=0 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 300 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 300 --methods Uncertainty --strategy gaussian_prior &
CUDA_VISIBLE_DEVICES=2 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 300 --methods Surrogate --strategy SVM &
CUDA_VISIBLE_DEVICES=3 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 300 --methods Surrogate --strategy RF 


#Coverage & Others
CUDA_VISIBLE_DEVICES=0 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 600 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 600 --methods Random &
CUDA_VISIBLE_DEVICES=3 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 600 --methods Stratified &
CUDA_VISIBLE_DEVICES=2 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 600 --methods Agreement

#Uncertainty and Surrogate
CUDA_VISIBLE_DEVICES=0 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 600 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 600 --methods Uncertainty --strategy gaussian_prior &
CUDA_VISIBLE_DEVICES=2 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 600 --methods Surrogate --strategy SVM &
CUDA_VISIBLE_DEVICES=3 python3 main.py --num_seeds 10 --dataset_name sst2 --budget 600 --methods Surrogate --strategy RF 

