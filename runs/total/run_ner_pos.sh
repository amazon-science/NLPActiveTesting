#!/bin/bash

#Coverage & Others
CUDA_VISIBLE_DEVICES=0 python3 pos_ner.py --model_name bert --predictor nuner --num_seeds 10 --dataset_name uniner --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 pos_ner.py --model_name bert --predictor nuner --num_seeds 10 --dataset_name uniner --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=3 python3 pos_ner.py --model_name bert --predictor nuner --num_seeds 10 --dataset_name uniner --budget 1000 --methods Stratified &
CUDA_VISIBLE_DEVICES=2 python3 pos_ner.py --model_name bert --predictor nuner --num_seeds 10 --dataset_name uniner --budget 1000 --methods Agreement

#Uncertainty and Surrogate
CUDA_VISIBLE_DEVICES=0 python3 pos_ner.py --model_name bert --predictor nuner --num_seeds 10 --dataset_name uniner --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 pos_ner.py --model_name bert --predictor nuner --num_seeds 10 --dataset_name uniner --budget 1000 --methods Uncertainty --strategy gaussian_prior &
CUDA_VISIBLE_DEVICES=2 python3 pos_ner.py --model_name bert --predictor nuner --num_seeds 10 --dataset_name uniner --budget 1000 --methods Surrogate --strategy SVM &
CUDA_VISIBLE_DEVICES=3 python3 pos_ner.py --model_name bert --predictor nuner --num_seeds 10 --dataset_name uniner --budget 1000 --methods Surrogate --strategy RF 


#Coverage & Others
CUDA_VISIBLE_DEVICES=0 python3 pos_ner.py --model_name bert --predictor deberta --num_seeds 10 --dataset_name ed-ewt --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 pos_ner.py --model_name bert --predictor deberta --num_seeds 10 --dataset_name ed-ewt --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=3 python3 pos_ner.py --model_name bert --predictor deberta --num_seeds 10 --dataset_name ed-ewt --budget 1000 --methods Stratified &
CUDA_VISIBLE_DEVICES=2 python3 pos_ner.py --model_name bert --predictor deberta --num_seeds 10 --dataset_name ed-ewt --budget 1000 --methods Agreement

#Uncertainty and Surrogate
CUDA_VISIBLE_DEVICES=0 python3 pos_ner.py --model_name bert --predictor deberta --num_seeds 10 --dataset_name ed-ewt --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 pos_ner.py --model_name bert --predictor deberta --num_seeds 10 --dataset_name ed-ewt --budget 1000 --methods Uncertainty --strategy gaussian_prior &
CUDA_VISIBLE_DEVICES=2 python3 pos_ner.py --model_name bert --predictor deberta --num_seeds 10 --dataset_name ed-ewt --budget 1000 --methods Surrogate --strategy SVM &
CUDA_VISIBLE_DEVICES=3 python3 pos_ner.py --model_name bert --predictor deberta --num_seeds 10 --dataset_name ed-ewt --budget 1000 --methods Surrogate --strategy RF 
