#!/bin/bash

#Coverage & Others
CUDA_VISIBLE_DEVICES=0 python3 summarization.py --predictor claude --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor claude --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=2 python3 summarization.py --predictor claude --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Agreement &
CUDA_VISIBLE_DEVICES=3 python3 summarization.py --predictor claude --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor claude --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy gaussian_prior 

CUDA_VISIBLE_DEVICES=0 python3 summarization.py --predictor nova_pro --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor nova_pro --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=2 python3 summarization.py --predictor nova_pro --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Agreement &
CUDA_VISIBLE_DEVICES=3 python3 summarization.py --predictor nova_pro --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor nova_pro --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy gaussian_prior 

CUDA_VISIBLE_DEVICES=0 python3 summarization.py --predictor qwen --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor qwen --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=2 python3 summarization.py --predictor qwen --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Agreement &
CUDA_VISIBLE_DEVICES=3 python3 summarization.py --predictor qwen --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor qwen --model_name bert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy gaussian_prior 


#!/bin/bash

#Coverage & Others
CUDA_VISIBLE_DEVICES=0 python3 summarization.py --predictor claude --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor claude --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=2 python3 summarization.py --predictor claude --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Agreement &
CUDA_VISIBLE_DEVICES=3 python3 summarization.py --predictor claude --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor claude --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy gaussian_prior 

CUDA_VISIBLE_DEVICES=0 python3 summarization.py --predictor nova_pro --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor nova_pro --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=2 python3 summarization.py --predictor nova_pro --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Agreement &
CUDA_VISIBLE_DEVICES=3 python3 summarization.py --predictor nova_pro --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor nova_pro --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy gaussian_prior 

CUDA_VISIBLE_DEVICES=0 python3 summarization.py --predictor qwen --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor qwen --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=2 python3 summarization.py --predictor qwen --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Agreement &
CUDA_VISIBLE_DEVICES=3 python3 summarization.py --predictor qwen --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor qwen --model_name distilbert --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy gaussian_prior 

#!/bin/bash

#Coverage & Others
CUDA_VISIBLE_DEVICES=0 python3 summarization.py --predictor claude --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor claude --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=2 python3 summarization.py --predictor claude --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Agreement &
CUDA_VISIBLE_DEVICES=3 python3 summarization.py --predictor claude --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor claude --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy gaussian_prior 

CUDA_VISIBLE_DEVICES=0 python3 summarization.py --predictor nova_pro --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor nova_pro --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=2 python3 summarization.py --predictor nova_pro --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Agreement &
CUDA_VISIBLE_DEVICES=3 python3 summarization.py --predictor nova_pro --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor nova_pro --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy gaussian_prior 

CUDA_VISIBLE_DEVICES=0 python3 summarization.py --predictor qwen --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Coverage --strategy max_dist &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor qwen --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Random &
CUDA_VISIBLE_DEVICES=2 python3 summarization.py --predictor qwen --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Agreement &
CUDA_VISIBLE_DEVICES=3 python3 summarization.py --predictor qwen --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy mutual_information &
CUDA_VISIBLE_DEVICES=1 python3 summarization.py --predictor qwen --model_name stella --num_seeds 10 --dataset_name cnn --budget 1000 --methods Uncertainty --strategy gaussian_prior 