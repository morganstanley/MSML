# python main.py --mode grpo --vlm-name qwen-2b --dataset-name chartqa-src
# accelerate launch --config_file=deepspeed_zero3.yaml  main.py --mode grpo --vlm-name qwen-2b --dataset-name chartqa-src

accelerate launch --config_file=deepspeed_zero3.yaml main.py --mode grpo --vlm-name qwen2-5-3b --dataset-name chartqa-src