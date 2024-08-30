#!/bin/bash
python train.py  --model_name="meta-llama/Meta-Llama-3-8B" --new_model="StarkWizard/Meta-Llama-3-8B-PEFT" --max_seq_length=1024 --window=512 --lr=2e-4 q_proj k_proj v_proj o_proj up_proj down_proj gate_proj lm_head --epochs=3 --wandb_project="Llama"
