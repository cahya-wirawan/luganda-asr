#!/bin/sh
# Run multi parallel the fine-tuning using 8 GPUs.
# All options are writen in finetuning_common_voice.json.

python -m torch.distributed.launch --nproc_per_node=8 run_finetuning.py  finetuning_common_voice.json