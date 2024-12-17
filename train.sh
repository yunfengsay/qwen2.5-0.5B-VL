#!/bin/bash
torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    train.py
