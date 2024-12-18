#!/bin/bash

# Get list of all nodes with GPUs
nodes=$(sinfo -N -o "%N %G" | grep gpu | awk '{print $1}')

# Loop through each node and check GPU memory
for node in $nodes; do
    echo "Node: $node"
    srun --nodelist=$node --gres=gpu:1 --pty nvidia-smi | grep "Total Memory"
done
