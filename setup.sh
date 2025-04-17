#!/bin/bash

# Set environment variables for memory management
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=-1  # Disable GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MALLOC_ARENA_MAX=2
export PYTHONMALLOC=malloc

# Memory optimization for TensorFlow
export TF_ENABLE_AUTO_MIXED_PRECISION=1
export TF_GPU_THREAD_MODE=gpu_private

# Create necessary directories
mkdir -p .streamlit

# Configure memory limits for Python
ulimit -v 1048576  # Limit virtual memory to 1GB

# Install dependencies
pip install -r requirements.txt

# Clear any existing cache
rm -rf ~/.streamlit/cache/*

echo "Setup completed successfully!"
