#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Ensure the script outputs the execution of commands.
set -x


#set-up
pip install autoawq

git clone https://github.com/ggerganov/llama.cpp

git lfs install

git lfs clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 #replace with new model, etc

# Command to run AWQ Quantization
python run_awq.py 

# Command to run convert.py (convert to GGUF)
python convert.py

# python ./llama.cpp/convert.py TinyLlama-1.1B-Chat-v1.0/ --outfile TinyLlama-1.1B-Chat-v1.0/TinyLlama.gguf

# Command to run the chat application
python chat.py --model_path ./TinyLlama.gguf
