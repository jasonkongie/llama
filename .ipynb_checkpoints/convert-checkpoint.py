
import os
import subprocess
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'TinyLlama-1.1B-Chat-v1.0'
quant_path = './output/'
llama_cpp_path = './new/'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }


# GGUF conversion
print('Converting model to GGUF...')
llama_cpp_method = "q4_K_M"
convert_cmd_path = os.path.join(llama_cpp_path, "convert.py")
quantize_cmd_path = os.path.join(llama_cpp_path, "quantize")

if not os.path.exists(llama_cpp_path):
    cmd = f"git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_path} && cd {llama_cpp_path} && make LLAMA_CUBLAS=1 LLAMA_CUDA_F16=1"
    subprocess.run([cmd], shell=True, check=True)

subprocess.run([
    f"python {convert_cmd_path} {quant_path} --outfile {quant_path}/model.gguf"
], shell=True, check=True)

subprocess.run([
    f"{quantize_cmd_path} {quant_path}/model.gguf {quant_path}/model_{llama_cpp_method}.gguf {llama_cpp_method}"
], shell=True, check=True)

