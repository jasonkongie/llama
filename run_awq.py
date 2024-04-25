import os
import subprocess
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


model_path = 'TinyLlama-1.1B-Chat-v1.0'
quant_path = './output/'
llama_cpp_path = './new/'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }


# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def calibrate():
    # Load the dataset
    dataset = load_dataset("lmsys/chatbot_arena_conversations")

    # Define a function to concatenate question and response
    def concatenate_data(examples):
        concatenated_texts = []
        for conversation in examples['conversation_a']:
            if len(conversation) >= 2:  # Ensuring there are at least two turns in the conversation
                concatenated_text = conversation[0]['content'] + " " + conversation[1]['content']
                concatenated_texts.append(concatenated_text)
        return {"text": concatenated_texts}

    # Apply the concatenation to each example in the dataset
    concatenated = dataset.map(concatenate_data, batched=True)

    # Assuming we're working with the 'train' split of the dataset
    return concatenated['train']['text']


# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=calibrate())


#Discard the code below..
# model.quantize(
#     tokenizer,
#     quant_config=quant_config,
#     export_compatible=True
# )

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')