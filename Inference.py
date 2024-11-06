# 保存json
import os
import re
import json
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer, AutoModel, AutoTokenizer, AutoModelForCausalLM
from unsloth.chat_templates import get_chat_template
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()


INFER_PATH = "your path"
EVALUATE_PATH = "your path"




pubmed_prompt = """Your task is to answer biomdedical qustions using the given abstract. Please respond only with 'yes', 'no', or 'maybe'. Provide no additional text or repetition of this instruction."""

def key_value_map_data(item):
    prompt = f"Instruction:{pubmed_prompt}\nINPUT: {item}"
    return [{
                "role": "user",
                "content":prompt ,
            },
            {
                "role": "assistant",
                "content": ""
            }]


def key_value_process_function(cur_dataset):
    def map_function(item):
        # 对每一项的 "conversations" 列表进行映射
        item["query"] = key_value_map_data(item["query"])
        return item
    
    train_dataset = load_dataset(cur_dataset, split='test')
    # 应用映射函数
    processed_dataset = train_dataset.map(map_function)

    processed_dataset = [item for item in processed_dataset]
    return processed_dataset


def process(model_name, dataset):


    model_name_splited = model_name[model_name.find("/")+1: ]
    data_query = key_value_process_function(dataset)
    # print(data_query[0])
    # exit()

    max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        # mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
        mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
        # 要对照这个格式做出一个映射！
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------- # 
    # 预测后转换成目标的格式
    # huggingface文件中的属性：id, query（原始问题）, answer（ABCD答案）, choices（ABCD列表）, gold(数字)
    # 转换模型的模式，从 训练 -> 推理
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    answer_json = list()
    count = 0 
    for index in range(len(data_query)):
        count += 1
        current_answer = dict()
        query_messages = data_query[index]["query"]
        inputs = tokenizer.apply_chat_template(
                query_messages,
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
                ).to("cuda")

        text_streamer = TextStreamer(tokenizer)

        input_length = inputs.shape[1]  # 计算输入的长度
        # 生成答案
        answer_output_id = model.generate(
            input_ids=inputs, 
            streamer=text_streamer, 
            max_new_tokens=128, 
            pad_token_id=tokenizer.pad_token_id, 
            use_cache=True
        )
        # 只解码新生成的部分（不包括输入的 prompt）
        print("=== All answer output ===\n", tokenizer.decode(answer_output_id[0], skip_special_tokens=True))
        answer_text = tokenizer.decode(answer_output_id[0][input_length:], skip_special_tokens=True)
        # print("&"*10, "Original Output")
        print("=== Skip ===\n", answer_text)  # 只显示模型生成的答案

        # 转换成目标格式
        current_answer["doc_id"] = data_query[index]["id"]
        current_answer["prompt_0"] = data_query[index]["query"]
        current_answer["target"] = data_query[index]["answer"]
        current_answer["resps"] = answer_text
        answer_json.append(current_answer) 

    # ============================= #

    save_dir = INFER_PATH
    os.makedirs(save_dir, exist_ok=True)

    prediction_save_path = os.path.join(save_dir, f"{model_name_splited}_{dataset_name}.jsonl")

    with open(prediction_save_path, "w+") as fout:
        for item in answer_json:
            fout.write(json.dumps(item)+"\n")
    


if __name__ == "__main__":
    model_list = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta-llama/Llama-2-70b-chat",
        "YBXL/MeLLaMA-70B-chat",
    ]

    dataset_list = ["PubMedQA", "MedMCQA", "MedQA"]
    dataset_dict = {'PubMedQA': "clinicalnlplab/pubmedqa_test",\
                    "MedMCQA": "clinicalnlplab/medMCQA_test",\
                    "MedQA": "clinicalnlplab/medQA_test"}
    

    for model_name in model_list:
        for dataset_name in dataset_list:
            dataset = dataset_dict[dataset_name]
            process(model_name, dataset)



    
