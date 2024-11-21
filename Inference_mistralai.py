import os
import re
import json
import torch
from transformers import TextStreamer, AutoModel, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import login

# 输入你的访问令牌进行登录
login('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')

def generate(model_name, prompt, question, llm):
    return "yes"
    messages = [
        {
            "role": "user",
            "content": prompt+question
        },
    ]

    outputs = llm.chat(messages, sampling_params=sampling_params)

    return outputs


def process(model_name, dataset, llm):
    model_name_splited = model_name.split('/')[1]
    ds = load_dataset(dataset, cache_dir='.cache')
    data_query = ds['test']
    # print(data_query[0])
    # exit()

    answer_json = []
    for index in tqdm(range(len(data_query)), ncols=100):
        current_answer = {}
        prompt = data_query[index]["query"].split('\n')[0]
        querry = '\n'.join(data_query[index]["query"].split('\n')[1:])

        prediction = generate(model_name, prompt, querry, llm)

        # 转换成目标格式
        current_answer["doc_id"] = data_query[index]["id"]
        current_answer["prompt_0"] = data_query[index]["query"]
        current_answer["target"] = data_query[index]["answer"]
        current_answer["resps"] = prediction
        answer_json.append(current_answer)
        # print(current_answer)
        # exit()

    with open(f'./others/{model_name_splited}_{dataset_name}.jsonl', "w+") as fout:
        for item in answer_json:
            fout.write(json.dumps(item)+"\n")
    


if __name__ == "__main__":
    model_list = [
        "mistralai/Ministral-8B-Instruct-2410",
    ]

    dataset_list = ["MedMCQA", "PubMedQA", "MedQA"]
    dataset_dict = {'PubMedQA': "clinicalnlplab/pubmedqa_test",
                    "MedMCQA": "clinicalnlplab/medMCQA_test",
                    "MedQA": "clinicalnlplab/medQA_test"}
    
    for model_name in model_list:
        sampling_params = SamplingParams(max_tokens=5000)
        # note that running Ministral 8B on a single GPU requires 24 GB of GPU RAM
        # If you want to divide the GPU requirement over multiple devices, please add *e.g.* `tensor_parallel=2`
        llm = LLM(model=model_name, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")

        for dataset_name in dataset_list:
            dataset = dataset_dict[dataset_name]
            process(model_name, dataset, llm)
