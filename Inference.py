import os
import re
import json
import torch
from transformers import TextStreamer, AutoModel, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datasets import load_dataset


def generate(model_name, prompt, question):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        cache_dir='.cache'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='.cache')

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer


def process(model_name, dataset):
    model_name_splited = model_name.split('/')[1]
    ds = load_dataset(dataset, cache_dir='.cache')
    data_query = ds['test']
    # print(data_query[0])
    # exit()

    answer_json = []
    for index in range(len(data_query)):
        current_answer = {}
        prompt = data_query[index]["query"].split('\n')[0]
        querry = '\n'.join(data_query[index]["query"].split('\n')[1:])
        prediction = generate(model_name, prompt, querry)

        # 转换成目标格式
        current_answer["doc_id"] = data_query[index]["id"]
        current_answer["prompt_0"] = data_query[index]["query"]
        current_answer["target"] = data_query[index]["answer"]
        current_answer["resps"] = prediction
        answer_json.append(current_answer)
        # print(current_answer)
        # exit()

    with open(f'./Qwen/{model_name_splited}_{dataset_name}.jsonl', "w+") as fout:
        for item in answer_json:
            fout.write(json.dumps(item)+"\n")
    


if __name__ == "__main__":
    model_list = [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]

    dataset_list = ["PubMedQA", "MedMCQA", "MedQA"]
    dataset_dict = {'PubMedQA': "clinicalnlplab/pubmedqa_test",
                    "MedMCQA": "clinicalnlplab/medMCQA_test",
                    "MedQA": "clinicalnlplab/medQA_test"}
    
    for model_name in model_list:
        for dataset_name in dataset_list:
            dataset = dataset_dict[dataset_name]
            process(model_name, dataset)
