import os
import re
import json
import torch
from transformers import TextStreamer, AutoModel, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)


def generate(model_name, prompt, question, model, tokenizer):
    # return "yes"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)[0]["generated_text"]
    return output


def process(model_name, dataset, model, tokenizer):
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

        prediction = generate(model_name, prompt, querry, model, tokenizer)

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
        "microsoft/Phi-3.5-mini-instruct"
    ]

    dataset_list = ["MedMCQA", "MedQA", "PubMedQA"]
    dataset_dict = {'PubMedQA': "clinicalnlplab/pubmedqa_test",
                    "MedMCQA": "clinicalnlplab/medMCQA_test",
                    "MedQA": "clinicalnlplab/medQA_test"}
    
    for model_name in model_list:

        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='.cache')

        for dataset_name in dataset_list:
            dataset = dataset_dict[dataset_name]
            process(model_name, dataset, model, tokenizer)
