import os
import re
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI


def generate(model_name, prompt, question):
    # return 'Y'    # use, when you check the diagnoses list
    client = OpenAI()
    
    prompt ="""Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis. Carefully review the information and determine the correctness of the prediction. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words."""
    
    chat_return = client.chat.completions.create(model=model_name,temperature=0.0, messages=[{"role": "system", "content": prompt}, {"role":"user","content": question}])

    result = chat_return.choices[0].message.content
    return result


def process(model_name, dataset):
    model_name_splited = model_name
    ds = load_dataset(dataset, cache_dir='.cache')
    data_query = ds['test']
    # print(data_query[0])
    # exit()

    answer_json = []
    for index in tqdm(range(len(data_query)), ncols=100):
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

    with open(f'./GPT/{model_name_splited}_{dataset_name}.jsonl', "w+") as fout:
        for item in answer_json:
            fout.write(json.dumps(item)+"\n")
    


if __name__ == "__main__":
    # model_list = ['gpt-4o-2024-08-06']
    # model_list = ['gpt-4-turbo-2024-04-09']
    model_list = ['gpt-3.5-turbo-0125']

    dataset_list = ["MedMCQA", "MedQA", "PubMedQA"]
    dataset_dict = {'PubMedQA': "clinicalnlplab/pubmedqa_test",
                    "MedMCQA": "clinicalnlplab/medMCQA_test",
                    "MedQA": "clinicalnlplab/medQA_test"}
    
    for model_name in model_list:
        for dataset_name in dataset_list:
            dataset = dataset_dict[dataset_name]
            process(model_name, dataset)
