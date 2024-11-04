#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel
import string, re
import csv


# In[2]:


def process_text(txt):
    t_list = []
    for t in txt.split(' '):
        t = t.lower()
        if t.endswith('.'):
            t = t[:-1]
        if t not in stop_words:
            if len(t) > 1:
                t_list.append(t)

    return t_list


# In[3]:


from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))

def extract_correct_option_medmcqa(text, answer):
    """
    Extract the correct option from a text using predefined patterns.
    """
    assert len([ i for i in answer if len(i) > 2]) == 4, answer
    if not text or len(text) == 0:
        return "missing"
    
    text = text.strip()
    
    if text not in ["A", "B", "C", "D"]:
        if len(text) > 1 and text[1]=='.':
            text = text[0]
    
    # 针对杂序列
    count = 0
    if text not in ["A", "B", "C", "D"]:
        for choice in ["A", "B", "C", "D"]:
            if f' {choice}.' in text:
                count += 1
                ans = choice
            elif f' {choice},' in text:
                count += 1
                ans = choice
            elif f' {choice}:' in text:
                count += 1
                ans = choice
            elif f'\n{choice}:' in text:
                count += 1
                ans = choice

            if text.startswith(f'{choice}:'):
                text = choice
            elif text.startswith(f'{choice},'):
                text = choice
            elif text.startswith(f'{choice}.'):
                text = choice
            elif f'Options: {choice}' in text:
                text = choice
            elif f'Answer: {choice}' in text:
                text = choice
    if count == 1:
        text = ans


    # 针对段序列
    if text not in ["A", "B", "C", "D"] and len(text) > 1:
        count = 0
        for an in answer:
            if text in an:
                text = an[0]
            elif an in text:
                count += 1
                ans = an[0]
        if count == 1:
            text = ans

    # 针对长序列
    if text not in ["A", "B", "C", "D"]:
        count = 0
        for an in answer:
            t_list = process_text(text)
            a_list = process_text(an)
            if t_list == [] or a_list == []:
                continue
            
            if set(t_list).issubset(set(a_list)):
                text = an[0] 
            elif set(a_list).issubset(set(t_list)):
                count += 1
                ans = an[0]
        if count == 1:
            text = ans
            

    if text not in ["A", "B", "C", "D"]:
        for an in answer:
            t_list = process_text(text)
            a_list = process_text(an)
            logging.warning('-'*200)
            logging.warning(text)
            logging.warning(t_list)
            logging.warning(an)
            logging.warning(a_list)
        text = 'missing'
    
    return text


# In[4]:


def extract_correct_option_medqa(text, answer):
    """
    Extract the correct option from a text using predefined patterns.
    """
    assert len([ i for i in answer if len(i) > 2]) == 5, answer
    if not text or len(text) == 0:
        return "missing"
    
    text = text.strip()
    
    if text not in ["A", "B", "C", "D", "E"]:
        if len(text) > 1 and text[1]=='.':
            text = text[0]
            
    # 针对杂序列
    count = 0
    if text not in ["A", "B", "C", "D", "E"]:
        for choice in ["A", "B", "C", "D", "E"]:
            if f' {choice}.' in text:
                count += 1
                ans = choice
            elif f' {choice},' in text:
                count += 1
                ans = choice
            elif f' {choice}:' in text:
                count += 1
                ans = choice
            elif f'\n{choice}:' in text:
                count += 1
                ans = choice

            if text.startswith(f'{choice}:'):
                text = choice
            elif text.startswith(f'{choice},'):
                text = choice
            elif text.startswith(f'{choice}.'):
                text = choice
            elif f'Options: {choice}' in text:
                text = choice
            elif f'Answer: {choice}' in text:
                text = choice
    if count == 1:
        text = ans

    # 针对短序列
    if text not in ["A", "B", "C", "D", "E"] and len(text) > 1:
        count = 0
        for an in answer:
            if text in an:
                text = an[0]
            elif an in text:
                count += 1
                ans = an[0]
        if count == 1:
            text = ans

    if text not in ["A", "B", "C", "D", "E"]:
        count = 0
        for an in answer:
            t_list = process_text(text)
            a_list = process_text(an)
            if t_list == [] or a_list == []:
                continue
            
            if set(t_list).issubset(set(a_list)):
                text = an[0]
            elif set(a_list).issubset(set(t_list)):
                count += 1
                ans = an[0]
        if count == 1:
            text = ans

    if text not in ["A", "B", "C", "D", "E"]:
        for an in answer:
            t_list = process_text(text)
            a_list = process_text(an)
            logging.warning('-'*200)
            logging.warning(text)
            logging.warning(t_list)
            logging.warning(an)
            logging.warning(a_list)

    return text


# In[5]:


def extract_correct_option_pubmedqa(text):
    """
    Extract the correct option from a text using predefined patterns.
    """

    if not text or len(text) == 0:
        return "missing"
    
    text = text.lstrip()
    text = text.lower()
    # return text
    
    if text not in ["yes", "no", "maybe"]:
        if text.startswith('yes'):
            text = 'yes'
        elif text.startswith('no'):
            text = 'no'
        elif text.startswith('maybe'):
            text = 'maybe'
    
    if text not in ["yes", "no", "maybe"]:
        if 'answer: maybe' in text or 'answer is maybe' in text:
            text = 'maybe'
        elif 'answer: no' in text or 'answer is no' in text:
            text = 'no'
        elif 'answer: yes' in text or 'answer is yes' in text:
            text = 'yes'

        
    
    return text


# In[6]:


def process_jsonl_format(f):
    data = []
    with open(f, 'r', encoding='utf-8') as file:
        for line in file:
            # item = json.loads(line.strip())
            # data.append(item)
            flag = False
            try:
                item = json.loads(line.strip())  # 尝试解析 JSON
                data.append(item)
            except json.JSONDecodeError as e:
                flag = True
                print(f"Offending line: {line.strip()}")
            if flag:
                exit()
    golds = []
    preds = []
    answers = []
    data_list = []
    for i in range(len(data)):
        if 'MedMCQA' in f:
            answer = data[i]['doc']['query'].split('Options:')[1].split('The answer is')[0].strip()
            answer = answer.split('\n')
            pred = extract_correct_option_medmcqa(data[i]['target'], answer)
            gold = extract_correct_option_medmcqa(data[i]['resps'][0][0], answer)
        elif 'PubMedQA' in f:
            pred = extract_correct_option_pubmedqa(data[i]['target'])
            gold = extract_correct_option_pubmedqa(data[i]['resps'][0][0])
        elif 'MedQA' in f:
            answer = data[i]['doc']['query'].split('Options:')[1].split('The answer is')[0].strip()
            answer = answer.split('\n')
            pred = extract_correct_option_medqa(data[i]['target'], answer)
            gold = extract_correct_option_medqa(data[i]['resps'][0][0], answer)
        else:
            print('Error Foramt')

        golds.append(pred)
        preds.append(gold)
        data_list.append(data[i])

    
    return preds, golds, data


# In[7]:


def compute_accuracy_scores(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    return round(accuracy, 4)


# In[8]:


def compute_f1_scores(predictions, true_labels):
    f1 = f1_score(true_labels, predictions, average='macro')
    return round(f1, 4)


# In[9]:


def magic(file, preds):
    cho_medmc = ["A", "B", "C", "D"]
    cho_med = ["A", "B", "C", "D", "E"]
    cho_pubmed = ["maybe", "no", "yes"]
    # cho_pubmed = ["yes", "no","maybe"]

    if 'MedMCQA' in file:
        for i in range(len(preds)):
            if preds[i] not in cho_medmc:
                error.append(data[i])
                if len(preds[i]) > 0 and preds[i][0] in cho_medmc:
                    preds[i] = preds[i][0]
                else:
                    preds[i] = 'missing'
    elif 'PubMedQA' in file:
        print('PubMed')
        for i in range(len(preds)):
            if preds[i] not in cho_pubmed:
                error.append(data[i])
                for cho in cho_pubmed:
                    if cho in preds[i]:
                        preds[i] = cho
            if preds[i] not in cho_pubmed:
                preds[i] = 'missing'

    elif 'MedQA' in file:
        for i in range(len(preds)):
            if preds[i] not in cho_med:
                error.append(data[i])
                if len(preds[i]) > 0 and preds[i][0] in cho_med:
                    preds[i] = preds[i][0]
                else:
                    preds[i] = 'missing'
    else:
        print('Error format')

    dic = {}
    for i in preds:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1
    print(file, ':')
    print(dic)

    return preds, dic
    


# # Calculate scores

# In[10]:


import os
import logging

# 配置logging模块，将日志输出到output.log文件
logging.basicConfig(filename="/home/gy237/project/light_weight_llama/output.log", filemode="w", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# 使用logging.info()写入日志
logging.info("This is an info message.")

choices = ["A", "B", "C", "D", "E", "yes", "no", "maybe"]

file_path = '/home/gy237/project/light_weight_llama/data_12'
files = os.listdir(file_path)
files = [i for i in files if i.endswith('.jsonl')]
files = sorted(files)
print(len(files))

results = []
error = []
for file in files:
    file = f'{file_path}/{file}'
    error.append(file)

    preds, golds, data = process_jsonl_format(file)
    preds, dic = magic(file, preds)

    golds_e = [i for i in golds if i not in choices]
    assert len(golds_e) == 0
    # assert len(preds) in [4183, 500, 1273]
    
    accuracy = compute_accuracy_scores(preds, golds)
    f1_scores = compute_f1_scores(preds, golds)
    
    results.append({
                'file': file,
                'accuracy': accuracy,
                'f1_scores': f1_scores,
                'labe_dic': dic,
            })


print(len(error))


# In[ ]:


import json
with open(f'{file_path}/results.json', mode='w', encoding='utf-8') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)

with open(f'{file_path}/rerror.json', mode='w', encoding='utf-8') as file:
    json.dump(error, file, ensure_ascii=False, indent=4)


# In[ ]:




