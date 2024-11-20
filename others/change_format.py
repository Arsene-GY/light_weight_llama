import os
import json

file_path = './other'

files = os.listdir(file_path)
files = [i for i in files if i.endswith('.jsonl')]
files = sorted(files)
print(len(files))

for file in files:
    data = []
    file_ = file_path + '/' + file
    with open(file_, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())  # 尝试解析 JSON
            data.append(item)

    print(len(data))
    for i in range(len(data)):
        data[i]['resps'] = data[i]['resps'][0]['generated_text']
    
    with open(f'./others/{file}', "w+") as fout:
        for item in data:
            fout.write(json.dumps(item)+"\n")
    