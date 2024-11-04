from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("clinicalnlplab/pubmedqa_test", cache_dir='/home/gy237/project/llama3/new_data')

data = ds['train'] + ds['test'] + ds['valid']
data = list(set(data))

print(data)