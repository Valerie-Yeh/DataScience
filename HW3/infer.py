import json
import jsonlines
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, default_data_collator
import torch
from datasets import Dataset
import os
from peft import PeftModel
from tqdm import tqdm
import sys

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def summarizer(model, tokenizer, dataset):
    results = []
    for item in tqdm(dataset):
        input_ids = tokenizer.encode(item, max_length=1024, truncation=True, return_tensors='pt')
        outputs = model.generate(input_ids=input_ids).to(device)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(result)
    return results


if __name__ == '__main__':

    base_model_name = 'facebook/bart-base'
    model_dir = 'cache/outputs/model'
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, model_dir)

    with jsonlines.open(sys.argv[1], 'r') as f:
        dataset = []
        for item in f:
            dataset.append(item['body'])

    preds = summarizer(model, tokenizer, dataset)

    with open('312555023.json', 'w') as f:
        for item in preds:
            json.dump({"headline": item}, f)
            f.write('\n')
    print("**********End of Prediction**********")