import json
import jsonlines
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, default_data_collator
import torch
from datasets import Dataset
import os
from peft import PeftModel
from tqdm import tqdm

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'facebook/bart-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

with jsonlines.open('train.json', 'r') as file:
    datasets = []
    for item in file:
        datasets.append(item)

dataset = Dataset.from_list(datasets)

def preprocess(dataset):
    input_encodings = tokenizer(dataset['body'], max_length=1024, padding="max_length", truncation=True, return_tensors='pt')
    target_encodings = tokenizer(dataset['headline'], max_length=150, padding="max_length", truncation=True, return_tensors='pt')
    
    input_encodings['labels'] = target_encodings['input_ids']
    return input_encodings

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# Create a configuration (LoraConfig) where I define LoRA-specific parameters.
lora_config = LoraConfig(
    r = 32,
    lora_alpha = 16, # a scaling factor that adjusts the magnitude of the weight matrix. Usually set to 1
    target_modules = ["q_proj", "v_proj"],
    lora_dropout = 0.05, 
    bias = "none", # this specifies if the bias parameter should be trained. 
    task_type = "SEQ_2_SEQ_LM"
)

# Wrap the base model with get_peft_model() to get a trainable PeftModel.
peft_model = get_peft_model(model, lora_config)
print(peft_model.print_trainable_parameters())


output_directory = os.path.join("cache", "outputs")
training_args = TrainingArguments(
    report_to="none",
    output_dir = output_directory,
    per_device_train_batch_size = 16,
    learning_rate= 2e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=4,
    save_strategy = 'epoch',
)

trainer = Trainer(
    model = peft_model.to(device),
    args = training_args,
    train_dataset = tokenized_dataset,
    tokenizer = tokenizer,
    data_collator = default_data_collator
)
trainer.train()

peft_model_path = os.path.join(output_directory, 'model')
trainer.model.save_pretrained(peft_model_path)
print("**********End of Training**********")

def summarizer(model, tokenizer, dataset):
    results = []
    for item in tqdm(dataset):
        input_ids = tokenizer.encode(item, max_length=1024, truncation=True, return_tensors='pt')
        outputs = model.generate(input_ids=input_ids).to(device)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(result)
    return results

base_model_name = 'facebook/bart-base'
model_dir = 'cache/outputs/model'
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, model_dir)

with jsonlines.open('test.json', 'r') as f:
    dataset = []
    for item in f:
        dataset.append(item['body'])

preds = summarizer(model, tokenizer, dataset)

with open('312555023.json', 'w') as f:
    for item in preds:
        json.dump({"headline": item}, f)
        f.write('\n')

print("**********End of Prediction**********")