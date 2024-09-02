import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from transformers import Trainer,TrainingArguments, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric
import pandas as pd
import evaluate
from tqdm import tqdm
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')

billsum = load_dataset("billsum", split="train")
billsum_test = load_dataset("billsum", split="test")

prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, padding='max_length', truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, padding='max_length', truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_billsum = billsum.map(preprocess_function, batched=True)
tokenized_billsum = tokenized_billsum.remove_columns(billsum.column_names)
tokenized_billsum_test = billsum_test.map(preprocess_function, batched=True)
tokenized_billsum_test = tokenized_billsum_test.remove_columns(billsum_test.column_names)

class KnowledgeDistillationTrainingArguments(TrainingArguments):
  def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
    super().__init__(*args, **kwargs)
    self.alpha = alpha
    self.temperature = temperature


class KnowledgeDistillationTrainer(Trainer):
  def __init__(self, *args, teacher_model=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.teacher_model = teacher_model

  def compute_loss(self, model, inputs, return_outputs=False):
    # Extract cross-entropy loss and logits from student
    outputs_student = model(**inputs)
    loss_ce = outputs_student.loss
    logits_student = outputs_student.logits
    # Extract logits from teacher
    outputs_teacher = self.teacher_model(**inputs)
    logits_teacher = outputs_teacher.logits
    # Computing distillation loss by Softening probabilities
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    loss_kd = self.args.temperature ** 2 * loss_fct(
                F.log_softmax(logits_student / self.args.temperature, dim=-1),
                F.softmax(logits_teacher / self.args.temperature, dim=-1))

    loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd
    return (loss, outputs_student) if return_outputs else loss
  
rouge = evaluate.load("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def show_param_ratio(model):
    num_param = 0
    for param in model.parameters():
        num_param += param.numel()
    num_mask = 0
    for name, param in model.named_buffers():
        if "mask" in name:
            num_mask += (param == 0).sum()
    return (num_param - num_mask) / num_param

# Teacher Model
model_path = "t5small_TextSummarization"
teacher_model = (AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device))
student_model = (AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device))

# Pruning and KD
for i in range(4):
    for module in student_model.modules():
        if isinstance(module, torch.nn.Linear):  # Check the layer type
            prune.l1_unstructured(module, name="weight", amount=0.26)

    print(f'Student Model Parameter Ratio: {show_param_ratio(student_model)}')

    ##### Training
    batch_size = 2
    kd_student_ckpt = "kd_pruned_t5"
    training_args = KnowledgeDistillationTrainingArguments(
        report_to="none",
        output_dir=kd_student_ckpt, 
        #evaluation_strategy = "epoch",
        save_strategy='no',
        num_train_epochs=1,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        #per_device_eval_batch_size=batch_size, 
        alpha=1,
        weight_decay=0.01)

    trainer = KnowledgeDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model, 
        args=training_args,
        train_dataset=tokenized_billsum,
        #compute_metrics=compute_metrics, 
        tokenizer=tokenizer
    )
    trainer.train()

pth = os.path.join(kd_student_ckpt, 'model')
trainer.model.save_pretrained(pth, from_pt=True)
pth_pth = os.path.join(pth, 'kd_pruned_t5.pth')
torch.save(student_model.state_dict(), pth_pth)

##### Load Pruned and KD Model
model = AutoModelForSeq2SeqLM.from_pretrained(pth)

# Apply prune.identity to the layers that were pruned
for module in model.modules():
    if isinstance(module, torch.nn.Linear):  # Check the layer type as per your model's pruned layers
        prune.identity(module, 'weight')
model.load_state_dict(torch.load(pth_pth))
model = model.to(device)

test_targets = []
for i in tqdm(range(len(billsum_test["text"]))):
    input_text = prefix + billsum_test["text"][i]
    inputs = tokenizer(input_text, return_tensors='pt')
    inputs = inputs.to(device)
    pred = model.generate(**inputs)
    recovered_text = tokenizer.decode(pred[0], skip_special_tokens=True)
    #print(recovered_text)
    test_targets.append(recovered_text)

df_results = pd.DataFrame(columns=['ID','Predict'])

for i, prediction in enumerate(test_targets):
    # Escape quotes by replacing "," with "."
    summary_escaped = prediction.replace(',', '.')
    
    # Create a new row DataFrame and append it
    new_row = pd.DataFrame({'ID': [i], 'Predict': [summary_escaped]})
    df_results = pd.concat([df_results, new_row], ignore_index=True)

# Function to escape double quotes and handle newlines
def escape_special_characters(text):
    return text.replace('"', '""').replace('\n', ' ')

# Apply escaping to the 'Summary' column
df_results['Predict'] = df_results['Predict'].apply(escape_special_characters)
df_results.to_csv('submission.csv', index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')

def calculate_lcs(X, Y):
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    # Ensure indices for proper alignment
    solution.set_index(row_id_column_name, inplace=True)
    submission.set_index(row_id_column_name, inplace=True)

    total_score = 0

    for idx in solution.index:
        #if idx not in submission.index:
        #    raise ParticipantVisibleError(f"Missing prediction for ID {idx}.")

        ref_summary = solution.loc[idx, 'Label']
        pred_summary = submission.loc[idx, 'Predict']

        # Tokenize sentences
        ref_sentences = ref_summary.split('.')
        pred_sentences = pred_summary.split('.')

        # Calculate LCS for each sentence pair
        lcs_sum = 0
        for ref_sent in ref_sentences:
            ref_tokens = ref_sent.strip().lower().split()
            best_lcs = 0
            for pred_sent in pred_sentences:
                pred_tokens = pred_sent.strip().lower().split()
                lcs_length = calculate_lcs(ref_tokens, pred_tokens)
                best_lcs = max(best_lcs, lcs_length)
            lcs_sum += best_lcs

        # Calculate ROUGE-L for the current pair of summaries
        ref_length = sum(len(sent.strip().split()) for sent in ref_sentences)
        if ref_length > 0:
            rouge_l = lcs_sum / ref_length
        else:
            rouge_l = 0
        total_score += rouge_l

    # Compute the average ROUGE-L score across all submissions
    mean_rouge_lsum = total_score / len(solution)

    return mean_rouge_lsum

df_label = pd.DataFrame(columns=['ID','Label'])

for i, label in enumerate(billsum_test):
    # Escape quotes by replacing "," with "."
    label_escaped = label['summary'].replace(',', '.')
    
    # Create a new row DataFrame and append it
    new_row = pd.DataFrame({'ID': [i], 'Label': [label_escaped]})
    df_label = pd.concat([df_label, new_row], ignore_index=True)

print(score(df_label, df_results, 'ID'))