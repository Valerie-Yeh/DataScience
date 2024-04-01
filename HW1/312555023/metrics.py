import numpy as np
from sklearn.metrics import f1_score
import json

with open('image_ground_truths.json', 'r') as file:
    json_data = json.load(file)
    true = json_data.get('image_ground_truths', [])

with open('image_predictions.json', 'r') as file:
    json_data = json.load(file)
    pred = json_data.get('image_predictions', [])

score = f1_score(true, pred, average='micro')
print(score)