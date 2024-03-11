import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
from torch import nn
import torch.optim as optim
import cv2
import json
import sys
from tqdm import tqdm
from model import Model


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(image_name):
    # Convert the file path to an image
    img = cv2.imread(image_name)
    # Resize the image and normalize the pixel range to be within 0 and 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype("float32") / 255.0
    img -= MEAN
    img /= STD
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    img = img.to(device)
    # Use the model to predict the class representing the image
    model = Model(test = True).to(device)
    model.eval()
    checkpoint = torch.load('resnet50_checkpoint/epoch10.pyt')
    model.load_state_dict(checkpoint['resnet_classifier'])
    pred = model(img)
    probabilities = torch.nn.Softmax(dim=-1)(pred)
    # Print the predicted character's name
    return probabilities.argmax().item()
    


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file:
        json_data = json.load(file)

    image_paths = json_data.get('image_paths', [])

    image_predictions = []
    for i in tqdm(range(len(image_paths))):
        result = test(image_paths[i])
        image_predictions.append(result)

    with open('image_predictions.json', 'w') as f:
            json.dump({"image_predictions": image_predictions}, f, indent=4)
    print("Finish Prediction")