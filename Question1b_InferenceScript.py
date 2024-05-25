from torchvision import transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import os

output_file = 'preds.csv'

model = models.resnet50()

# # Change the last layer to classify into 6 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)

# Define the file paths
model_path = 'model_weights.pth'

# Load the saved model
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
print(f"Model weights loaded from '{model_path}'")

model.eval()

# Define the transformations to be applied to each image
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])
# Loop through all images in the folder
outputs = []
for filename in os.listdir():
    if filename.endswith('.jpg'):
        # Open and transform the image
        img = Image.open(filename)
        img = transform(img)

        # Make a prediction
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            prediction = torch.argmax(output, dim=1)
        
        # Print the filename and predicted class
        row = [filename, prediction.item()]
        outputs.append(row)

np.array(outputs)

data = pd.DataFrame(outputs)
data.to_csv(output_file, header=['image', 'class'], index=False)

print(f"Predictions saved to '{output_file}'")