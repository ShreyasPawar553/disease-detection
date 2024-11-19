from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pickle

app = Flask(__name__)

# Define the neural network class
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.fc1 = nn.Linear(in_features=48 * 12 * 12, out_features=240)
        self.fc2 = nn.Linear(in_features=240, out_features=120)
        self.out = nn.Linear(in_features=120, out_features=17)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.conv3(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.conv4(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = t.reshape(-1, 48 * 12 * 12)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        return self.out(t)

# Load the pre-trained model
model = Network()
model.load_state_dict(torch.load("model002_ep20.pth"))  # Adjust the path if needed
model.eval()

# Load reference labels
with open('labels.json', 'rb') as f:
    reference = pickle.load(f)

# Define transformation for image resizing
resize = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Plant Disease Prediction API
@app.route('/predict-disease', methods=['POST'])
def disease_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Process the image
        image = Image.open(file).convert('RGB')
        image = resize(image).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            y_result = model(image)
            result_idx = y_result.argmax(dim=1).item()

        # Find the corresponding label
        predicted_label = [k for k, v in reference.items() if v == result_idx][0]

        return jsonify({"predicted_label": predicted_label})

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


import os

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Get port from environment or default to 5000
    app.run(host='0.0.0.0', port=port)
