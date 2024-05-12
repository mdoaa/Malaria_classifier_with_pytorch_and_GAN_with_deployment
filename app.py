from __future__ import division, print_function
# coding=utf-8
import torch.nn as nn
import sys
import os
import glob
import numpy as np
import io
import torch 
import torchvision
from torchvision import transforms

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Define a flask app
app = Flask(__name__)

class MosquitoNet(nn.Module):
    
    def __init__(self):
        super(MosquitoNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        self.fc1 = nn.Linear(64*15*15, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.drop = nn.Dropout2d(0.2)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)    # flatten out a input for Dense Layer
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc3(out)
        
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models/model.pt'
model=MosquitoNet()
model.to(device)
#Load your trained model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
     
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')



def model_predict(img_path, model):
    #target_size must agree with what the trained model expects!!
    image = Image.open((img_path))
    # Preprocessing the image
    transformss = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = transformss(image).unsqueeze(0)
    output = model(img)
    _,predicted = torch.max(output,1)
    return str(predicted.item())


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned

        # Arrange the correct return according to the model. 
		# In this model 1 is Parasitized and 0 is Normal.
        str0='Uninfected'
        str1='Parasitized'
        if pred=='0':
            return str1
        else :
            return str0
    
    return None

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
