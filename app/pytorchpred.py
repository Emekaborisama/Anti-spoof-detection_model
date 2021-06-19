from haroun import Data, Model, ConvPool, losses
from haroun.losses import rmse
import numpy as np
import os
import pathlib
import skimage.io as io
import skimage.transform as sktransform
import skimage.color as color
import torch
import tensorflow as tf
import skimage.exposure as ex


import os
here = os.path.dirname(__file__)
modelpath = os.path.join(here, 'module.pth')

class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.input_norm = torch.nn.BatchNorm2d(3, affine=False)
        self.layer1 = ConvPool(in_features=3, out_features=8)
        self.layer2 = ConvPool(in_features=8, out_features=16)
        self.layer3 = ConvPool(in_features=16, out_features=32)
        self.layer4 = ConvPool(in_features=32, out_features=64)
        self.layer5 = ConvPool(in_features=64, out_features=128)
        self.layer6 = ConvPool(in_features=128, out_features=256)
        
        

        self.net = torch.nn.Sequential(self.layer1, self.layer2, self.layer3, 
                                       self.layer4, self.layer5, self.layer6)
            
        
        self.fc1 = torch.nn.Linear(in_features=256, out_features=128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        
        self.fc2 = torch.nn.Linear(in_features=128, out_features=32)
        self.bn2 = torch.nn.BatchNorm1d(32)

        self.fc3 = torch.nn.Linear(in_features=32, out_features=8)
        self.bn3 = torch.nn.BatchNorm1d(8)

        self.fc4 = torch.nn.Linear(in_features=8, out_features=2)


        self.lin = torch.nn.Sequential(self.fc1, self.bn1, self.fc2, self.bn2,
                                       self.fc3, self.bn3, self.fc4)  


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.input_norm(X)
        X = self.net(X)
        X = X.reshape(X.size(0), -1)
        X = self.lin(X)
        X = torch.nn.functional.elu(X, alpha=1.0, inplace=False)
        return X
device = torch.device("cpu")

net = Network()
AntiSpoofClassifier = Model(net, "adam", rmse, device)


def flip(images, axis):
    flipped_images = np.flip(images, axis)
    return flipped_images


def gamma(images, gamma):
    brightness_images = np.array([ex.adjust_gamma(image, gamma, gain=1) for image in images])
    return brightness_images


def augmentation(images, flip_y, flip_x, brightness):
    if flip_y:
        # Data augmentation (flip_horizontal)
        flipped_y_images = flip(images, axis=2)

        # Concatenate arrays
        images = np.concatenate([images, flipped_y_images])

    if flip_x:
        # Data augmentation (flip_horizontal)
        flipped_x_images = flip(images, axis=1)

        # Concatenate arrays
        images = np.concatenate([images, flipped_x_images])

    if brightness:
        darken_images = gamma(images, gamma=1.5)
        brighten_images = gamma(images, gamma=0.5)

        # Concatenate arrays
        images = np.concatenate([images, darken_images, brighten_images])

    return images

map_location=torch.device('cpu')

state_dict = torch.load(modelpath, map_location)
model3 = Network()
model3.load_state_dict(state_dict, map_location)

def load_predict(image_path):
    image = io.imread(image_path)
    image = sktransform.resize(image, (64, 64))
    image = np.expand_dims(image, 0)
    images = augmentation(image, flip_y=True, flip_x=True, brightness=True)
    x = torch.from_numpy(images).to(torch.float32)
    x = x.permute(0,3,1,2)
    real, fake = 0, 0
    preds = torch.nn.functional.softmax(model3(x), dim=0).cpu().detach().numpy()
    for pred in preds:
        if np.argmax(pred) == 0:
            real +=1
        elif np.argmax(pred) == 1:
            fake +=1
    preds = np.argmax(preds[0])
    if preds == 0:
        print("real")
    elif preds == 1:
        print("pls retake the profile image")
    return(f"real : {100*real/(real+fake):.3g}% , fake : {100*fake/(real+fake):.3g}%")