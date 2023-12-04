import copy
import math
import random
import time
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import os
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from collections import OrderedDict, defaultdict
from typing import Union, List
from torchprofile import profile_macs # Helps us to obtain mac calculations
from torch.utils.data import Dataset
from PIL import Image

IMAGESIZE = 128

@torch.no_grad()
def processFrame(model, cam, device):
    ret, frame = cam.read()
    if not ret:
        cam.release()
        cv2.destroyAllWindows()
        raise Exception("uh huh huh no camera frame")
    
    frameOriginal = cv2.resize(frame, (IMAGESIZE,IMAGESIZE))
    cv2.imshow("original", frameOriginal)
    frameBlackWhite = cv2.cvtColor(frameOriginal, cv2.COLOR_BGR2GRAY)
    cv2.imshow("BW", frameBlackWhite)
    
    cv2tensor = transforms.ToTensor()
    
    inp = cv2tensor(frameBlackWhite).to(device).reshape(1,1,IMAGESIZE,IMAGESIZE)
    inp = torch.cat((inp,inp,inp),1)

    output = model(inp)

    tmp = output.reshape(3,IMAGESIZE,IMAGESIZE).cpu().numpy()
    tmp = np.transpose(tmp, (1,2,0))

    frameOutput = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    cv2.imshow("output", frameOutput)



class GrayNet(nn.Module):
    def __init__(self):
        super(GrayNet, self).__init__()
        channels = 64
        #block 1
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        #block 2
        self.conv3 = nn.Conv2d(channels, channels*2, 3, padding=1)
        self.conv4 = nn.Conv2d(channels*2, channels*2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels*2)
        self.bn4 = nn.BatchNorm2d(channels*2)

        #block 3
        self.conv5 = nn.Conv2d(channels*2, channels*4, 3, padding=1)
        self.conv6 = nn.Conv2d(channels*4, channels*4, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(channels*4)
        self.bn6 = nn.BatchNorm2d(channels*4)

        #block 4
        self.conv7 = nn.Conv2d(channels*4, channels*8, 3, padding=1)
        self.conv8 = nn.Conv2d(channels*8, channels*8, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(channels*8)
        self.bn8 = nn.BatchNorm2d(channels*8)

        #block 5
        self.conv9 = nn.Conv2d(channels*8, channels*8, 3, padding=1)
        self.conv10 = nn.Conv2d(channels*8, channels*8, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(channels*8)
        self.bn10 = nn.BatchNorm2d(channels*8)
        self.conv11 = nn.ConvTranspose2d(channels*8,channels*8, kernel_size=2,stride=2)

        #block 6
        self.conv12 = nn.Conv2d(channels*16, channels*4, 1, padding=0)
        self.conv13 = nn.Conv2d(channels*4, channels*4, 3, padding=1)
        self.conv14 = nn.Conv2d(channels*4, channels*4, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(channels*4)
        self.bn13 = nn.BatchNorm2d(channels*4)
        self.bn14 = nn.BatchNorm2d(channels*4)
        self.conv15 = nn.ConvTranspose2d(channels*4,channels*4, kernel_size=2,stride=2)

        #block 7
        self.conv16 = nn.Conv2d(channels*8, channels*2, 1, padding=0)
        self.conv17 = nn.Conv2d(channels*2, channels*2, 3, padding=1)
        self.conv18 = nn.Conv2d(channels*2, channels*2, 3, padding=1)
        self.bn16 = nn.BatchNorm2d(channels*2)
        self.bn17 = nn.BatchNorm2d(channels*2)
        self.bn18 = nn.BatchNorm2d(channels*2)
        self.conv19 = nn.ConvTranspose2d(channels*2,channels*2, kernel_size=2,stride=2)

        #block 8
        self.conv20 = nn.Conv2d(channels*4, channels, 1, padding=0)
        self.conv21 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv22 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn20 = nn.BatchNorm2d(channels)
        self.bn21 = nn.BatchNorm2d(channels)
        self.bn22 = nn.BatchNorm2d(channels)
        self.conv23 = nn.ConvTranspose2d(channels,channels, kernel_size=2,stride=2)

        #block 9
        self.conv24 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv25 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv26 = nn.Conv2d(channels, 3, 3, padding=1)
        self.bn24 = nn.BatchNorm2d(channels)
        self.bn25 = nn.BatchNorm2d(channels)
        self.bn26 = nn.BatchNorm2d(3)

    def forward(self, x):
        #print(x.size())
        # Input 64x64x3
        #block1
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))) # 32x32xchannels
        block1 = x
        x = F.max_pool2d(x,2)
        
        #block2
        x = F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))) # 16x16xchannels*2
        block2 = x
        x = F.max_pool2d(x,2)
        
        #block3
        x = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))) # 8x8xchannels*4
        block3 = x
        x = F.max_pool2d(x,2)
        
        #block4
        x = F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(x)))))) # 4x4xchannels*8
        block4 = x
        x = F.max_pool2d(x,2)
        
        #block5
        x = self.conv11(F.relu(self.bn10(self.conv10(F.relu(self.bn9(self.conv9(x))))))) # 8x8xchannels*8

        #block6
        x = torch.cat((x,block4),1) #8x8xchannels*16
        x = self.conv15(F.relu(self.bn14(self.conv14(F.relu(self.bn13(self.conv13(F.relu(self.bn12(self.conv12(x)))))))))) #16x16xchannels*4
        
        #block7
        x =torch.cat((x,block3),1)#16x16xchannels*8
        x = self.conv19(F.relu(self.bn18(self.conv18(F.relu(self.bn17(self.conv17(F.relu(self.bn16(self.conv16(x)))))))))) #32x32xchannels*2
        
        #block8
        x =torch.cat((x,block2),1) #32x32xchannels*4
        x = self.conv23(F.relu(self.bn22(self.conv22(F.relu(self.bn21(self.conv21(F.relu(self.bn20(self.conv20(x)))))))))) #64x64xchannels
        
        #block9
        x = F.relu(self.bn26(self.conv26(F.relu(self.bn25(self.conv25(F.relu(self.bn24(self.conv24(x))))))))) #64x64x3 RGB
        return x


print("Testing if GPU is available.")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU is not available, Falling back to CPU.")


## Declare camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

model = GrayNet().to(device)
# Load model from file.
loadPath = "finaltrainedweights.pt"#"largeimagetrainedweights.pt"
model.load_state_dict(torch.load(loadPath))

while True:
    processFrame(model, cam, device)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
                
cam.release()
cv2.destroyAllWindows()

