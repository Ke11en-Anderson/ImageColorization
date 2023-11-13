import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
#import matplotlib.pyplot as plt


# print("Testing if GPU is available.")
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("GPU is available and being used.")
# else:
#     device = torch.device("cpu")
#     print("GPU is not available, Falling back to CPU.")


img = Image.open(f"{os.getcwd()}/images/color/testImg.jpg")
img.show()
#img = img.resize((256,256))
#convert_tensor = transforms.ToTensor()
#gray = transforms.Grayscale()
#img = convert_tensor(img)
#img = gray(img)
#print(img.size())
#img.show()