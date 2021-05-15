import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import numpy as np
import cv2
from torchvision.transforms import Compose, Normalize, ToTensor
import random


class binary_model(nn.Module):

    def __init__(self, number_classes=2):
        super(binary_model, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.classes = number_classes
        self.model._fc = nn.Linear(in_features=2560, out_features=self.classes, bias=True)

    def forward(self, x):
        x = self.model.forward(x)
        l = nn.Sigmoid()
        x = l(x)
        return x


def process_bb(model, I, bounding_boxes, image_size=(412, 412)):
    """
    :param model: A binary model to create the bounding boxes
    :param I: PIL image
    :param bounding_boxes: Bounding boxes containing regions of interest
    :param image_size: Choose the size of the patches
    :return: Patches with the class of the ROIS
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patches = np.array([])
    normalization = Compose([ToTensor(),
                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    for (x, y, w, h) in bounding_boxes:
        patch = np.array(I.crop((x, y, x + w, y + h)))
        patch = cv2.resize(patch, image_size)
        patch = normalization(patch).unsqueeze(0).to(device)
        patch = model(patch).cpu().detach().numpy()
        patches = np.concatenate(patches, patch)
    return patches

def draw_rois(I,bounding_boxes,scores,threshold = 0.5):
    """
    Draws the bounding boxes on image I
    :param I: Image as numpy array
    :param bounding_boxes: Bounding boxes
    :param scores: Score of each bounding boxes
    :param threshold: Threshold to show bounding boxes
    :return: Numpy array with bounding boxes on it
    """
    I2 = I.copy()
    i = 0
    for (x, y, w, h) in bounding_boxes:
        if scores[i][0][0]>0.4:
            # draw the region proposal bounding box on the image
            color = [random.randint(0, 255) for j in range(0, 3)]
            cv2.rectangle(I2, (x, y), (x + w, y + h), color, 2)
        i = i+1
    return I2

def draw_ROIs(I,bounding_boxes):
    """
    Draws all the bounding boxes in the image
    :param I: Image
    :param bounding_boxes: Bounding boxes to draw in the image
    :retrun: Image with the bounding boxes
    """
    I2 = I.copy()
    for (x, y, w, h) in bounding_boxes:
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for _ in range(0, 3)]
        cv2.rectangle(I2, (x, y), (x + w, y + h), color, 2)













