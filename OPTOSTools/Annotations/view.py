import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..Annotations.Formats import VOC_format_V2


def show_annotations(Annotations: VOC_format_V2, FilePath="", frsize=(16, 8),show = True):
    """Automatically plots the image with the annotations in it and returns the marked image"""
    # Get the right path
    if FilePath == "":
        FilePath = os.path.join(Annotations.folder.text, Annotations.filename.text)
    else:
        FilePath = os.path.join(FilePath, Annotations.filename.text)
    # Load image into memory
    I = cv2.cvtColor(cv2.imread(FilePath), cv2.COLOR_BGR2RGB)
    # Create colors for the different classes
    colors = {}
    for Class in list(Annotations.get_classes()):
        colors[Class] = tuple([int(np.random.randint(150, 255,1)[0]) for i in range(3)])

    ## Extract all the annotations and draw them in the image
    plt.figure(figsize=frsize)
    for obj in Annotations.objects:
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)

        I = cv2.rectangle(I, (xmin, ymin), (xmax, ymax), colors[class_name], 10)
        I = cv2.putText(I, class_name, (xmin, ymin - 10), 0, 1, colors[class_name], 4)
    if show:
        plt.imshow(I)
    return I
