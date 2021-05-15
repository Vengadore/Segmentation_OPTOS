import copy
import os

import cv2
import imutils
import numpy as np

from ..Annotations.Formats import VOC_format_V2


def augment(img_data: str, image_path = "" ,random_rot=False, horizontal_flips=False, vertical_flips=False, augment=False):
    """ This function takes three parameters to define if data augmentation will be performed in the image given a
    set of parameters. It has no effect if {augmented} is set to False

        :param img_data Path to VOC annotation
        :param image_path Path to look for the image
        :param random_rot Performs random rotation from -15 to 15 degrees

        Data augmentation is performed in a way that the transformation is done only in a 90 degree multiple.
        A hard copy of the DataFrame is made and the coordinates are changed.

        NOTE: an implementation of (-x to x) degrees must be done.

        At the end the function returns the loaded image and the modified DataFrame.
    """

    # Load VOC annotation
    Data = VOC_format_V2(img_data)

    # Copy original annotation to make changes
    img_data_aug = copy.deepcopy(Data)

    # Extract the correct path to the image
    if image_path == "":
        filepath = os.path.join(Data.folder.text,Data.filename.text)
    else:
        filepath = os.path.join(image_path,Data.filename.text)
        # Change folder
        img_data_aug.folder.text = image_path

    # Correction to be read as RGB
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if augment:
        rows, cols = img.shape[:2]
        Applied_aug = "Augmented"

        # Random horizontal rotation
        if horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            Applied_aug += "H"
            for obj in img_data_aug.objects:
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)
                x2 = float(bbox.find('xmax').text)
                obj.find('bndbox').find('xmax').text = str(cols - x1)
                obj.find('bndbox').find('xmin').text = str(cols - x2)
        # Random vertical rotation
        if vertical_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            Applied_aug += "V"
            for obj in img_data_aug.objects:
                bbox = obj.find('bndbox')
                y1 = float(bbox.find('ymin').text)
                y2 = float(bbox.find('ymax').text)
                obj.find('bndbox').find('ymax').text = str(rows - y1)
                obj.find('bndbox').find('ymin').text = str(rows - y2)
        # If rotation allowed
        if random_rot:
            ## Rotate the image first from -15 to 15 degrees, limit this so the bounding box doesn't have a big area
            angle = np.random.randint(-15, 15, 1)[0]
            img = imutils.rotate(img.copy(), angle)
            if angle < 0:
                Applied_aug += "R_" + str(np.abs(angle))
            else:
                Applied_aug += "R" + str(angle)
            for obj in img_data_aug.objects:
                if angle == 0:
                    pass
                # Extract coordinates of Bounding Box
                bbox = obj.find('bndbox')
                bb = {}
                bb['x1'] = float(bbox.find('xmin').text)
                bb['x2'] = float(bbox.find('xmax').text)
                bb['y1'] = float(bbox.find('ymin').text)
                bb['y2'] = float(bbox.find('ymax').text)
                # Rotate bouding box
                new_bbox = bb_rot(angle,img, bb)
                # Overwrite new bounding box
                obj.find('bndbox').find('xmin').text = str(new_bbox['x1'])
                obj.find('bndbox').find('xmax').text = str(new_bbox['x2'])
                obj.find('bndbox').find('ymin').text = str(new_bbox['y1'])
                obj.find('bndbox').find('ymax').text = str(new_bbox['y2'])

        # Change name with data augmented values
        path = img_data_aug.get_attribute('filename')
        s = path.split('.')
        path = path[:-len(s[-1]) - 1] + Applied_aug + "." + s[-1]  # Append the applied augmentation before the extension
        img_data_aug.change_attribute('filename',path)

    img_data_aug.root.find('size').find('width').text = str(img.shape[0])
    img_data_aug.root.find('size').find('height').text = str(img.shape[1])

    # Returns the modified structure and the modified image
    return img_data_aug, img


def bb_rot(angle, I, bb):
    """
    bb_rot rotates the coordinates of a bounding box with coordinates (x1,y1),(x2,y2)
    :param angle  Is the angle in which the image was rotated
    :param I      Is the original image
    :param bb     Is the old bounding box with the coordinates
    """
    # Matrix of rotation for the points
    rad = np.deg2rad(angle)
    mat = [[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]

    # Shape of the image to make corrections in the coordinates
    M, N, C = I.shape

    """Due to the deformation for the rotation the new bounding box must be computed out
       of the four points of the original bounding box. In a clockwise"""
    ## 1
    P1 = (np.dot(mat, np.array([[bb['x1'] - N // 2], [-bb['y1'] + M // 2]])))
    P1 = np.array([int(P1[0] + N // 2), int(-P1[1] + M // 2)])
    ## 2
    P2 = (np.dot(mat, np.array([[bb['x2'] - N // 2], [-bb['y1'] + M // 2]])))
    P2 = np.array([int(P2[0] + N // 2), int(-P2[1] + M // 2)])
    ## 3
    P3 = (np.dot(mat, np.array([[bb['x2'] - N // 2], [-bb['y2'] + M // 2]])))
    P3 = np.array([int(P3[0] + N // 2), int(-P3[1] + M // 2)])
    ## 4
    P4 = (np.dot(mat, np.array([[bb['x1'] - N // 2], [-bb['y2'] + M // 2]])))
    P4 = np.array([int(P4[0] + N // 2), int(-P4[1] + M // 2)])

    ## New bounding box
    new_bb = {}
    ## Look for the smallest and greates values
    new_bb['x1'] = np.min([P1[0], P2[0], P3[0], P4[0]])
    new_bb['x2'] = np.max([P1[0], P2[0], P3[0], P4[0]])
    new_bb['y1'] = np.min([P1[1], P2[1], P3[1], P4[1]])
    new_bb['y2'] = np.max([P1[1], P2[1], P3[1], P4[1]])

    return new_bb