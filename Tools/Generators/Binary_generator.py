import pandas as pd
import numpy as np
import cv2
import os
import random


class Generator_from_DataFrame:
    """ Generator_from_DataFrame(DataFrame, X="File", target_size=(224, 244), batch_size=32, pre_processing_Image=lambda x: x,
                 pre_processing_ROI=lambda x: x, read_function=cv2.imread)
        - Reads a DataFrame and creates a Generator that yields the ROI and a random patch

        :param DataFrame:      Pandas DataFrame that contains the path to the files.
        :param X:              Name of the column that contains the original data.
        :param target_size:    Size of the images to generate.
        :param batch_size:     Number of images in the batch.
        :param pre_processing_Image:     Function to preprocess image.
        :param pre_processing_ROI:     Function to preprocess the extracted ROI.
        :param read_function:     Function to read the image, default is cv2 function.

    """

    def __init__(self, DataFrame, X="File", target_size=(224, 244), batch_size=32, pre_processing_Image=lambda x: x,
                 pre_processing_ROI=lambda x: x, read_function=cv2.imread):
        # Save the name of the column with the data
        self.X = X
        # Save the function to read the images
        self.read_f = read_function
        # Save the function for pre-processing images
        self.pre_proc = pre_processing_Image
        # Save the function for pre-processing the ROI
        self.pre_proc_ROI = pre_processing_ROI
        # Save the size of the output image
        self.target_size = target_size
        # Save the batch size
        self.batch_size = (batch_size//2)*2

        # Check if the files exists for X
        CheckedFilesX = [os.path.isfile(x) for x in DataFrame[X]]
        print(f"{sum(CheckedFilesX)} files found out of {len(DataFrame)} in the DataFrame for X")

        # If not data found, raise an exception
        if sum(CheckedFilesX) == 0:
            raise Exception(f"Sorry, no data in {X} to process")

        # If all the files in the DataFrame are checked then there are no missing files
        # if not, then there could be missing files from X and an exception is trown
        if sum(CheckedFilesX) == len(DataFrame):
            self.data = DataFrame[CheckedFilesX].copy()
        else:
            raise Exception(f"Sorry, there are missing files in {X}")

        # Create a random order for the data and shuffle it
        self.order = [x for x in range(len(self.data))]
        self.order = random.sample(self.order, len(self.data))

        # User notification
        print(f"A generator object has been created with {sum(CheckedFilesX)} elements of batch_size = {self.batch_size}")

    def __next__(self):
        """
        Yields the next training batch.
        """
        # If the order is empty then generate a new order
        if len(self.order) <= self.batch_size:
            self.order = [x for x in range(len(self.data))]
            self.order = random.sample(self.order, len(self.data))
        # Extract a random sample from the order and delete the index                                   -- EPOCH START
        index = random.sample(self.order,
                              self.batch_size // 2)  ## Half of the batch is part of labeled as 1 and the other
        ## half as zero
        [self.order.remove(element) for element in index]

        ## Empty arrays
        X_train = []
        y_train = []

        ## Read the images
        for i in index:
            # Read the image
            I = self.read_f(self.data[self.X].iloc[i])
            # Apply a transformation
            I = self.pre_proc(I)
            # Get the shape to perform operations
            M, N, C= I.shape
            ## Get the points of the patch
            x = self.data['y'].iloc[i]
            y = self.data['x'].iloc[i]
            x_width = self.data['y_width'].iloc[i]
            y_width = self.data['x_width'].iloc[i]

            ## Get the points of the negative patch
            xN = int(np.random.randint(0, M, 1))
            yN = int(np.random.randint(0, N, 1))

            ## Compute intersection over union to see that the negative patch doesn't belong to the annotation
            BoxGT = [x,y,x+x_width,y+y_width]
            BoxN  = [xN,yN,xN+x_width,yN+y_width]

            while bb_intersection_over_union(BoxGT,BoxN) != 0:
                ## Get the points of the negative patch
                xN = int(np.random.randint(0, M-x_width-1, 1))
                yN = int(np.random.randint(0, N-y_width-1, 1))
                BoxN = [xN, yN, xN + x_width, yN + y_width]
            ## Add image to batch
            X_train.append(cv2.resize(self.pre_proc_ROI(I[x:x+x_width,y:y+y_width,:]),self.target_size))
            ## Add negative image to batch
            X_train.append(cv2.resize(self.pre_proc_ROI(I[xN:xN + x_width, yN:yN + y_width, :]),self.target_size))

            ## Add output class
            y_train.append([1.])
            y_train.append([0.])


        # Yield the nex batch
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        return X_train, y_train

    def __iter__(self):
        return self


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
