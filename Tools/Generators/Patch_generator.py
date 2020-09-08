import pandas as pd
import numpy as np
import cv2
import os
import random


class Generator_from_DataFrame:
    """ Generator_from_DataFrame(DataFrame,patch_size = [32,32], X = "File", y = "GT")
        - Reads a DataFrame and creates a Generator that yields the same image divided in patches. The patch must be
        smaller than the whole image. The DataFrame must contain the original File and the Ground Truth.

        :param DataFrame:      Pandas DataFrame that contains the path to the files.
        :param patch_size:     Size of the patches to generate.
        :param X:              Name of the column that contains the original data.
        :param y:              Name of the column that contains the Ground Truth
        :param n_patches:      Number of patches to extract from the image.
    """

    def __init__(self, DataFrame, patch_size=[32, 32], X="File", y="GT", n_patches=100):

        # Get the name of the columns with the data
        self.X = X
        self.y = y

        # Save the number of patches per image
        self.n_patches = n_patches

        # Save the size of the patches
        self.patch_size = patch_size

        # Check if the files exists for X
        CheckedFilesX = [os.path.isfile(x) for x in DataFrame[X]]
        print(f"{sum(CheckedFilesX)} files found out of {len(DataFrame)} in the DataFrame for X")

        # If not data found, raise an exception
        if sum(CheckedFilesX) == 0:
            raise Exception(f"Sorry, no data in {X} to process")

        # Check if the files exists for y
        CheckedFilesy = [os.path.isfile(x) for x in DataFrame[y]]
        print(f"{sum(CheckedFilesy)} files found out of {len(DataFrame)} in the DataFrame for y")

        # If no data found in y, raise an exception
        if sum(CheckedFilesy) == 0:
            raise Exception(f"Sorry, no data in {y} to process")

        # If both Columns are equal, then save the DataFrame with the existing files
        # if not, then there could be missing files from X that correspond to y
        if CheckedFilesX == CheckedFilesy:
            self.data = DataFrame[CheckedFilesX].copy()
        else:
            raise Exception(f"Sorry, there are missing files in {X} --> {y}")

        # Create a random order for the data and shuffle it
        self.order = [x for x in range(len(self.data))]
        self.order = random.sample(self.order, len(self.data))

        # User notification
        print(f"A generator object has been created with {n_patches} per image")

    def __next__(self):
        """
        Yields the next training batch.
        """
        # If the order is empty then generate a new order
        if len(self.order) == 0:
            self.order = [x for x in range(len(self.data))]
            self.order = random.sample(self.order, len(self.data))
        # Extract a random sample from the order and delete the index                                   -- EPOCH START
        image = random.sample(self.order, 1)
        self.order.remove(image[0])

        # Given the new order read the images
        #                                                                         -- STEP START
        # Read the corresponding frame
        Frame = self.data.iloc[image].copy()
        # Read X and Y
        X = cv2.imread(Frame[self.X].values[0])  # The image is read as BGR
        X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)  # Color space correction is performed
        X = cv2.normalize(X, None, alpha=0,  # Normalize image to fit from 0 to 1
                          beta=1,
                          norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F)
        (M, N, C) = X.shape  # Get the size of the image
        
        y = cv2.imread(Frame[self.y].values[0])  # Resulting image read as BGR
        (My, Ny, Cy) = y.shape  # Get the size of the target image
        if Cy == 3:  # If the target is a three dimension image, the color space is changed
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        y = cv2.normalize(y, None, alpha=0,  # Normalize image to fit from 0 to 1
                          beta=1,
                          norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F)

        # Check if the dimensions of the original and target image are the same
        if M != My or N != Ny:
            raise Exception(f"Sorry, dimensions do not match {M}<-->{My}, {N}<-->{Ny}")

        # Initialize X_train and y_train arrays for the batch
        X_train = []
        y_train = []

        # Create the batch given the number of patches
        for patch in range(self.n_patches):  # -- BATCH START
            """
                For the selection of the patch we will base the selection in a normal distribution to chose the 
                origin (x_coor,y_coor) with the mean as half the length of the image and the deviation as 1/7th
                of the length, then the total size of the image is given by 'self.patch_size'
            """
            # The original coordinates are computed, while is executed until the coordinates are valid for the patch
            x_coor, y_coor = (0, 0)
            while (x_coor <= 0) or (x_coor + self.patch_size[0] > M) or (y_coor <= 0) or (
                    y_coor + self.patch_size[1] > N):
                x_coor, y_coor = (int(np.random.normal(M / 2, M / 4)), int(np.random.normal(N / 2, N / 4)))

            # Append the patch to the images
            X_train.append(X[x_coor:x_coor + self.patch_size[0],
                           y_coor:y_coor + self.patch_size[1], :])

            # Append the patch of target image
            y_train.append(y[x_coor:x_coor + self.patch_size[0],
                           y_coor:y_coor + self.patch_size[1], :])

        # Make sure they're numpy arrays (as opposed to lists)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Yield the nex batch
        return X_train, y_train

    def __iter__(self):
        return self


########################################################################################################################
########################################################################################################################
########################################################################################################################
class Image2Generator:
    """ Image_to_Generator(File,patch_size = [32,32],batch_size = 100)
        - Reads a DataFrame and creates a Generator that yields the same image divided in patches. The patch must be
        smaller than the whole image. The DataFrame must contain the original File.

        :param File:           Numpy array containing the image.
        :param patch_size:     Size of the patches to generate.
        :param batch_size:     Number of patches to generate per batch
    """

    def __init__(self, File, patch_size=[32, 32], batch_size=100):

        # Save the size of the patches
        self.patch_size = patch_size

        # Save the batch size
        self.batch_size = batch_size

        # If not data found, raise an exception
        #if not os.path.isfile(File):
        #    raise Exception(f"Sorry, no data in {File} to process")

        # If found, read image
        #X = cv2.imread(File)  # The image is read as BGR
        X = cv2.cvtColor(File, cv2.COLOR_BGR2RGB)  # Color space correction is performed
        X = cv2.normalize(X, None, alpha=0,  # Normalize image to fit from 0 to 1
                          beta=1,
                          norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F)
        (self.M, self.N, self.C) = X.shape  # Get the size of the image

        # Compute the number of patches
        n_patches_x = int(np.ceil(self.M / self.patch_size[0]))
        n_patches_y = int(np.ceil(self.N / self.patch_size[1]))

        # Create a blank image to store the padded image
        self.I = np.zeros([n_patches_x * self.patch_size[0], n_patches_y * self.patch_size[1], self.C])

        # Add the original image
        self.I[0:self.M, 0:self.N, :] = X

        Patches = []

        # Create the patches for the image
        for X_coordinate in range(0, (n_patches_x - 1) * self.patch_size[0], self.patch_size[0]):  # -- BATCH START
            for Y_coordinate in range(0, (n_patches_y - 1) * self.patch_size[1], self.patch_size[1]):
                Patches.append(self.I[X_coordinate:X_coordinate + self.patch_size[0],
                               Y_coordinate:Y_coordinate + self.patch_size[1], :])

        # Store the patches for later retrieval
        self.patches = np.array(Patches)
        self.n_patches_x = n_patches_x
        self.n_patches_y = n_patches_y
        self.total_patches = n_patches_y * n_patches_x + (n_patches_y * n_patches_x)%batch_size

        self.index = 0
        # User notification
        print(f"A generator object containing all the image has been created")

    def __next__(self):
        """
        Yields the next training batch.
        """
        # If there are not enough patches to complete the batch fill the remaining with zeros, else return the batch
        if self.index + self.batch_size > self.n_patches_x * self.n_patches_y:
            fill = np.zeros([self.patch_size[0], self.patch_size[1], self.C])
            fill = np.array([fill for i in range(self.index + self.batch_size - self.n_patches_x * self.n_patches_y)])
            return np.concatenate((self.patches[self.index:], fill))
        else:
            self.index = self.index + self.batch_size
            return self.patches[self.index - self.batch_size: self.index]

    def __iter__(self):
        return self


def Generator2Image(model,generator):
    """
    Takes a Sequential model and computes its output on the batches of a generator created with Image2Generator
    :param model:       Sequential model.
    :param generator:   Generator created with Image2Generator
    :return: (Original,New_image)
    """
    # Compute the first batch to create the variable
    Output = model.predict_on_batch(next(generator))
    # Compute the remaining batches
    for i in range(int(generator.total_patches / generator.batch_size) -1):
        Output = np.concatenate((Output,model.predict_on_batch(next(generator))))
    # Extract a template image to store the patches
    I = 0*generator.I
    # Replace the patch in the image
    i = 0
    for X_coordinate in range(0, (generator.n_patches_x - 1) * generator.patch_size[0], generator.patch_size[0]):  # -- BATCH START
        for Y_coordinate in range(0, (generator.n_patches_y -1) * generator.patch_size[1], generator.patch_size[1]):
            I[X_coordinate:X_coordinate + generator.patch_size[0],
              Y_coordinate:Y_coordinate + generator.patch_size[1],0] = Output[i,:,:,0]
            i = i+1
    return I[0:generator.M,0:generator.N,0]

