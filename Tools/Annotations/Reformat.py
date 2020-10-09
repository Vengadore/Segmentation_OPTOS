import cv2
from lxml.etree import Element, SubElement, tostring
import pandas as pd
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import imutils
from ..Generators.BoundingB import VOC_format


def csv2xml(data: pd.DataFrame, PathToAppend: str = "./", DatasetName: str = "", seed=32):
    """
    csv2xml iterates over a dataframe to create VOC annotations from a csv with the following format

    ############################################### \\
    ## Filename %% x1 %% y1 %% x2 %% y2 %% class ## \\
    ############################################### \\

    :param data: Dataframe containing the data
    :param PathToAppend: Path to append to creation of the dataset
    :param DatasetName: Name of the dataset
    :return:

    Creates the following structure in PathToAppend
    ./VOC2012
       |-------> Annotations (xml individual files)
       |-------> ImageSets   (Separation for training, validation and test in txt)
       |-------> JPEGImages  (Images to new folder)
    """

    ## Creation of new directories
    main_directory = os.path.join(PathToAppend, "VOC2012")
    os.mkdir(main_directory)
    os.mkdir(os.path.join(main_directory, "Annotations"))  ## Annotations directory
    os.mkdir(
        os.path.join(main_directory, "ImageSets"))  ## ImageSets directory (Files with Training,Validation and Test)
    os.mkdir(os.path.join(main_directory, "JPEGImages"))  ## Saved images

    ## DataFrame manipulation
    Files = set(data['Filename'])  ## Extract unique Filenames
    Files = [x for x in Files]  ## Convert it to a list to iterate

    Classes = set(data['class'])
    Classes = [Class for Class in Classes]
    ImageSet = {}
    for Class in Classes:
        ImageSet[Class] = []

    for Image in Files:  ## For every Image (DataFrame) there are several bounding boxes
        path = os.path.split(Image)

        node_root = Element('annotation')  ## Base node
        node_folder = SubElement(node_root, 'folder')  ## Folder node
        node_folder.text = main_directory
        node_filename = SubElement(node_root, 'filename')  ## Filename node
        node_filename.text = path[-1]
        node_source = SubElement(node_root, 'source')  ## Source node
        node_sourceDatabase = SubElement(node_source, 'database')  ## Source database
        node_sourceDatabase.text = DatasetName
        node_sourceAnnotation = SubElement(node_source, 'annotation')  ## Source annotation
        node_sourceAnnotation.text = "PASCAL VOC2012"
        node_sourceImage = SubElement(node_source, 'image')  ## Source image
        node_sourceImage.text = "APEC"

        ## Load image and check for size
        I = cv2.imread(Image)
        [M, N, C] = I.shape
        # Move the image to ImageSets
        copyfile(Image, os.path.join(os.path.join(main_directory, "JPEGImages"), path[-1]), follow_symlinks=False)

        node_size = SubElement(node_root, 'size')

        node_sizeWidth = SubElement(node_size, 'width')
        node_sizeWidth.text = str(N)
        node_sizeHeight = SubElement(node_size, 'height')
        node_sizeHeight.text = str(M)
        node_sizeDepth = SubElement(node_size, 'depth')
        node_sizeDepth.text = str(C)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = "0"

        for i, bb in data[data['Filename'] == Image].iterrows():  ## Every Image has one or more bounding boxes
            node_object = SubElement(node_root, 'object')
            node_Oname = SubElement(node_object, 'name')
            node_Opose = SubElement(node_object, 'pose')
            node_Otruncated = SubElement(node_object, 'truncated')
            node_Odifficult = SubElement(node_object, 'difficult')

            ## Append to file for txt
            ImageSet[bb['class']].append(path[-1])  ## We append the name of the image given a class

            node_Oname.text = bb['class']
            node_Opose.text = "Unspecified"
            node_Otruncated.text = "0"
            node_Odifficult.text = "0"

            node_Obndbox = SubElement(node_object, 'bndbox')
            node_OBB_xmin = SubElement(node_Obndbox, 'xmin')
            node_OBB_xmin.text = str(bb.x1)
            node_OBB_ymin = SubElement(node_Obndbox, 'ymin')
            node_OBB_ymin.text = str(bb.y1)
            node_OBB_xmax = SubElement(node_Obndbox, 'xmax')
            node_OBB_xmax.text = str(bb.x2)
            node_OBB_ymax = SubElement(node_Obndbox, 'ymax')
            node_OBB_ymax.text = str(bb.y2)

        ## Final node
        s = tostring(node_root, pretty_print=True)
        name = path[-1].split('.')[0]
        with open(os.path.join(os.path.join(main_directory, "Annotations"), name + ".xml"), 'wb') as f:
            f.write(s)

    # Indexes for train,trainval and val
    train = []
    trainval = []
    val = []

    ## Split data for each class
    for Class in Classes:
        names = set(ImageSet[Class])
        names = [n for n in names]
        X_train, X_test = train_test_split(names, test_size=0.2, random_state=seed)
        # We perform split again in the training set
        X_train, X_validation = train_test_split(X_train, test_size=0.2, random_state=seed)
        X_train = [X.split('.')[0] for X in X_train]
        X_validation = [X.split('.')[0] for X in X_validation]
        X_test = [X.split('.')[0] for X in X_test]
        with open(os.path.join(os.path.join(main_directory, "ImageSets"), f"{Class}_train.txt"), "x") as f:
            for item in X_train:
                f.write("%s\n" % item)
        with open(os.path.join(os.path.join(main_directory, "ImageSets"), f"{Class}_trainval.txt"), "x") as f:
            for item in X_validation:
                f.write("%s\n" % item)
        with open(os.path.join(os.path.join(main_directory, "ImageSets"), f"{Class}_val.txt"), "x") as f:
            for item in X_test:
                f.write("%s\n" % item)

        train = train + X_train
        trainval = trainval + X_validation
        val = val + X_test
    # Write train, trainval and val txt
    with open(os.path.join(os.path.join(main_directory, "ImageSets"), "train.txt"), "x") as f:
        for item in train:
            f.write("%s\n" % item)
    with open(os.path.join(os.path.join(main_directory, "ImageSets"), "trainval.txt"), "x") as f:
        for item in trainval:
            f.write("%s\n" % item)
    with open(os.path.join(os.path.join(main_directory, "ImageSets"), "val.txt"), "x") as f:
        for item in val:
            f.write("%s\n" % item)

    print(""" Create label_map_json_path manually
 item
 {
     name: "cat"
     id: 1
 }
 item
 {
     name: "dog"
     id: 2
 }
 item
 {
     name: "fox"
     id: 3
 }
 item
 {
     name: "squirrel"
     id: 4
 }""")


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
    Data = VOC_format(img_data)

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
    print(filepath)
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
        path = img_data_aug.filename.text
        s = path.split('.')
        path = path[:-len(s[-1]) - 1] + Applied_aug + "." + s[-1]  # Append the applied augmentation before the extension
        img_data_aug.filename.text = path

    img_data_aug.size.find('width').text = str(img.shape[0])
    img_data_aug.size.find('height').text = str(img.shape[1])
    img_data_aug.update_dependencies()

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
