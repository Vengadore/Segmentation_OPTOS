import cv2
from lxml.etree import Element, SubElement, tostring
import pandas as pd
import os
import json
from shutil import copyfile
from sklearn.model_selection import train_test_split



def csv2xml(data: pd.DataFrame):
    """
    csv2xml iterates over a dataframe to create VOC annotations from a csv with the following format

    ############################################### \\
    ## Filename %% x1 %% y1 %% x2 %% y2 %% class ## \\
    ############################################### \\

    :param data: Dataframe containing the data
    :param PathToAppend: Path to append to the path
    :return:
    """
    os.mkdir("./VOCdevkit")

    os.mkdir("./VOCdevkit/VOC2012")
    os.mkdir("./VOCdevkit/VOC2012/Annotations")
    os.mkdir("./VOCdevkit/VOC2012/ImageSets")
    os.mkdir("./VOCdevkit/VOC2012/ImageSets/Main")
    os.mkdir("./VOCdevkit/VOC2012/JPEGImages")
    Files = set(data['Filename'])  ## Extract unique Filenames
    Files = [x for x in Files]  ## Convert it to a list to iterate

    Classes = set(data['class'])
    Classes = [Class for Class in Classes]
    ImageSet = {}

    for Image in Files:  ## For every Image (DataFrame) there are several bounding boxes
        path = os.path.split(Image)

        node_root = Element('annotation')  ## Base node
        node_folder = SubElement(node_root, 'folder')  ## Folder node
        node_folder.text = "./VOCdevkit/VOC2012/JPEGImages"
        node_filename = SubElement(node_root, 'filename')  ## Filename node
        node_filename.text = path[-1]
        node_source = SubElement(node_root, 'source')  ## Source node
        node_sourceDatabase = SubElement(node_source, 'database')  ## Source database
        node_sourceDatabase.text = "Neovessels OPTOS"
        node_sourceAnnotation = SubElement(node_source, 'annotation')  ## Source annotation
        node_sourceAnnotation.text = "PASCAL VOC2012"
        node_sourceImage = SubElement(node_source, 'image')  ## Source image
        node_sourceImage.text = "APEC"

        ## Load image and check for size
        I = cv2.imread(Image)
        [M, N, C] = I.shape
        # Move the image to ImageSets
        copyfile(Image, os.path.join("./VOCdevkit/VOC2012/JPEGImages", path[-1]))

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
            ImageSet[bb['class']].append(path[-1]) ## We append the name of the image given a class

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
        with open(os.path.join("./VOCdevkit/VOC2012/Annotations", name + ".xml"), 'wb') as f:
            f.write(s)

        #Indexes for train,trainval and val
        train = []
        trainval = []
        val = []

        ## Split data for each class
        for Class in Classes:
            names = set(ImageSet[Class])
            names = [n for n in names]

            X_train, X_test = train_test_split(names, test_size=0.2)
            with open(f"./VOCdevkit/VOC2012/ImageSet/Main/{Class}_train.txt","w") as f:
                f.writelines(names[X_train])
            with open(f"./VOCdevkit/VOC2012/ImageSet/Main/{Class}_trainval.txt", "w") as f:
                f.writelines(names[X_test])
            with open(f"./VOCdevkit/VOC2012/ImageSet/Main/{Class}_val.txt", "w") as f:
                f.writelines(names[X_test])

            train = train + names[X_train]
            trainval = trainval + names[X_test]
            val = val + names[X_test]
        # Write train, trainval and val txt
        with open(f"./VOCdevkit/VOC2012/ImageSet/Main/train.txt","w") as f:
            f.writelines(train)
        with open(f"./VOCdevkit/VOC2012/ImageSet/Main/trainval.txt", "w") as f:
            f.writelines(trainval)
        with open(f"./VOCdevkit/VOC2012/ImageSet/Main/val.txt", "w") as f:
            f.writelines(val)

        ## Create label_map_json_path manually

