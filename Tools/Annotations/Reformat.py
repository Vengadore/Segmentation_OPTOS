import cv2
from lxml.etree import Element, SubElement, tostring
import pandas as pd
import os


def csv2xml(data: pd.DataFrame):
    """
    csv2xml iterates over a dataframe to create VOC annotations from a csv with the following format

    ###############################################
    ## Filename %% x1 %% y1 %% x2 %% y2 %% class ##
    ###############################################

    :param data: Dataframe containing the data
    :param PathToAppend: Path to append to the path
    :return:
    """
    os.mkdir("./Annotations")
    Files = set(data['Filename'])  ## Extract unique Filenames
    Files = [x for x in Files]  ## Convert it to a list to iterate

    Classes = set(data['class'])
    Classes = [Class for Class in Classes]
    Code = {}
    for i in range(len(Classes)):
        Code[Classes[i]] = i

    for Image in Files:  ## For every Image (DataFrame) there are several bounding boxes
        path = os.path.split(Image)

        node_root = Element('annotation')  ## Base node
        node_folder = SubElement(node_root, 'folder')  ## Folder node
        node_folder.text = path[0]
        node_filename = SubElement(node_root, 'filename')  ## Filename node
        node_filename.text = path[-1]
        node_source = SubElement(node_root, 'source')  ## Source node
        node_sourceDatabase = SubElement(node_source, 'database')  ## Source database
        node_sourceDatabase.text = "Neovessels OPTOS"
        node_sourceAnnotation = SubElement(node_source, 'annotation')  ## Source annotation
        node_sourceAnnotation.text = "PASCAL VOC2007"
        node_sourceImage = SubElement(node_source, 'image')  ## Source image
        node_sourceImage.text = "APEC"

        ## Load image and check for size
        I = cv2.imread(os.path.join(path[0], path[-1]))
        [M, N, C] = I.shape

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
        with open(os.path.join("./Annotations", name + ".xml"), 'wb') as f:
            f.write(s)