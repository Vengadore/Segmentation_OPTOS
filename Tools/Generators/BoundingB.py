import xml.etree.ElementTree as ET
import copy
import numpy as np
from ..Generators.Binary_generator import bb_intersection_over_union as bbIoU
import os


class VOC_format:
    def __init__(self, xml_file: str):
        # Filename
        self.name = xml_file
        # Extract the root
        self.root = ET.parse(xml_file).getroot()
        self.update_dependencies()

    def createBackgroundBB(self, n=1):
        # Extract all the bounding boxes
        bboxes = [obj[-1] for obj in self.objects if obj[-1].tag == "bndbox"]
        # The means of the bounding boxes will be the height and width of the background bounding boxes
        width_mean = sum([float(x[2].text) - float(x[0].text) for x in bboxes]) / len(
            bboxes)  # Extract all the widths and compute mean
        height_mean = sum([float(x[3].text) - float(x[1].text) for x in bboxes]) / len(
            bboxes)  # Extract all the heights and compute mean
        # Loop over the number of bounding boxes to create
        for i in range(n):
            IoU = 1.0
            counter = 0
            while IoU > 0.01:  # While the Intersection over Union is too big we will create new coordinates
                xmin = float(np.random.randint(0, self.width - width_mean, 1)[0])
                xmax = xmin + width_mean
                ymin = float(np.random.randint(0, self.height - height_mean, 1)[0])
                ymax = ymin + height_mean
                # Compute all the IoU with the other bounding boxes
                IoU = sum([bbIoU([xmin, ymin, xmax, ymax],
                                 [float(x[0].text), float(x[1].text), float(x[2].text), float(x[3].text)]) for x in
                           bboxes])
                counter += 1
                if counter > 99:
                    # If after 100 tries no bounding box is created then it is highly possible that there is no space left for a BackgroundBB
                    print("After {} tries no Background Bounding Box was created".format(counter))
                    print("{} BackgroundBB created".format(i + 1))
                    self.saveXml()
                    return None
            # Once the IoU is small enough then we can add that bounding box as an object to root
            obj = copy.deepcopy(self.objects[-1])
            obj[0].text = "Background"  # Name
            obj[1].text = "Unspecified"  # Pose
            obj[2].text = "0"  # Truncated
            obj[3].text = "0"  # Difficult
            # Add coordinates to object
            obj[4][0].text = str(xmin).split('.')[0]  # Keep only the integer part
            obj[4][1].text = str(ymin).split('.')[0]
            obj[4][2].text = str(xmax).split('.')[0]
            obj[4][3].text = str(ymax).split('.')[0]

            # Append new object to root
            self.root.append(obj)
            # Compute objects and bounding boxes again for new IoU
            self.update_dependencies()
            bboxes = [obj[-1] for obj in self.objects if obj[-1].tag == "bndbox"]
        print("{} BackgroundBB created".format(i + 1))
        self.saveXml()

    def saveXml(self,New_path = ""):
        """This function saves the changes and overwrites the xml file or writes a new file"""
        if New_path == "":
            # Saves the file in the same directory with the name in file
            splited = self.name.split(os.path.sep)
            if splited[0] == ".":
                full_path = ""
            else:
                full_path = os.path.sep
            for i in splited[:-1]:
                full_path = os.path.join(full_path, i)
            new_name = os.path.join(full_path, self.filename.text.split(".")[-2] + ".xml")
            with open(new_name, "wb") as f:
                f.write(ET.tostring(self.root))
                f.close()
        else:
            # Saves the file in a new directory
            new_name = self.filename.text.split('.')[-2]+".xml"
            with open(os.path.join(New_path,new_name), "wb") as f:
                f.write(ET.tostring(self.root))
                f.close()
        else:
            # Saves the file in a new directory
            new_name = self.filename.text.split('.')[-2]+".xml"
            with open(os.path.join(New_path,new_name), "wb") as f:
                f.write(ET.tostring(self.root))
                f.close()

    def update_dependencies(self):
        """This function updates all the variables of the object"""
        # Extract folder
        self.folder = [x for x in self.root if x.tag == "folder"][0]
        # Extract filename
        self.filename = [x for x in self.root if x.tag == "filename"][0]
        # Extract source
        self.source = [x for x in self.root if x.tag == "source"][0]
        # Extract size
        self.size = [x for x in self.root if x.tag == "size"][0]
        self.width = int([x for x in self.size if x.tag == "width"][0].text)
        self.height = int([x for x in self.size if x.tag == "height"][0].text)
        # Extract objects
        self.objects = [x for x in self.root if x.tag == "object"]
        # Extract segmented
        self.segmented = [x for x in self.root if x.tag == "segmented"]
        # Compute classes
        self.classes = set([x[0].text for x in self.objects])
