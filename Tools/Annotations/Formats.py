import xml.etree.ElementTree as ET
import os
from copy import deepcopy


class VOC_format_V2:
    def __init__(self, xml_file: str):
        # Filename
        self.name = xml_file
        # Extract the root
        self.root = ET.parse(xml_file).getroot()
        self.update_dependencies()

    def saveXml(self, New_path=""):
        """This function saves the changes and overwrites the xml file or writes a new file"""
        if New_path == "":
            # Saves the file in the same directory with the name in file
            with open(self.name, "wb") as f:
                f.write(ET.tostring(self.root))
                f.close()
        else:
            # Saves the file in a new directory
            new_name = self.filename.text.split('.')[-2] + ".xml"
            with open(os.path.join(New_path, new_name), "wb") as f:
                f.write(ET.tostring(self.root))
                f.close()

    # Read all the information from the file
    def update_dependencies(self):
        """This function updates all the variables of the object"""
        # Extract folder
        self.folder = self.root.find('folder')
        # Extract filename
        self.filename = self.root.find('filename')
        # Extract source
        self.source = self.root.find('source')
        # Extract size
        self.size = self.root.find('size')
        self.width = self.size.find('width')
        self.height = self.size.find('height')
        # Extract objects
        self.objects = self.root.findall('object')
        # Extract segmented
        self.segmented = self.root.findall('segmented')

    # Compute classes
    def get_classes(self):
        return set([obj.find('name').text for obj in self.root.findall('object')])

    # Get the text of the attribute
    def get_attribute(self, attrib_name):
        try:
            R = self.root.find(attrib_name).text
            return R
        except:
            raise NameError(f"{attrib_name} not found in root")

    # Change the value of the attribute
    def change_attribute(self, attrib_name, new_value):
        if type(new_value) == str:
            self.root.find(attrib_name).text = new_value
        else:
            self.root.find(attrib_name).text = str(new_value)
        self.update_dependencies()

    # Remove objects
    def remove_object(self, obj):
        self.root.remove(obj)
        self.update_dependencies()

    # Add objects
    def add_object(self, obj):
        self.root.append(deepcopy(obj))
        self.update_dependencies()

    # Get Object attributes
    def get_object(self, n_object):
        obj = self.objects[n_object]
        return_value = {'name':obj.find('name').text,
                        'xmin':int(obj.find('bndbox').find('xmin').text),
                        'ymin':int(obj.find('bndbox').find('ymin').text),
                        'xmax':int(obj.find('bndbox').find('xmax').text),
                        'ymax':int(obj.find('bndbox').find('ymax').text)}
        return return_value


    ## Print feature
    def __str__(self):
        Object =f"--- Annotation --- \n" + \
                f"|-- Folder:{self.get_attribute('folder')}\n" + \
                f"|-- Filename: {self.get_attribute('filename')}\n" + \
                "|-- Size: \n" + \
                f"\t|- width: {self.size.find('width').text}\n" + \
                f"\t|- width: {self.size.find('height').text}\n" + \
                "|-- Objects: \n"
        def print_objects(objects):
            O = ""
            for obj in objects:
                O = O + "\n"+ f"    \t|- name: {obj.find('name').text}\n"+ \
                              f"    \t|- bndbox\n" + \
                              f"    \t\t |- xmin: {obj.find('bndbox').find('xmin').text}\n" + \
                              f"    \t\t |- xmax: {obj.find('bndbox').find('xmax').text}\n" + \
                              f"    \t\t |- ymin: {obj.find('bndbox').find('ymin').text}\n" + \
                              f"    \t\t |- ymax: {obj.find('bndbox').find('ymax').text}\n" + \
                              f"    \t----------------------------------------------------"
            return O
        return Object + print_objects(self.objects)
