import xml.etree.ElementTree as ET
import os


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
        return set([obj.find('name').text for obj in self.root.findall('objects')])

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

    # Remove objects
    def remove_object(self, obj):
        self.root.remove(obj)
        self.update_dependencies()

    # Add objects
    def add_object(self, obj):
        self.root.append(obj)
        self.update_dependencies()

