import os
import cv2
import numpy as np
from PIL import Image
from OPTOSTools.Annotations import VOC_format_V2


def compute_activations(Annotation:VOC_format_V2,model,path_images:str,layer_name:str,transformation,device):
    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    # Compute the feature maps
    t = get_image(Annotation, path_images, transformation, device)
    model.eval(layer_name).register_forward_hook(get_activation(layer_name))
    output = model(t)

    act = activation[layer_name].squeeze()




def get_image(VOC_format, path_to_image, transfor, dev):
    """Gets the image and converts it into a tensor
    :param VOC_format: The VOC format with the annotations
    :param path_to_image: Path where the image can be found
    :param transfor: Transformation to apply to the image
    :param dev: Device where the tensor should be load to
    :return: Returns tensor in the image shape"""

    # Load image
    I = Image.open(os.path.join(path_to_image, VOC_format.get_attribute('filename')))
    # Transform image, expand it and send it to the device
    I = transfor(I).unsqueeze(0).to(dev)
    return I

def add_heatmap(heatmap,Image):
    # Get heatmap
    heatmap = cv2.resize(heatmap.numpy(), (Image.shape[1], Image.shape[0]))
    heatmap = np.uint8(cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Add heatmap
    fin = cv2.addWeighted(heatmap, 0.4, Image, 0.7, 0)
    I = Image.fromarray(fin)
    return I