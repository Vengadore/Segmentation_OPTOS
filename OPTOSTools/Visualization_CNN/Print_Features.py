import cv2
from tensorflow.keras.models import Model


class Model_CNN:
    """ Model_CNN(model)

        - Reads a CNN model and looks in the name of the layers for "conv", if found it is saved as an index for extracting feature maps.

        model:  CNN model to extract feature maps from.

    """
    def __init__(self,model):
        # Create a CNN Model
        self.model = model
        # Select the layers that have a convolutional layer
        self.conv_index = [ind for (ind,layer) in enumerate(model.layers) if "conv" in layer.name]
        # Feature map shapes
        self.conv_shapes = [(ind,model.layers[ind].name,model.layers[ind].output.shape) for ind in self.conv_index]
        outputs = [self.model.layers[i].output for i in self.conv_index]
        self.model = Model(inputs=self.model.inputs, outputs = outputs)
        # Extract the weights of the kernels in the convolutional layers
        self.conv_weights = [(ind,model.layers[ind].name,model.layers[ind].get_weights()) for ind in self.conv_index]
        #self.model.summary()
        print(f"Input shape of visualization model {model.layers[0].output.shape}")

    def feature_map(self,image):
        """
            Computes the Feature Maps given an image, the output is a list of the various convolutional layers
        """
        return self.model.predict(image)


class ImageT:
    """ ImageT(Reescale = False, Resize = False)

        - To create transformations between colors spaces

        Reescale:   Reescales image to 0 and 1 dividing by 255
        Resize:     Resizes the image to a given size by a tuple

    """

    def __init__(self,Reescale = False, Resize = False):
        self.R = Reescale
        self.size = Resize

        ""

    def BGR2RGB(self,image):
        """

        :param image:
        :return:
        """
        image = cv2.cvtColor(image, 4)
        # If reescale parameter is true the image values are divided by 255 to fit values between 0 and 1
        if self.R:
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # If Resize is a tuple then the image is resized
        if type((1,1)) == type(self.size):
            image = cv2.resize(image,self.size)
        return image

    def RGB2BGR(self,image):
        """

        :param image:
        :return:
        """

        image = cv2.cvtColor(image, 4)
        # If reescale parameter is true the image values are divided by 255 to fit values between 0 and 1
        if self.R:
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # If Resize is a tuple then the image is resized
        if type((1,1)) == type(self.size):
            image = cv2.resize(image, self.size)
        return image