import numpy as np
import cv2


def mask_pixels(Image: np.array, mask: np.array):
    """
    This function extracts the pixels inside the mask and returns a 3D image with a single row.
    It also returns the indices belonging to the extracted pixels for reconstruction
    :param: Image : The RGB image in format (-,-,Channels)
    :param: mask  : The binary image with 0,1 in format (-,-)
    """

    # Extract shape of the image
    (M, N, C) = Image.shape
    mask = mask >= 1
    mask = mask.reshape(-1)
    # Reshape each channel of the image
    Image = Image.reshape((-1, C)).transpose(1, 0)
    # Extract only the pixels from the mask
    Pixels_inMask = []
    # Compute indices that belong to the mask
    mask_ind = [i for i in range(mask.shape[0]) if mask[i]]
    for channel in Image:
        # Extract (position of pixel in the mask, value of the pixel)
        Pixels_inMask.append([channel[index] for index in mask_ind])
    Pixels_inMask = np.array(Pixels_inMask).transpose(1, 0)
    Pixels_inMask = np.expand_dims(Pixels_inMask, axis=1)
    return Pixels_inMask, np.array(mask_ind)


def unmask_pixels(Original_image: np.array, I: np.array, index: np.array):
    """
    This function extracts the pixels inside the mask and returns a 3D image with a single row.
    It also returns the indices belonging to the extracted pixels for reconstruction
    :param: Original_image : The RGB image in format (-,-,Channels) used before processing
    :param: I  : The single row array with three channels
    :param: index : The indexes of the mask for reconstruction
    """
    # Extract shape of the original image for reconstruction
    (M, N, C) = Original_image.shape
    # We will store the new values in the same image and reshape the image
    Image = 0 * Original_image.reshape((-1, C)).transpose(1, 0)
    # Reshape I to iterate over each channel
    I = np.squeeze(I.transpose(2, 0, 1), axis=-1)
    for channel in range(C):
        for (ii, i) in enumerate(index):
            Image[channel][i] = I[channel][ii]
    Image = Image.reshape(C, M, N).transpose(1, 2, 0)
    return Image


def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    Source: https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    Source: https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def mask_image(image, mask):
    """
    This method overlays a mask on top of an image
    :param image image: The color image that you want to mask
    :param image mask: The mask
    :return: masked_image
    :rtype: image (array)
    Source: https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/
    """

    # Split the colors into the different color channels
    blue_color, green_color, red_color = cv2.split(image)

    # Resize the mask to be the same size as the source image
    resized_mask = cv2.resize(
        mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)

    # Normalize the mask
    normalized_resized_mask = resized_mask / float(255)

    # Scale the color values
    blue_color = blue_color * normalized_resized_mask
    blue_color = blue_color.astype(int)
    green_color = green_color * normalized_resized_mask
    green_color = green_color.astype(int)
    red_color = red_color * normalized_resized_mask
    red_color = red_color.astype(int)

    # Put the image back together again
    merged_image = cv2.merge([blue_color, green_color, red_color])
    masked_image = cv2.convertScaleAbs(merged_image)
    return masked_image


def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    Source: https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)

    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0, 256])
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0, 256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0, 256])

    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)

    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)

    # Put the image back together
    image_after_matching = cv2.merge([
        blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching


def adjust_gamma(image, gamma=1.0):
    """ This function adjusts the image from a given image
    :param: image : image to apply correction
    :param: gamma : value of gamma to implement
    Source: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/ """
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def compute_mask(I: np.array):
    """ This function computes the mask of an image
    :param: I : The image to compute the mask from"""
    blur = cv2.GaussianBlur(I, (5, 5), 0)
    for i in range(50):
        blur = cv2.GaussianBlur(blur, (11, 11), 0)
    ret3, th3 = cv2.threshold(blur[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = np.array(th3 > ret3, dtype=np.uint8)

    kernel = np.ones((31, 31), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
