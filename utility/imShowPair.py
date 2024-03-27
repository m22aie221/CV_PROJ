import numpy as np
import cv2

def imShowPair(im1, im2):
    """
    Display a pair of images side by side.

    Parameters:
        im1: numpy array
            First input image.
        im2: numpy array
            Second input image.

    Returns:
        fusion: numpy array
            Fused image of im1 and im2, with im1 in green channel and im2 in red channel.
    """
    # Normalize the images
    im1_norm = im1 / np.max(im1)
    im2_norm = im2 / np.max(im2)

    # Create a blank image to hold the fused result
    fusion = np.zeros((im1.shape[0], im1.shape[1], 3), dtype=np.float32)

    # Assign the normalized images to the respective channels
    fusion[:, :, 1] = im1_norm  # Green channel for im1
    fusion[:, :, 0] = im2_norm  # Red channel for im2
    fusion[:, :, 2] = fusion[:, :, 0]  # Blue channel (same as red channel)

    # Display the fused image
    cv2.imshow('Fused Image', fusion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return fusion
