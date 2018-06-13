import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.filters import threshold_otsu
from stl import mesh

IMAGE_PATH = 'E:/Internships/IIITB2018/src/non_blender_trials/images/photosynthesis.PNG'
OUTPUT_PATH = 'E:/Internships/IIITB2018/src/non_blender_trials/stl_files/'
SCALING_FACTOR = 0.9

hue_map = {"purple": 260, "blue": 230, "red": 10, "yellow": 45, "light_green": 75, "dark_green": 130, "light_blue": 190}
gray_intensity = {"red": 0.1, "yellow": 0.3, "light_green": 0.5, "dark_green": 0.6, "light_blue": 0.7, "blue": 0.8,
                  "purple": 0.9}


def convert_from_colored_to_gray(image):
    rows = image.shape[0]
    cols = image.shape[1]
    gray_image = np.zeros((rows, cols)).astype('float32')
    for i in range(0, rows):
        for j in range(0, cols):
            pixel_val = tuple(image[i, j, :].astype('int'))
            gray_image[i][j] = get_gray_value_from_color(pixel_val)
    return gray_image


def get_gray_value_from_color(pixel):
    pixel_hue = pixel[0]
    pixel_saturation = pixel[1]
    gray_value = 0
    for color in hue_map:
        color_hue = hue_map[color]
        if color == "red":
            if pixel_hue in range(color_hue-15, color_hue + 15) and pixel_saturation > 10:
                gray_value = gray_intensity[color]
        elif color == "yellow":
            if pixel_hue in range(color_hue-15, color_hue + 15) and pixel_saturation < 250:
                gray_value = gray_intensity[color]
        else:
            if pixel_hue in range(color_hue-15, color_hue + 15):
                gray_value = gray_intensity[color]
    return gray_value


def get_hsv_from_bgr(bgr_image):
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_image = rgb_image.astype('float32')
    rgb_image = rgb_image/255

    hsv_image = matplotlib.colors.rgb_to_hsv(rgb_image)
    hue_values = hsv_image[:, :, 0]
    saturation_values = hsv_image[:, :, 1]
    hsv_image[:,:,0] = hue_values*360
    hsv_image[:,:,1] = saturation_values*255
    return hsv_image


if __name__ == "__main__":
    image = cv2.imread(IMAGE_PATH)
    hsv_image = get_hsv_from_bgr(image)
    gray_image = convert_from_colored_to_gray(hsv_image)

    gray_image = cv2.medianBlur(gray_image, 5)
    cv2.imshow('image', gray_image)
    cv2.waitKey(0)

    # Perform 'closing' morphological operation on the image
    kernel = np.ones((1, 1), np.uint8)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

    # Figure out the scaling parameter according to original size and then scale
    gray_image = cv2.resize(gray_image, (0, 0), fx=SCALING_FACTOR, fy=SCALING_FACTOR)

    # Find the threshold to separate foreground from background using OTSU's thresholding method
    threshold = threshold_otsu(gray_image)
    (rows, cols) = gray_image.shape

    '''
    Create a 3D voxel data from the image
    The top-most (#1) and bottom-most (#13) layer will contain all zeros
    The middle 10 layers (#3 to #12) contain the same pixel values as the grayscale image
    There is an additional layer(#2) for the base of the model 
    '''
    layers = 14
    rows += 2
    cols += 2
    voxel = np.zeros((rows, cols, layers))
    voxel[:, :, 1] = np.ones((rows, cols)).astype('float32')
    voxel[:, :, 2] = np.ones((rows, cols)).astype('float32')

    # making the boundary voxel values to be zero, for the marching cubes algorithm to work correctly
    voxel[0, :, :] = np.zeros((cols, layers)).astype('float32')
    voxel[(rows - 1), :, :] = np.zeros((cols, layers)).astype('float32')

    voxel[:, 0, :] = np.zeros((rows, layers)).astype('float32')
    voxel[:, (cols - 1), :] = np.zeros((rows, layers)).astype('float32')

    '''
    Create the middle 10 layers from the image
    Based on the pixel values the layers are created to assign different heights to different regions in the image
    '''
    for level in range(1, 10):
        level_threshold = level * 0.1;
        for j in range(0, rows - 2):
            for k in range(0, cols - 2):
                pixel_value = gray_image[j][k]
                if (pixel_value > level_threshold):
                    voxel[j + 1][k + 1][level + 2] = pixel_value

    '''
    Run the marching cubes algorithm to extract surface mesh from 3D volume. Params:
        volume : (M, N, P) array of doubles
            Input data volume to find isosurfaces. Will be cast to `np.float64`.
        level : float
            Contour value to search for isosurfaces in `volume`. If not
            given or None, the average of the min and max of vol is used.
        spacing : length-3 tuple of floats
            Voxel spacing in spatial dimensions corresponding to numpy array
            indexing dimensions (M, N, P) as in `volume`.
        gradient_direction : string
            Controls if the mesh was generated from an isosurface with gradient
            descent toward objects of interest (the default), or the opposite.
            The two options are:
            * descent : Object was greater than exterior
            * ascent : Exterior was greater than object
    '''
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume=voxel, level=threshold,
                                                                   spacing=(1., 1., 1.), gradient_direction='descent')

    # Export the mesh as stl
    mymesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        for j in range(3):
            mymesh.vectors[i][j] = verts[f[j], :]

    # Get the file name (excluding extension) from the input image path
    filename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    mymesh.save(OUTPUT_PATH + filename + '.stl')
