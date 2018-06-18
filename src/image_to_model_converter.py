import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.filters import threshold_otsu
from stl import mesh
from collections import defaultdict
from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog


IMAGE_PATH = ''
OUTPUT_PATH = '../stl-files/'
SCALING_FACTOR = 0.9

height_map = {}


def convert_from_colored_to_gray(image, distinct_colors):
    rows = image.shape[0]
    cols = image.shape[1]
    gray_image = np.zeros((rows, cols)).astype('float32')
    for i in range(0, rows):
        for j in range(0, cols):
            pixel_val = tuple(image[i, j, :].astype('int'))
            gray_image[i][j] = get_gray_value_from_color(pixel_val, distinct_colors)
    return gray_image


def get_gray_value_from_color(pixel, distinct_colors):
    pixel_hue = pixel[0]
    pixel_sat = pixel[1]
    pixel_brightness = pixel[2]
    gray_value = 0
    for color in distinct_colors:
        color_hue = color[0]
        color_sat = color[1]
        color_brightness = color[2]
        if pixel_hue in range(color_hue-5, color_hue + 5) and pixel_sat in range(color_sat - 10, color_sat + 10)\
                and pixel_brightness in range(color_brightness - 10, color_brightness + 10):
            gray_value = height_map[color]*0.1
    return gray_value


def get_hsv_from_bgr_image(bgr_image):
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_image = rgb_image.astype('float32')
    rgb_image = rgb_image/255

    hsv_image = matplotlib.colors.rgb_to_hsv(rgb_image)
    hue_values = hsv_image[:, :, 0]
    saturation_values = hsv_image[:, :, 1]
    brightness_values = hsv_image[:, :, 2]
    hsv_image[:, :, 0] = (hue_values*360).astype('int')
    hsv_image[:, :, 1] = (saturation_values*100).astype('int')
    hsv_image[:, :, 2] = (brightness_values*100).astype('int')
    return hsv_image


def get_hsv_from_bgr_pixel(pixel):
    pixel_arr = [pixel[2], pixel[1], pixel[0]]
    rgb = tuple(pixel_arr)
    rgb_normalized = map(lambda x: x/255.0, rgb)
    hsv_pixel_val = matplotlib.colors.rgb_to_hsv(rgb_normalized)
    hue = int(hsv_pixel_val[0]*360)
    sat = int(hsv_pixel_val[1]*100)
    brightness = int(hsv_pixel_val[2]*100)
    hsv = tuple([hue, sat, brightness])
    return hsv


def get_bgr_from_hsv_pixel(pixel):
    hue = pixel[0]/360.0
    sat = pixel[1]/100.0
    brightness = pixel[2]/100.0
    rgb_pixel_normalized = matplotlib.colors.hsv_to_rgb([hue, sat, brightness])
    rgb_pixel = np.multiply(rgb_pixel_normalized, 255).astype('int')
    bgr_pixel = [rgb_pixel[2], rgb_pixel[1], rgb_pixel[0]]
    return tuple(bgr_pixel)


def get_matching_color(color_to_be_matched, color_list):
    hue = color_to_be_matched[0]
    sat = color_to_be_matched[1]
    brightness = color_to_be_matched[2]
    if sat < 10 and brightness > 90:
        return tuple([0, 0, 100])
    if sat < 10 and brightness < 10:
        return tuple([0, 0, 0])
    for color in color_list:
        color_hue = color[0]
        color_sat = color[1]
        color_brightness = color[2]
        if hue in range(color_hue - 5, color_hue + 5) and sat in range(color_sat - 10, color_sat + 10)\
                and brightness in range(color_brightness - 10, color_brightness + 10):
            return color
    return None


def deduplicate_entries(colors_map, top_colors_limit, total_number_of_pixels):
    top_colors = {}
    top_colors[(0, 0, 100)] = 0
    top_colors[(0, 0, 0)] = 0
    for color in colors_map:
        matching_color = get_matching_color(color, top_colors.keys())
        if matching_color is None:
            top_colors[color] = colors_map[color]
        else:
            top_colors[matching_color] += colors_map[color]
    top_colors_filtered = {k: v for k, v in top_colors.iteritems() if v/float(total_number_of_pixels)*100 > 0.1}
    top_colors_sorted = sorted(top_colors_filtered, key=top_colors_filtered.get, reverse=True)
    return top_colors_sorted[0:top_colors_limit]


def fetch(entries, root):
    for entry in entries:
        field = entry[0]
        text = entry[1].get()
        if text == '':
            height_map[field] = 0
        else:
            height_map[field] = int(text)
    root.destroy()


def makeform(root, colors):
    entries = []
    for color in colors:
        bgr_color = get_bgr_from_hsv_pixel(color)
        rgb_color = tuple([bgr_color[2], bgr_color[1], bgr_color[0]])
        background_color = '#%02x%02x%02x' % rgb_color
        row = Frame(root)
        lab = Label(row, width=10, text='', bg=background_color, anchor='w')
        ent = Entry(row)
        row.pack(side=TOP, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT)
        entries.append((color, ent))
    return entries


def assign_height_to_colors(distinct_colors):
    root = Tk()
    root.title("Enter heights")
    instruction = Label(0, text="Please enter the height you want to assign to each color,\n"
                                "in the range 0 to 10. (Blank entries will be considered as 0)", padx=10)
    instruction.pack()
    ents = makeform(root, distinct_colors)
    root.bind('<Return>', (lambda event, e=ents: fetch(e, root)))
    b1 = Button(root, text='Submit', command=(lambda e=ents: fetch(e, root)))
    b1.pack(side=LEFT, padx=5, pady=5)
    root.mainloop()


def get_image_path():
    root = Tk()
    def browsefunc():
        filename = tkFileDialog.askopenfilename(initialdir="/", title="Select file")
        pathlabel.config(text=filename)
        global IMAGE_PATH
        IMAGE_PATH = filename
        root.destroy()

    instruction = Label(0, text="Choose the image that has to be converted", padx=10)
    instruction.pack()
    browsebutton = Button(root, text="Browse", command=browsefunc)
    browsebutton.pack()

    pathlabel = Label(root)
    pathlabel.pack()
    root.mainloop()


def get_top_colors_hsv(image, no_of_colors):
    # map storing the frequency of each pixel color (in bgr) present in the image
    color_freq_map = defaultdict(int)
    (rows, cols, levels) = image.shape
    for i in range(0, rows):
        for j in range(0, cols):
            pixel_color = tuple(image[i, j, :].astype('int'))
            color_freq_map[pixel_color] += 1
    sorted_colors = sorted(color_freq_map, key=color_freq_map.get, reverse=True)
    top_colors = sorted_colors[0:100]
    top_colors_map = {}
    for color in top_colors:
        hsv_color = get_hsv_from_bgr_pixel(color)
        if hsv_color in top_colors_map:
            top_colors_map[hsv_color] += color_freq_map[color]
        else:
            top_colors_map[hsv_color] = color_freq_map[color]
    top_colors_hsv = deduplicate_entries(top_colors_map, no_of_colors, rows*cols)
    return top_colors_hsv


if __name__ == "__main__":
    get_image_path()
    image = cv2.imread(IMAGE_PATH)

    distinct_colors = get_top_colors_hsv(image, 11)

    assign_height_to_colors(distinct_colors)

    hsv_image = get_hsv_from_bgr_image(image)
    gray_image = convert_from_colored_to_gray(hsv_image, distinct_colors)

    gray_image = cv2.medianBlur(gray_image, 3)
#    cv2.imshow('image', gray_image)
#    cv2.waitKey(0)

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
        level_threshold = level * 0.1
        for j in range(0, rows - 2):
            for k in range(0, cols - 2):
                pixel_value = gray_image[j][k]
                if pixel_value > level_threshold:
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

'''
In [47]: get_top_colors_hsv(image, 10)
Out[47]:
[(0, 0, 100), : white
 (40, 99, 53), : brown
 (54, 97, 99), : yellow
 (83, 99, 60), : light green
 (252, 22, 74), : purple
 (66, 99, 49), : root green
 (13, 75, 79), : red
 (133, 35, 50), : stem green
 (225, 92, 76), : blue
 (192, 87, 96), : sky blue
 '''

'''
TODO:
'''
