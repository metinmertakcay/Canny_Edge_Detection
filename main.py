# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:35:18 2019
@author: Metin Mert Akçay
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

FOLDER_PATH = "images"
OUTPUT_PATH = "output"
SMOOTHING = "smoothing"
GRADIENT= "gradient"
SUPRESSION = "non-maximum-supression"
THRESHOLD = "threshold"
HYSTERESIS = "canny_output"

""" This function is used for print all image names to console
    @param image_list: all image names in the "images" path
"""
def show_image_names(image_list):
    for index, image_name in enumerate(image_list):
        print("%d) %s" % (index, image_name.split(".")[0]))


""" This function is used for read image as grayscale image
    @param image_name: name of the image
    @return image: grayscale image
"""
def read_image_as_grayscale(image_name):
    print("- Read image as grayscale -")
    image = cv2.imread(os.path.join(FOLDER_PATH, image_name), cv2.IMREAD_GRAYSCALE)
    return image


""" This function is used for write image to given folder
    @param image: image to be written
    @param image_name: name of the image to be saved
    @param operation: name of the canny edge detection step
"""
def write_image(image, image_name, operation):
    if(not(os.path.exists(OUTPUT_PATH))):
        os.makedirs(OUTPUT_PATH)
    cv2.imwrite(os.path.join(OUTPUT_PATH , image_name), np.array(image))
    plt.imshow(np.array(image), cmap='gray', vmin=0, vmax=255)
    plt.title(operation)
    plt.show()


""" This function is used for convolution operation on image
    @param image: matrix to be convoluted
    @param kernel: matrix to be used in convolution
    @return new_image: matrix formed by convolution
"""
def convolution(image, kernel):
    image_row_len = len(image)
    image_col_len = len(image[0])
    i = int(len(kernel) / 2)
    
    # Created new matrix for convolution operation
    new_image = []
    while (i < image_row_len - int(len(kernel) / 2)):
        j = int(len(kernel) / 2)
        new_image_col = []
        while (j < image_col_len - int(len(kernel) / 2)):
            pixel_value = 0
            k = (-1) * int(len(kernel) / 2)
            while (k <= int(len(kernel) / 2)):
                l = (-1) * int(len(kernel) / 2)
                while (l <= int(len(kernel) / 2)):
                    pixel_value += image[i + k][j + l] * kernel[k + int(len(kernel) / 2)][l + int(len(kernel) / 2)]
                    l +=1
                k += 1
            new_image_col.append(int(round(pixel_value)))
            j += 1
        new_image.append(new_image_col)
        i += 1
    return new_image


""" This function is used for blur the image to remove noise.
    @param image: grayscale image
    @return: blurred image
"""
def smoothing(image):
    print("- Smoothing operation -")    
    gauss_kernel = np.array([[1, 4,  7,  4,  1],
                             [4, 16, 26, 16, 4],
                             [7, 26, 41, 26, 7],
                             [4, 16, 26, 16, 4],
                             [1, 4,  7,  4,  1]]) / 273;

    return convolution(image, gauss_kernel)   


""" This function is used for normalization operation
    @param image: image to be normalized
    @return image: normalized image
"""
def normalization(image):
    max_val = np.amax(image)
    min_val = np.amin(image)
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = int(((image[i][j] - min_val) / (max_val - min_val) * 255))
    return image


""" This function is used for finding gradients operation
    @param: smoothed image
    @return angle: image angle information
    @return output: image with gradients
"""
def find_gradients(image):
    print("- Finding gradient operation -")
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    output_Gx = convolution(image, Gx)    
    output_Gy = convolution(image, Gy)
    
    angle = []
    output = []
    for i in range(len(output_Gx)):
        angle_column = []
        output_column = []
        for j in range(len(output_Gx[0])):
            try:
                angle_val = np.arctan(output_Gy[i][j] / output_Gx[i][j]) * 180 / np.pi
                if (angle_val < 0):
                    angle_val += 180
                angle_column.append(angle_val)
            except ZeroDivisionError: # argtan(Inf)
                angle_column.append(90)
            output_column.append(int(round(np.sqrt(output_Gx[i][j] ** 2 + output_Gy[i][j] ** 2))))
        angle.append(angle_column)
        output.append(output_column)
    return angle, normalization(output)


""" This function is used for non_maximum supression operation
    @param image: image to be processed
    @param angle: image angle information
    @return nms_image: the result of the non_maximum supression operation
"""
def non_maximum_supression(image, angle):
    print("- Non maximum supression operation -")
    nms_image = np.zeros([len(image), len(image[0])], dtype=int)
    
    for i in range(1, len(image) - 1):
        for j in range(1, len(image[0]) - 1):
            # angle 0
            if (0 <= angle[i][j] and angle[i][j] < 22.5) or (157.5 <= angle[i][j] and angle[i][j] <= 180):
                if (image[i][j] >= image[i][j+1]) and (image[i][j] >= image[i][j-1]):
                    nms_image[i][j] = image[i][j]
            # angle 45
            elif (22.5 <= angle[i][j] and angle[i][j] < 67.5):
                if (image[i][j] >= image[i-1][j+1]) and (image[i][j] >= image[i+1][j-1]):
                    nms_image[i][j] = image[i][j]
            # angle 90
            elif (67.5 <= angle[i][j] and angle[i][j] < 112.5):
                if (image[i][j] >= image[i+1][j]) and (image[i][j] >= image[i-1][j]):
                    nms_image[i][j] = image[i][j]
            # angle 135
            elif (112.5 <= angle[i][j] and angle[i][j] < 157.5):
                if (image[i][j] >= image[i+1][j+1]) and (image[i][j] >= image[i-1][j-1]):
                    nms_image[i][j] = image[i][j]
    return nms_image


""" This function is used for identify strong or weak edges
    @param image: image to be processed
    @param low_thresh_ratio: ratio used to determine the low threshold value
    @param high_thresh_ratio: ratio used to determine the high threshold value
    @param weak: weak edge pixel value
    @param strong: strong edge pixel value
    @return output_image: the result of the threshold operation
"""
def threshold(image, low_thresh_ratio=0.06, high_thresh_ratio=0.12, weak=50, strong=255):
    print("- Threshold operation -")
    high_threshold = np.amax(image) * high_thresh_ratio;
    low_threshold = np.amax(image) * low_thresh_ratio;

    output_image = []
    for i in range(len(image)):
        output_image_col = []
        for j in range(len(image[0])):
            if(image[i][j] >= high_threshold):
                output_image_col.append(strong)
            elif(high_threshold >= image[i][j] and low_threshold <= image[i][j]):
                output_image_col.append(weak)
            else:
                output_image_col.append(0)
        output_image.append(output_image_col)
    return output_image


""" This function is used for copy the image pixels.
    @param image: image to be copy
    @return new_image: copied image
"""
def copy(image):
    copy_image = []
    for i in range(len(image)):
        copy_image_col = []
        for j in range(len(image[0])):
            copy_image_col.append(image[i][j])
        copy_image.append(copy_image_col)
    return copy_image


""" This function is used for determine determines whether weak edges remain or delete
    @param image: image to be processed
    @param weak: weak edge pixel value
    @param strong: strong edge pixel value
    @return output_image: canny edge detection output image
"""
def hysteresis(image, weak=50, strong=255):
    print("- Hysteresis operation -")
    output_image = copy(image)
    for i in range(1, len(image) - 1):
        for j in range(1, len(image[0]) - 1):
            if (image[i][j] == weak):
                if ((image[i+1][j-1] == strong) or (image[i+1][j] == strong) or (image[i+1][j+1] == strong)
                 or (image[i][j-1] == strong) or (image[i][j+1] == strong) 
                 or (image[i-1][j-1] == strong) or (image[i-1][j] == strong) or (image[i-1][j+1] == strong)):
                    output_image[i][j] = strong
                else:
                    output_image[i][j] = 0
    return output_image


""" This function is used for get the image number
    @return image_no: number of the image
"""
def get_image_no():
    try:
        image_no = int(input("\nPlease enter the number of image -->"))
    except ValueError:
        image_no = -1
    return image_no


""" This is where the code starts """
if __name__ == '__main__':
    # Get image names given folder
    image_list = os.listdir(FOLDER_PATH)
    # Print image names to console.
    show_image_names(image_list)
    
    cont = True
    while(cont):
        # Select picture to user
        image_no = get_image_no()
        while (image_no < 0) or (len(image_list) < image_no):
            image_no = get_image_no()
        image_name = image_list[image_no]
        
        # Read image as grayscale
        image = read_image_as_grayscale(image_name)
        # Make smoothing operation on image
        image = smoothing(image)
        # Write smoothed image
        write_image(image, SMOOTHING + "_" + image_name, SMOOTHING)
        # Apply sobel filters on image and find angle of the edge
        angle, image = find_gradients(image)
        # Write image with edges
        write_image(image, GRADIENT + "_" + image_name, GRADIENT)
        # Non-maximum supression
        image = non_maximum_supression(image, angle)
        # Write image with supression
        write_image(image, SUPRESSION + "_" + image_name, SUPRESSION)
        # Threshold method apply image
        image = threshold(image)
        # Write image result threshold method
        write_image(image, THRESHOLD + "_" + image_name, THRESHOLD)
        # Hysteresis method apply image
        image = hysteresis(image)
        #write image result of hysteresis method
        write_image(image, HYSTERESIS + "_" + image_name, HYSTERESIS)
        
        press = input("If you want to continue please enter 'c' character-->")
        if(not(press == "c" or press == "C")):
            cont = False
    