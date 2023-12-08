import os
from tqdm import tqdm
from glob import glob

import numpy as np
import pandas as pd

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_moments(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    
    if std_dev == 0:
        skewness = 0
        kurtosis = 0
    else:
        skewness = np.mean((image - mean) ** 3) / (std_dev ** 3)
        kurtosis = np.mean((image - mean) ** 4) / (std_dev ** 4) - 3
    
    return mean, std_dev, skewness, kurtosis

def extract_color_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract moments for each HSV component
    h_mean, h_std, h_skewness, h_kurtosis = calculate_moments(hsv_image[:,:,0])
    s_mean, s_std, s_skewness, s_kurtosis = calculate_moments(hsv_image[:,:,1])
    v_mean, v_std, v_skewness, v_kurtosis = calculate_moments(hsv_image[:,:,2])

    return [h_mean, h_std, h_skewness, h_kurtosis,
            s_mean, s_std, s_skewness, s_kurtosis,
            v_mean, v_std, v_skewness, v_kurtosis]

def create_ring_masks_const(image, num_rings):
    p = num_rings
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    masks = []

    for i in range(num_rings):
        area_ratio = (i + 1) / num_rings
        radius_outer = int(np.sqrt(area_ratio) * min(height/p, width/p))

        if i == 0:
            radius_inner = 0
        else:
            area_ratio_prev = i / num_rings
            radius_inner = int(np.sqrt(area_ratio_prev) * min(height/(p+1), width/(p+1)))

        mask = np.zeros_like(image, dtype = np.uint8)
        cv2.circle(mask, center, radius_outer, (255, 255, 255), thickness = -1)
        cv2.circle(mask, center, radius_inner, (0, 0, 0), thickness = -1)

        masks.append(mask)
        p -= 1

    return masks

def create_ring_masks_with_growth_factor(image, num_rings, growth_rate = 0.8, inner_outer_ratio = 0.5):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    masks = []

    for i in range(num_rings):
        adjusted_growth_rate = growth_rate ** i
        radius_ratio = inner_outer_ratio + (1 - inner_outer_ratio) * (i / num_rings)
        radius_outer = int(adjusted_growth_rate * radius_ratio * min(height, width))

        radius_inner = int(radius_outer * inner_outer_ratio)

        mask = np.zeros_like(image, dtype = np.uint8)
        cv2.circle(mask, center, radius_outer, (255, 255, 255), thickness = -1)
        cv2.circle(mask, center, radius_inner, (0, 0, 0), thickness = -1)

        masks.append(mask)

    return masks


def lab_threshold(image_path, lower_bound = np.array([0, 0, 0]), upper_bound = np.array([255, 255, 255])):
    image = cv2.imread(image_path)
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    mask = cv2.inRange(lab_img, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask = mask)

    return result

def hsl_threshold(image_path, lower_bound = np.array([0, 0, 0]), upper_bound = np.array([179, 255, 100])):
    image = cv2.imread(image_path)
    hsl_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    mask = cv2.inRange(hsl_img, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask = mask)

    return result

def apply_adaptive_histogram_equalization(image, clip_limit = 1.5, tile_size = (8, 8)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_image, (15, 15), 0)

    clahe = cv2.createCLAHE(clipLimit = clip_limit, 
                            tileGridSize = tile_size)
    equalized_image = clahe.apply(blur_img)
    
    return equalized_image

def calculate_gradient(image, kernel_size = 3):
    equalized_img = apply_adaptive_histogram_equalization(image, tile_size = (3, 3))
    # Compute gradient using Sobel operators
    gradient_x = cv2.Sobel(equalized_img, cv2.CV_64F, 1, 0, ksize = kernel_size)
    gradient_y = cv2.Sobel(equalized_img, cv2.CV_64F, 0, 1, ksize = kernel_size)

    # Compute gradient magnitude and orientation
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_orientation = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_orientation


def crop_coins_from_background(img_file_name, color_threshold_method = None, kernel_size = 3):
    if color_threshold_method is None:
        image = cv2.imread(img_file_name)
    elif color_threshold_method == 'lab':
        image = lab_threshold(img_file_name, 
                              lower_bound = np.array([0, 0, 0]), 
                              upper_bound = np.array([255, 255, 255]))
    elif color_threshold_method == 'hsl':
        image = hsl_threshold(img_file_name, 
                              lower_bound = np.array([0, 0, 0]), 
                              upper_bound = np.array([179, 255, 100]))
    else:
        raise ValueError("Unsupported color space")

    equalized_img = apply_adaptive_histogram_equalization(image, tile_size = (3, 3))

    # Compute gradient using Sobel operators
    gradient_x = cv2.Sobel(equalized_img, cv2.CV_64F, 1, 0, ksize = kernel_size)
    gradient_y = cv2.Sobel(equalized_img, cv2.CV_64F, 0, 1, ksize = kernel_size)

    # Compute gradient magnitude and orientation
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    percentage_threshold = 0.1
    absolute_threshold = percentage_threshold * np.max(gradient_magnitude)
    edges = (gradient_magnitude > absolute_threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped_objects = []
    min_area_threshold = 1000
    len_contours = len(cropped_objects)

    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area_threshold:
            valid_contours.append(contour)
    
    valid_contours_len = len(valid_contours)
    if valid_contours_len > 15:
        raise Exception(f"Detected valid coins {valid_contours_len}/{len_contours} [TOO MANY, FAILED]")


    for i, contour in enumerate(valid_contours):
        mask = np.zeros_like(edges)
        cv2.drawContours(mask, [contour], 0, (255), thickness = cv2.FILLED)

        object_cropped = cv2.bitwise_and(image, image, mask = mask)
        x, y, w, h = cv2.boundingRect(contour)

        if w * h > min_area_threshold:
            object_cropped = object_cropped[y: y + h, x: x + w]
            print(f"Cropped {i+1}/{valid_contours_len}")
            cropped_objects.append(object_cropped)

    return cropped_objects

def extract_shape_features(image, coin_features_df, ind):
    num_rings = 5
    color_features = extract_color_features(image)
    ring_masks = create_ring_masks_with_growth_factor(image, num_rings, growth_rate = 0.8, inner_outer_ratio = 0.5)

    # s_std
    coin_features_df.loc[ind, 's_std'] = color_features[5]

    # ring_1_s_kurtosis
    cropped_1 = cv2.bitwise_and(image, image, mask = ring_masks[0][:, :, 0])
    ring_1_color_features = extract_color_features(cropped_1)
    coin_features_df.loc[ind, 'ring_1_s_kurtosis'] = ring_1_color_features[7]

    # Ring 2
    cropped_2 = cv2.bitwise_and(image, image, mask = ring_masks[1][:, :, 0])
    ring_2_color_features = extract_color_features(cropped_2)
    coin_features_df.loc[ind, 'ring_2_s_std'] = ring_2_color_features[5]
    coin_features_df.loc[ind, 'ring_2_v_kurtosis'] = ring_2_color_features[11]
    
    # Ring 3
    cropped_3 = cv2.bitwise_and(image, image, mask = ring_masks[2][:, :, 0])
    ring_3_color_features = extract_color_features(cropped_3)
    coin_features_df.loc[ind, 'ring_3_h_std'] = ring_3_color_features[1]
    coin_features_df.loc[ind, 'ring_3_s_kurtosis'] = ring_3_color_features[7]

    # Ring 4
    cropped_4 = cv2.bitwise_and(image, image, mask = ring_masks[3][:, :, 0])
    ring_4_gradient_magnitude, _ = calculate_gradient(cropped_4, kernel_size = 3)
    coin_features_df.loc[ind, 'ring_4_magnitude_median'] = np.median(ring_4_gradient_magnitude)

def extract_features_single_image(image):
    sigle_img_features_df = pd.DataFrame({
        "s_std": [],

        "ring_1_s_kurtosis": [],

        "ring_2_s_std": [],
        "ring_2_v_kurtosis": [],

        "ring_3_h_std": [],
        "ring_3_s_kurtosis": [],

        "ring_4_magnitude_median": []
    })
    extract_shape_features(image, sigle_img_features_df, 0)
    return sigle_img_features_df

def extract_image_features(img_file_name):
    coin_features_df = pd.DataFrame({
        "s_std": [],

        "ring_1_s_kurtosis": [],

        "ring_2_s_std": [],
        "ring_2_v_kurtosis": [],

        "ring_3_h_std": [],
        "ring_3_s_kurtosis": [],

        "ring_4_magnitude_median": []
    })
    cropped_objects = crop_coins_from_background(img_file_name)
    
    for ind, cropped_obj in enumerate(cropped_objects):
        extract_shape_features(cropped_obj, coin_features_df, ind)

    return coin_features_df
