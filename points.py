#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 10:26:54 2017

@author: easton
"""
#%cd ~/Documents/Shape Recognition/Drawing App/Draw Shapes/

import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
#import matplotlib.pyplot as plt


def process_image(subdir, IMG_SIZE = 28):
    features = []
    images = []
    names = []
    for filename in os.listdir(subdir):
        if filename.endswith('.csv'):
            connected = False
            points = pd.read_csv(os.path.join(subdir, filename))
            
            mins = points.min()
            maxs = points.max() - mins
            
            max_big = maxs.max()
            max_small = maxs.min()
            max_dim = maxs.argmax()
            min_dim = 'x' if max_dim == 'y' else 'y'
            
            divisor = max_big // IMG_SIZE + 1
            MAX_CENTERING_CONST = (IMG_SIZE - max_big % IMG_SIZE) / 2 - mins[max_dim]
            MIN_CENTERING_CONST = (divisor * IMG_SIZE - max_small) / 2 - mins[min_dim]
            
            pixels = pd.DataFrame({
                    max_dim:(points[max_dim] + MAX_CENTERING_CONST) // divisor,
                    min_dim:(points[min_dim] + MIN_CENTERING_CONST) // divisor
                    }, dtype = int)
            pixels = pixels.drop_duplicates().values

            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
            for pixel in pixels:
                img[pixel[0], pixel[1]] = True
            images.append(img)

# plot original shapes
#             print(filename)
#             plt.imshow(img, cmap='gray')
#             plt.show()           
#             
#==============================================================================
            pixels = pixels.tolist()
            pixels.append(pixels[0])
            next_round = pixels
            pixels = np.array(pixels)    
            deleted_one = True
            while deleted_one:
                deleted_one = False
                h_list = []
                for i in range(1, len(pixels)-1):
                    u = pixels[i] - pixels[i-1]
                    v = pixels[i+1] - pixels[i-1]
                    theta = np.arccos(u.dot(v)/np.ceil(np.linalg.norm(u) * np.linalg.norm(v)))
                    h_list.append(np.sin(theta) * np.linalg.norm(u))
                min_h = min(h_list)
                if min_h < IMG_SIZE / 8:
                    deleted_one = True                
                    min_indices = [i+1 for i, h in enumerate(h_list) if h == min_h]
                    pixels_left = len(pixels) - len(min_indices)
                    if pixels_left <= 3:
                        deleted_one = False
                        for i in range(3-pixels_left):
                            min_indices.pop()
                    next_round = [point for i, point in enumerate(next_round) if i not in min_indices]
                    pixels = np.array(next_round)
                        
            # Only delete last point if it's really close to the first one
            last_idx = len(pixels) -2
            first_to_last = pixels[0].dot(pixels[last_idx]) ** 0.5
            pixels = pixels.tolist()
            pixels.append(pixels[1])
            if first_to_last < (IMG_SIZE / 4) and len(pixels)  > 3:
                pixels.pop(last_idx)
                connected = True
            pixels = np.array(pixels)
                      
# plot new shape
#==============================================================================
#             img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
#             for pixel in pixels:
#                 img[pixel[0], pixel[1]] = True
#             images.append(img)
#             plt.imshow(img, cmap='gray')
#             plt.show()
#             
#==============================================================================
            angles = []
            side_lengths = []
            for i in range(1, len(pixels) -1): #angles and side lengths
                u = pixels[i] - pixels[i-1]
                v = pixels[i+1] - pixels[i]
                theta = np.arccos(u.dot(v)/np.ceil(np.linalg.norm(u) * np.linalg.norm(v)))
                angles.append(theta)
                side_lengths.append(np.linalg.norm(u))
            
            features_current = []
            features_current.append(connected) # Is shape connected?
            features_current.append(len(angles)) # number of sides
            features_current.append(np.mean(angles))
            features_current.append(np.var(angles))
            features_current.append(skew(angles))
            features_current.append(kurtosis(angles))
            features_current.append(np.mean(side_lengths))
            features_current.append(np.var(side_lengths))
            features_current.append(skew(side_lengths))
            features_current.append(kurtosis(side_lengths))
            
            features.append(features_current)
            names.append(filename.split(' ')[0])
    features = np.array(features)
    images = np.array(images)
    return features, images, names
    
        
