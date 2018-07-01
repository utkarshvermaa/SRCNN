import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import scipy as scp

def getPatches(image, height, width, patch_container, patch_dimension, overlap):
    i=0
    while (i<height):
        j=0
        while (j<width):
            if i+patch_dimension <= height-1 and j+patch_dimension <= width-1:
                rs=i
                re = i+patch_dimension
                cs = j
                ce = j+patch_dimension
                #print 'if-1'
            if i+patch_dimension >= height and j+patch_dimension <=width-1:
                rs = height-(patch_dimension)
                re = height
                cs = j
                ce = j+patch_dimension
                #print 'if-2'
            if i+patch_dimension <= height-1 and j+patch_dimension >=width:
                rs = i
                re = i+patch_dimension
                cs = width - (patch_dimension)
                ce = width
                #print 'if-3'
                #print j
            if i+patch_dimension >= height and j+patch_dimension >=width:
                rs = height-(patch_dimension)
                re = height
                cs = width - (patch_dimension)
                ce = width
                #print 'if-4'

            cropimage = image[rs:re, cs:ce]
            patch_container.append(cropimage)
            j=j+(patch_dimension-overlap)
        i=i+(patch_dimension-overlap)
    return patch_container

def append_image(location):
    patch_container = []
    for img_from_folder in sorted(glob.glob(location + "/*png")):
        img = cv2.imread(img_from_folder)
        (img,_,__) = cv2.split(img)
        patch_container.append(img)
    return patch_container

def create_patch_normal(patchsize,location,overlap):
    patch_container = []
    for img_from_folder in sorted(glob.glob(location + "/*png")):
        img = cv2.imread(img_from_folder)
        (img,_,__) = cv2.split(img)
        width = img.shape[1]
        height = img.shape[0]
        patch_container = getPatches(img, height, width, patch_container, patchsize, overlap)
    return patch_container

def sub_im_create_patch(patchsize,gt_array,res_array,overlap):
    patch_container = []
    k = np.array(gt_array).shape[0]
    for i in range(k):
        diff_im = cv2.subtract(gt_array[i],res_array[i])
        print("printing actual_residue_image")
        cv2.imwrite("actual_residue_image.png", diff_im)
        width = diff_im.shape[1]
        height = diff_im.shape[0]
        patch_container = getPatches(diff_im, height, width, patch_container, patchsize, overlap)
    return patch_container

def nextbatch(batch_i):
    global ll
    global hl
    
    ll = batch_i*batch_size
    ul = batch_i*batch_size + (batch_size)
    #print (ll)
    #print (ul)
    #if ul < totsize
    tempx = inputs[ll:ul].copy()
    #print('tempx_data shape:', np.array(tempx).shape)
    tempy = lables[ll:ul].copy()
    #print('tempy_data shape:', np.array(tempy).shape)
    """
    test_tempx = np.reshape(tempx[25],(patch_size,patch_size))
    test_tempy = np.reshape(tempy[25],(patch_size,patch_size))
    plt.imshow(test_tempx)
    plt.show()
    plt.imshow(test_tempy)
    plt.show()
    """

    #print tempnoisy.shape
    #tempx = tempx.reshape(batch_size, patch_width*patch_height)
    #tempy = tempy.reshape(batch_size, patch_width*patch_height)
    #print(tempy)
    #print(tempx)
    #ll = ll+incr
    return tempy, tempx

import tensorflow as tf
import numpy as np

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

