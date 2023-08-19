# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 21:13:35 2023

@author: Wahyu Hutomo Nugroho (hutomonugroho@gmail.com)
"""
import os                                                                       #It returns a dictionary having user’s environmental variable as key and their values as value
os.environ['THEANO_FLAGS']="device=cuda0, floatX=float32"
import numpy as np                                                             # pydicom reads modify and write data in DICOM files
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)   
from scipy.ndimage import morphology
from skimage.measure import label, regionprops                                                                   # a scientific computation library that uses NumPy underneath It provides more utility functions for optimization, stats and signal processing
from scipy import ndimage as ndi
import cv2

def gambar_mask(fig_img, fill_holes) :
    mask = np.zeros(fig_img.shape, dtype=int)
    contours = {}
    segmen = np.zeros(fig_img.shape)
    kontur = np.asarray(fig_img, dtype = 'uint8')
    for i in range(kontur.shape[-1]) :    # i adalah indeks slice
        contours[i], hierarchy = cv2.findContours(image= kontur[:,:,i], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        for j in range(len(contours[i])) :  #j adalah indeks kontur dalam tiap slice
            xy = np.array([contours[i][j][:,0,1], contours[i][j][:,0,0]])
            kontur[xy[0,:], xy[1,:], i] = 1
        if fill_holes==True :
            segmen[:,:,i] = ndi.binary_fill_holes(kontur[:,:,i])
        else:
            segmen[:,:,i] = kontur[:,:,i]
        mask_label = np.vectorize(label, signature = '(n,m)->(n,m)')(segmen[:,:,i])    #Memberikan label pada setiap segmen
        rps = regionprops(mask_label)
        areas = [r.area for r in rps]
        idx_area = np.argsort(areas)[::-1]
        if idx_area.size == 0:
            mask[:,:,i] = fig_img[:,:,i]
            print("Tidak ada area ditemukan")
            continue
        koordinat = rps[idx_area[0]].coords
        mask[koordinat[:, 0][:], koordinat[:, 1][:], i] = 1
        # print("Ketemu : ", idx_area.size)
        # if idx_area.size > 1 :
        #     koordinat2 = rps[idx_area[1]].coords
        #     mask[koordinat2[:, 0][:], koordinat2[:, 1][:], i] = 1
    return mask


def membuat_mask(calon_mask, treshold, fill_holes) :
     fig_img = calon_mask > treshold #860
     mask = gambar_mask(fig_img, fill_holes)
     return mask


def image_processing(fig_prediksi) :
    key_image = {}
    for i in list(range(0, len(fig_prediksi))) :
        shape = fig_prediksi[i].shape
        image = np.zeros(shape)
        for j in range(shape[-1]) :
            tmp = fig_prediksi[i][:, :, j]
            image_filled =  np.asarray(ndi.binary_fill_holes(tmp), dtype = 'uint8')
            kernel = np.ones((3,3), np.uint8)
            image_erotion = cv2.erode(image_filled, kernel, iterations=1)
            image_dilation = cv2.dilate(image_erotion, kernel, iterations=9)
            image_closing = cv2.erode(image_dilation, kernel, iterations=8)
            
            image_filled_2 =  np.asarray(ndi.binary_fill_holes(image_closing), dtype = 'uint8')
            image_erotion_2 = cv2.erode(image_filled_2, kernel, iterations=3)
            image_dilation_2 = cv2.dilate(image_erotion_2, kernel, iterations=3)
            
            image[:,:,j] = image_dilation_2
        key_image[i] = image
    return key_image

def image_processing_1(fig_prediksi) :
    shape = fig_prediksi.shape
    image = np.zeros(shape)
    for j in range(shape[-1]) :
        image_filled =  np.asarray(fig_prediksi[:, :, j], dtype = 'uint8')
        kernel = np.ones((3,3), np.uint8)
        image_erotion = cv2.erode(image_filled, kernel, iterations=9)
        image_dilation = cv2.dilate(image_erotion, kernel, iterations=9)
        image[:,:,j] = image_dilation
    return image

def surfd(p, t, sampling):
    def pembuat_kontur(fig_img) :
        contours = {}
        segmen = np.zeros(fig_img.shape, dtype=int)
        fig_img = np.asarray(fig_img, dtype = 'uint8')
        #ret, kontur = cv2.threshold(fig_img, 0, 255, cv2.THRESH_BINARY)
        kontur = fig_img
        for i in range(kontur.shape[-1]) :    # i adalah indeks slice
            contours[i], hierarchy = cv2.findContours(image=kontur[:,:,i], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            for j in range(len(contours[i])) :  #j adalah indeks kontur dalam tiap slice
                xy = np.array([contours[i][j][:,0,1], contours[i][j][:,0,0]])
                segmen[xy[0,:], xy[1,:], i] = 1
        return segmen
    
    def buat_gambar_prediksi(img, slices) :
        padding_xy = int((img.shape[0]-512)/2)
        if img.shape[-1] > slices :
            padding_z = int((img.shape[-1]-slices)/2)
        else:
            padding_z = 0
        #print("Padding_Z = ", padding_z)
        img = img[padding_xy:img.shape[0]-padding_xy, padding_xy:img.shape[1]-padding_xy, padding_z:img.shape[2]-padding_z]    
        return img
    #print('Sampling = ', sampling)
    img_g = t 
    img_p = p 
    
    input_1 = np.atleast_1d(img_p.astype(np.bool))
    input_2 = np.atleast_1d(img_g.astype(np.bool))
    S = pembuat_kontur(input_1)
    Sprime = pembuat_kontur(input_2)
    dta = morphology.distance_transform_edt(~img_p,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
    return sds