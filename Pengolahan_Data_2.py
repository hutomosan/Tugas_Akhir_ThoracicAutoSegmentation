# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:30:49 2023

@author: hutom
"""
from Image_Pre_and_Post_Processing import*
from Input import*
import re
def dsc_and_hd(im1, im2, pixel_spacing, empty_score = 1.0) :
    
    def penghitung_dsc(prediksi, segmentasi) :
        if prediksi.shape != segmentasi.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
        im_sum = prediksi.sum() + segmentasi.sum()
        if im_sum == 0:
            return 1
        intersection = np.logical_and(prediksi, segmentasi)     # Compute the truth value of x1 AND x2 element-wise.
        dsc = 2. * intersection.sum() / im_sum
        return dsc
    
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    dsc_avg = penghitung_dsc(im1, im2)
    
    dsc_slice = []
    for j in list(range(im1.shape[-1])) :
        dsc_slice.append(penghitung_dsc(im1[:,:,j], im2[:,:,j])) 
    dsc = dsc_slice 
    
    hd_slice = []
    for j in list(range(im1.shape[-1])) :
        temp = max(dist.directed_hausdorff(im1[:,:,j], im2[:,:,j])[0], dist.directed_hausdorff(im2[:,:,j], im1[:,:,j])[0])
        hd_slice.append(temp) 
    hd = hd_slice  
    
    sds = surfd(im1, im2, pixel_spacing)
    hd_95 = np.percentile(sds, 95)
    msd = sds.mean()
    msd_rms = np.sqrt((sds**2).mean())
    return [dsc, hd, dsc_avg, hd_95, msd, msd_rms]

def pengolah_sd(d_metrik) :
    dsc = d_metrik[0]
    hd = d_metrik[1]
    
    dsc_avg = d_metrik[2]
    hd_avg = d_metrik[3]
    
    dsc_sd = np.std(np.asarray(dsc))
    hd_sd = np.std(np.asarray(hd))
    return [dsc, hd, [dsc_avg, dsc_sd], [hd_avg, hd_sd]]

def pengolah_metrik(test_dataset_path, H_Folder_Path, Organ, post_p) :
    g_folders_path = os.path.join(test_dataset_path, "LCTSC")
    g_folders_path_list = glob.glob(os.path.join(g_folders_path, 'LCTSC-*')) #path setiap file dcm 
    
    g_folders = os.listdir(g_folders_path)
    g_folders.remove('LICENSE')
    #print("List g_files = \n", g_folders)
    
    p_files = os.listdir(H_Folder_Path)
    p_files.remove('Waktu.html')
    p_files.remove('metrik')
    p_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    #print("List p_files = \n", p_files)
    
    if len(g_folders) != len(p_files) :
        print("Eror bro : Jumlah file tidak setara")
        
    def buat_gambar_prediksi(img, slices) :
        padding_xy = int((img.shape[0]-512)/2)
        if img.shape[-1] > slices :
            print("Panjang img > Slices")
            padding_z = int((img.shape[-1]-slices)/2)
        else :
            padding_z = 0
        img = img[padding_xy:img.shape[0]-padding_xy, padding_xy:img.shape[1]-padding_xy, padding_z:img.shape[2]-padding_z]    
        return img
    
    for itr in range(len(g_folders)) :
        '''Resolusi Voxel'''
        modalitas = 'modalitas 1'
        slices = []
        dicom_series = glob.glob(os.path.join(g_folders_path_list[itr], modalitas, '*.dcm')) #path setiap file dcm 
        dicom_series.remove(os.path.join(g_folders_path_list[itr], modalitas, 'RTStruct.dcm'))
        for CT_file in dicom_series:                                        
            slices.append(pydicom.dcmread(CT_file, force=True))             
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))                    
        pixel_spacing= [float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]),
                        slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]] 
        print("pixel_spacing = ", pixel_spacing)
            
        '''Membuat Ground_Truth'''
        folder = g_folders[itr]
        print("Folder Test = ", folder)
        pasien_path = os.path.join(g_folders_path, folder)
        case = dicomSubject(subject_folder_path=pasien_path, masking_status=True, Organ=Organ)
        image = case.image['modalitas 1']
        seg = case.segmentation
        
        ekstrim = case.plane_idx
        true_length = np.max(ekstrim)-np.min(ekstrim) + 1
        img_g = seg[:,:,np.min(ekstrim):np.max(ekstrim)+1]
        
        '''Membuat Prediction'''
        print("Nama file prediksi = ", p_files[itr])
        f = np.load(os.path.join(H_Folder_Path, p_files[itr]), allow_pickle = True)
        prediction = buat_gambar_prediksi(f['arr_0'], true_length)
        
        if Organ == "ESP" :
            bagi = 0.5 #0.36 #0.72 #0.5
        else:
            bagi = 0.5
            
        img_p = np.zeros(prediction.shape, dtype = 'uint8')
        for z in range(prediction.shape[-1]) :
            temp_1 = np.asarray(prediction[:,:,z], dtype = 'uint8')
            ret2, temp_2 = cv2.threshold(temp_1, 
                                   np.max(prediction[:,:,z])*bagi,
                                   np.max(prediction[:,:,z]),
                                   cv2.THRESH_BINARY)
            img_p[:,:,z] = temp_2
        if np.mean(img_p) == 0 :
            img_p = img_p > np.max(img_p)*0.33
            print("Mean dari Prediksi = 0")
            print("    G-Pred = ",  img_p.shape)
        elif post_p==True :
            print("Melakukan post_processing pada prediksi")
            img_p = gambar_mask(img_p, fill_holes=True)
            print("    G-Pred = ",  img_p.shape)
        else :
            print("Tidak ada masalah")
            print("    G-Pred = ",  img_p.shape)
    
        if img_p.shape != img_g.shape :
            print("img_p.shape = ", img_p.shape)
            print("img_g.shape = ", img_g.shape)
            print("Eror bro : Ukuran G dan P tidak sama")
        
        '''Kita telah mendapatkan img_g dan img_p'''
        path_save = os.path.join(H_Folder_Path, 'metrik', 'Data_Metrik_Pasien_%d' %int(itr) + '.npz') 
        data_metric = dsc_and_hd(img_p, img_g, pixel_spacing)   
        np.savez(path_save, data_metric)

test_dataset_path   = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\'
Folders_OAR = os.listdir(test_dataset_path)
Folders_OAR.remove('LCTSC')
print ("Daftar OAR = \n", Folders_OAR)
for OAR in Folders_OAR :
    if OAR != "Jantung" :
        print("kontinue")    
        continue
    OAR_Folders_Paths = os.path.join(test_dataset_path, OAR)
    OAR_Hyperparameters = os.listdir(OAR_Folders_Paths)
    print(" ")
    print("Folder OAR = ", OAR)
    print("Isi = \n", OAR_Hyperparameters)
    for Hyperparameter in OAR_Hyperparameters :
        H_Folder_Path = os.path.join(OAR_Folders_Paths, Hyperparameter)
        print(" ")
        if os.path.basename(H_Folder_Path) != "Prediksi_J_96_30_31_W=10" :
        #if os.path.basename(H_Folder_Path) != "Prediksi_J_96_30_31_W=4" :
             print("kontinue")        
             continue
        print ("Folder Hyperparameter = ", Hyperparameter)
        Organ = OAR
        post_p = True
        pengolah_metrik(test_dataset_path, H_Folder_Path, Organ, post_p)


'''
import os                                                                       #It returns a dictionary having user’s environmental variable as key and their values as value
import numpy as np  
from Output import*
training_dataset_path   = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\'
name_1 = "Data_Akhir_1"
name_2 = "Data_Akhir_2"

data_gabungan = penyambung_data(name_1, name_2)
pembuat_kurva(data_gabungan[0], data_gabungan[1], data_gabungan[2])
'''
'''
H_Folder_Path = "C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Prediksi_J_96_15\\metrik"
Organ = "Jantung"
post_p = True
pengolah_metrik(test_dataset_path, H_Folder_Path, Organ, post_p)
'''

'''
komparasi_var_patch(g_ct, g_pred, g_truth, data_metric[0], data_metric[2])

komparasi_var_loss(g_ct, g_pred, g_truth, data_metric[0], data_metric[2])

komparasi_var_sample(g_ct, g_pred, g_truth, data_metric[0], data_metric[2])
'''


'''
Arsip

 if os.path.exists(os.path.join(H_Folder_Path, 'metrik', 'Data_Metrik_Pasien_%d' %int(itr) + '.npz')):
     print("File sudah ada,  tambahkan data sd")
     d_metrik = np.load(os.path.join(H_Folder_Path, 'metrik', 'Data_Metrik_Pasien_%d' %int(itr) + '.npz'), allow_pickle = True)
     d_metrik = d_metrik['arr_0']
     data_metrik_dengan_sd = pengolah_sd(d_metrik)
     
     path_save_2 = os.path.join(H_Folder_Path, 'metrik', 'Data_Metrik_Pasien_SD_0.5_%d' %int(itr) + '.npz') 
     np.savez(path_save_2, data_metrik_dengan_sd)
     continue


name_1 = "Data_Akhir_5760"
name_2 = "Data_Akhir_7200(Sambungan)"

penyambung_data(name_1, name_2)

training_dataset_path   = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\'

import os                                                                       #It returns a dictionary having user’s environmental variable as key and their values as value
import numpy as np                                                             # pydicom reads modify and write data in DICOM files
import matplotlib.pyplot as plt
f = np.load(os.path.join(training_dataset_path, "segmentasi-Heart.npz"), allow_pickle = True)
data = f['arr_0']
plot_img(data[0])

s = np.load(os.path.join(training_dataset_path, 'segmentasi-'+ 'total' +'.npz'), allow_pickle = True)
s_classes = s['arr_0']

a = s_classes
unique, counts = np.unique(a, return_counts=True)
dict(zip(unique, counts))

plot_img(s_classes)
'''





