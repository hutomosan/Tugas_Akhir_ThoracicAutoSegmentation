# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:06:39 2023

@author: Wahyu Hutomo Nugroho (hutomonugroho@gmail.com)
"""
#import os                                                                      
#os.environ['THEANO_FLAGS']="device=cuda0"
#import numpy as np
#import sys
#np.set_printoptions(threshold=sys.maxsize) 
#import matplotlib.pyplot as plt
from Image_Pre_and_Post_Processing import * 

import pydicom                                                                  
import glob
import scipy                                                                   
from scipy import ndimage
import random
import medpy
import shutil
import scipy.interpolate as si
import scipy.spatial.distance as dist

def interplote(points):                                                              
    added = []                                                                  
    for i in list(range(len(points)-1)):                                         
        dist = np.linalg.norm(np.array(points[i+1]) - np.array(points[i]))           
        if dist > 1.4:
            pair = [points[i], points[i+1]]
            if np.abs(points[i][0]-points[i+1][0]) > np.abs(points[i][1]-points[i+1][1]): 
                min_idx = np.argmin([points[i][0],points[i+1][0]])          
                xx = np.linspace(start=pair[min_idx][0], stop=pair[1-min_idx][0], num=pair[1-min_idx][0]-pair[min_idx][0]+2, dtype='int32') 
                interp = np.interp(xx, [pair[min_idx][0],pair[1-min_idx][0]], [pair[min_idx][1],pair[1-min_idx][1]]) 
                for dummy in zip(xx, interp):                              
                    added.append([int(dummy[0]),int(dummy[1])])             
            else:                                                          
                min_idx = np.argmin([points[i][1],points[i+1][1]])          
                yy = np.linspace(start=pair[min_idx][1], stop=pair[1-min_idx][1], num=pair[1-min_idx][1]-pair[min_idx][1]+2, dtype='int32')
                interp = np.interp(yy, [pair[min_idx][1],pair[1-min_idx][1]], [pair[min_idx][0],pair[1-min_idx][0]]) 
                for dummy in zip(interp,yy):                               
                    added.append([int(dummy[0]),int(dummy[1])])            
    return [list(x) for x in set(tuple(x) for x in added+points)] 

class dicomSubject(object):
    def __init__(self, subject_folder_path, masking_status, Organ):                     
        self.origin = {}
        self.pixel_spacing = {}
        self.size = {}
        self.image = {}                                       
        self.image_CLAHE = {}

        self.gambar_mentah = {}
        self.masking = {}

        self.slice_num = {}
        self.contour = None                                   
        self.temp_contours = None
        self.segmentation = None
        self.temp_segmentation = None
        self.plane_idx = []                                                     

        folder_list = os.listdir(subject_folder_path)        #list folder modalitas                            
        folder_list.sort()
        self.folder_list = os.listdir(subject_folder_path)
        self.folder_list.sort()
       
        '''----Menciptakan Dataset Gambar----'''
        for modalitas in folder_list:   #folder modalitas                                                    
            slices = []                                                   
            dicom_series = glob.glob(os.path.join(subject_folder_path, modalitas, '*.dcm')) #path setiap file dcm 
            dicom_series.remove(os.path.join(subject_folder_path, modalitas, 'RTStruct.dcm'))
            for CT_file in dicom_series:                                        
                slices.append(pydicom.dcmread(CT_file, force=True))             
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))      
            
            self.origin[modalitas] = slices[0].ImagePositionPatient              
            self.pixel_spacing[modalitas] = [slices[0].PixelSpacing[0], slices[0].PixelSpacing[1], slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]] 
    
            raw_img = np.stack([s.pixel_array for s in slices], axis=-1)        
            
            '''----Me-Load atau Membuat Masking----'''
            masking_path = os.path.join(subject_folder_path, modalitas, 'masking.npz')
            if masking_status and os.path.exists(masking_path):
                data_mentah = np.load(masking_path, allow_pickle=True)
                self.masking[modalitas] = data_mentah['arr_0']
                #print('    Selesai me-load mask')             
            elif masking_status :
                self.masking[modalitas] = membuat_mask(raw_img, 860, True)
                np.savez(os.path.join(subject_folder_path, modalitas, 'masking'+'.npz'), self.masking[modalitas])
                print('    Selesai membuat dan simpan mask')
            else :
                self.masking[modalitas] = 1
                
            '''----Me-normalisasi Dataset Gambar----'''                                                                       
            max_value = np.max(raw_img)                                         
            min_value = np.min(raw_img)                                         
            raw_img[raw_img>max_value*0.9] = max_value * 0.9                   
            raw_img = raw_img/(max_value*0.9)                                   
            self.image[modalitas] = raw_img * self.masking[modalitas]              # Gambar hasil masking !                                    
            self.size[modalitas] = self.image[modalitas].shape
            #print('    Ukuran slices : ', self.size[modalitas], ' Pixel Spacing : ', self.pixel_spacing[modalitas], ' Origin : ', self.origin[modalitas])
            #print('    Jumlah slices = ', len(slices), ' nilai max raw = ', max_value, ' nilai min raw = ', min_value)    
        '''----Menciptakan Dataset Label----'''
        modalitas = 'modalitas 1'
        self.contour = np.zeros(self.size[modalitas])
        self.segmentation = np.zeros(self.size[modalitas])
        
        structure_set_file = glob.glob(os.path.join(subject_folder_path, modalitas, '*RTStruct.dcm')) 
        structure = pydicom.dcmread(structure_set_file[0], force=True)  
        plane_idx = []

        '''----Memilih nomor dan nama ROI----'''
        key_roi = [1, 2, 3, 4, 5]
        for roi in structure.ROIContourSequence :      
            number = roi.ReferencedROINumber   
            key_roi[number-1] = roi                                      
        
        key_name = [1, 2, 3, 4, 5]
        num = 0
        for roi_name in structure.StructureSetROISequence :
            name = roi_name.ROIName
            key_name[num] = name
            num = num+1   

        key_dicom = tuple(zip(key_roi, key_name))
        
        if Organ == 'Paru' :
            OAR1 = 'Lung_R'
            OAR2 = 'Lung_L'
        elif Organ == 'Jantung' :
            OAR1 = 'Heart'
            OAR2 = 'Null'
        elif Organ == "ESP" :
            OAR1 = 'Esophagus'
            OAR2 = 'SpinalCord'
        print("Organ = ", Organ)    
        for key in key_dicom :
            if key[1] == OAR1 or key[1] == OAR2 :   
            #if key[1] == 'Lung_R' or key[1] == 'Lung_L' :    
            #if key[1] == 'Heart' :
            #if key[1] == 'Esophagus' or key[1] == 'SpinalCord' :    
                print('   ', key[1])          
                '''----Membuat atau Me-load Dataset Segmentasi----'''
                segmentasi_path = os.path.join(subject_folder_path, modalitas, 'segmentasi-'+ key[1] +'.npz')
                if os.path.exists(segmentasi_path):
                    data_mentah = np.load(segmentasi_path, allow_pickle=True)
                    data_segmentasi = data_mentah['arr_0']
                    self.segmentation = data_segmentasi[0]
                    self.plane_idx = data_segmentasi[1]
                    #print('    Selesai me-load segmentasi')  
                else :
                    for plane_contour in key[0].ContourSequence:                                  
                        contour_points = list(zip(*[iter(plane_contour.ContourData)]*3))
                        z_voxel = int(round((contour_points[0][2] - self.origin[modalitas][2]) / self.pixel_spacing[modalitas][2]))           
                        test_aa = []                                                                                                                                                                               
                        for point in contour_points:                                  
                            x_voxel = int(round((point[0] - self.origin[modalitas][0]) / self.pixel_spacing[modalitas][0])) 
                            y_voxel = int(round((point[1] - self.origin[modalitas][1]) / self.pixel_spacing[modalitas][1]))  
                            test_aa.append([x_voxel,y_voxel])                                                                                    
                        test_aa.append(test_aa[0])                                     
                        temp_contour = interplote(test_aa)                             
                        temp_contour = np.array(temp_contour)
                        self.contour[temp_contour[:,1],temp_contour[:,0],z_voxel] = 1  
                        seg = ndimage.binary_fill_holes(self.contour[:,:,z_voxel])    
                        self.segmentation[:,:,z_voxel] = seg
                        plane_idx.append(z_voxel)
                        self.plane_idx = plane_idx  
                    np.savez(os.path.join(subject_folder_path, modalitas, 'segmentasi-'+ key[1] +'.npz'), [self.segmentation, self.plane_idx]) 
                    print('    Selesai membuat segmentasi')
                    #j_kontur = j_kontur + len(temp_contour)
                        #print('     Kontur Ke-', number, '  Jum. titik kontur = ', len(temp_contour), ' Jarak voxel z = ', z_voxel)
                        #x = x+1
                    #print('    ROI Ke- ', key[0].ReferencedROINumber, ' Total kontur = ', x, ' Total titik kontur = ', j_kontur)
            else :
                continue

def load_train(data_folder, m_batch_size, n_epochs, patchSize, masking_status, Organ): 
    patchX, patchY, patchZ = patchSize  
    folders = os.listdir(data_folder)
    folders.remove('LICENSE')
    case_num = len(folders) 
    print('Jumlah pasien = ', case_num)
    print(' ')
    epoch_num = 1
    print('Epoch ke-', epoch_num)
    for itr in list(range(n_epochs*case_num)):
        train_x = np.zeros([m_batch_size, 1, 2*patchX, 2*patchY, 2*patchZ])
        train_y = np.zeros([m_batch_size, 2*patchX, 2*patchY, 2*patchZ]) 
        j_epoch = itr%(case_num)
            
        if j_epoch == 0 and itr > 0 :
            random.shuffle(folders) 
            epoch_num = epoch_num+1
            print('Epoch ke-', epoch_num)
     
        folder = folders[j_epoch]   
        print("Nama Folder = ", os.path.dirname(folder))
        pasien_path = os.path.join(data_folder, folder)
        print('  Iterasi ke-', itr+1, ' Folder ', folder)
        case = dicomSubject(subject_folder_path=pasien_path, masking_status=masking_status, Organ=Organ)  
        seg = case.segmentation
        
        if masking_status == True :
            mask = case.masking['modalitas 1'] * np.invert(seg.astype(bool))
        '''----Koordinat elemen 1 dari seg----'''
        positive = np.where(seg==1)         # (array([0, 0, 0, 1, 1, 1]), array([0, 1, 2, 0, 1, 2]))                                           
        positive = list(zip(*positive))     # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        random.shuffle(positive)
        sizeX, sizeY, sizeZ = seg.shape        

        dummy_id = 0
        i = 0
        i_counted = 0
        while i < m_batch_size * 8 / m_batch_size :
            x, y, z = positive[i_counted]                                      
            if x < patchX or x > sizeX - patchX or y < patchY or y > sizeY - patchY or z < patchZ or z > sizeZ - patchZ:
                dummy_id = dummy_id + 1
                i_counted = i_counted + 1
                continue                                                       
            for idx, modality in enumerate(case.folder_list):                  
                img = case.image[modality] - 0.2                               
                train_x[i, idx, :, :, :] = img[x - patchX:x + patchX, y - patchY:y + patchY, z - patchZ:z + patchZ]
            train_y[i, :, :, :] = seg[x - patchX:x + patchX, y - patchY:y + patchY, z - patchZ:z + patchZ] 
            i = i + 1
            i_counted = i_counted + 1

        dummy_id = 0
        j = 0
        while i+j < m_batch_size:
            x,y,z = [random.randint(patchX,sizeX-patchX),                      
                     random.randint(patchY,sizeY-patchY),
                     random.randint(patchZ,sizeZ-patchZ)] 
            if masking_status == True : 
                if mask[x, y, z] == 0 :
                    dummy_id = dummy_id + 1
                    if dummy_id > 2 :
                        #print("skip bro ! ", dummy_id)
                        continue                
            for idx, modality in enumerate(case.folder_list):
                img = case.image[modality] - 0.2
                train_x[i+j,idx,:,:,:] = img[x-patchX:x+patchX,y-patchY:y+patchY,z-patchZ:z+patchZ]
            train_y[i+j,:,:,:] = seg[x-patchX:x+patchX,y-patchY:y+patchY,z-patchZ:z+patchZ]
            j = j + 1
        #print('    positive rate: ', np.mean(train_y))
        yield  train_x.astype('float32'), train_y.astype('int32'), case_num, itr+1

# data_folder_train = "C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\LCTSC"
# train_x, train_y, case_num, itr, positive = load_train(data_folder = data_folder_train, 
#                                               m_batch_size = 15, 
#                                               n_epochs=1, 
#                                               patchSize=[48,48,6], 
#                                               masking_status=True, 
#                                               Organ="ESP")
# p = positive
# train_x, train_y, case_num, itr, positive = load_train(data_folder = data_folder_train, 
#                                               m_batch_size = 15, 
#                                               n_epochs=1, 
#                                               patchSize=[72,72,6], 
#                                               masking_status=True, 
#                                               Organ="ESP",
#                                               positive=p)
# for i in range(train_y.shape[0]) :    
#         plt.figure(figsize=(6.4, 6.4), dpi=40)
#         plt.imshow(train_x[i,0,:,:,5], cmap='gray');
#         plt.axis('off')      
#         plt.show()
    
# for i in range(train_y.shape[0]) :    
#         plt.figure(figsize=(6.4, 6.4), dpi=40)
#         plt.imshow(train_y[i,:,:,5], cmap='gray');
#         plt.axis('off')      
#         plt.show()
'''----------Arsip-----------
def load_test(patchSize, dataset_path, masking_status):
    patchX, patchY, patchZ = patchSize  
    data_folder = dataset_path
    files = os.listdir(data_folder) 
    files.remove('LICENSE')
    case_num = len(files) 
    print('Jumlah pasien = ', case_num)
    for epoch in list(range(case_num)): 
        folder = files[epoch]   
        print(' ')
        print('Iterasi ', epoch, ' Folder ', folder)
        case = dicomSubject(subject_folder_path=os.path.join(data_folder, folder), masking_status = masking_status)  
        seg = case.segmentation                                               
        sizeX, sizeY, sizeZ = seg.shape                                       
        sizeZ = round(((max(case.plane_idx) - min(case.plane_idx))/patchZ)+0.5) 
        batch_size = 4*5*sizeZ      #batch_size = 6*8*sizeZ (56 X 56 X 16);  4*5*sizeZ (96 X 96 X 12); 3*4*sizeZ (120 X 120 X 8); 3*3*sizeZ (144 X 144 X 12)
        print('    batch = ', batch_size)
        test_x = np.zeros([int(batch_size), 1, patchX, patchY, patchZ]) 
        test_y = np.zeros([int(batch_size), patchX, patchY, patchZ])   
        print('    Tebal Slice = ', min(case.plane_idx), '-',max(case.plane_idx))
        b = 0
        pixel_y = 68               #pixel_y = 112 (56X56X16)   pixel_y = 88 (96X96X12)     pixel_y = 80 (120X120X8)    pixel_y = 50 (144X144X12)
        while pixel_y < 452 :       #pixel_y < 448              pixel_y < 472               pixel_y < 440               pixel_y < 482
            pixel_x = 24            #pixel_x = 56               pixel_x = 28                pixel_x = 24                pixel_x = 40
            while pixel_x < 504 :   #pixel_x < 504              pixel_x < 508               pixel_x < 504               pixel_x < 472              
                pixel_z = min(case.plane_idx)  
                while pixel_z < max(case.plane_idx) and b < batch_size :
                    #if (pixel_z + patchZ) > max(case.plane_idx) :
                        #break
                    for idx, modality in enumerate(case.folder_list):                   
                        img = case.image[modality]- 0.2                             
                        test_x[b, idx, :, :, :] = img[pixel_y:(pixel_y + patchY), pixel_x:(pixel_x + patchX), pixel_z:(pixel_z + patchZ)]
                        test_x.astype('float32')
                    test_y[b, :, :, :] = seg[pixel_y:(pixel_y + patchY), pixel_x:(pixel_x + patchX), pixel_z:(pixel_z + patchZ)]
                    test_y.astype('int32')
                    b = b + 1
                    #print('      Membuat batch : ',b, ', ', pixel_y, ', ', pixel_x,', ', pixel_z)
                    pixel_z = pixel_z + patchZ
                pixel_x = pixel_x + patchX
            pixel_y = pixel_y + patchY
        print('Total sample ', b)
        print('test_x : ', test_x.shape, ' ', 'test_y : ', test_y.shape)
        yield test_x, test_y, sizeZ
        

if case_num == 12:
    random.shuffle(files)


data_folder = "C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Test_Data_I\\LCTSC"
p = 0
sample = {}
label = {}
start_time = timeit.default_timer()
for train_x, train_y, num, itr in load_train(data_folder = data_folder, m_batch_size = 18, n_epochs=1, patchSize=[72,72,8]) :
    sample[p] = train_x
    label[p] = train_y
    p = p+1
end_time = timeit.default_timer()
print("The code ran for %.1fs" % (end_time - start_time))

for i in range(train_x.shape[-1]) :
    fig = plt.figure(dpi=80, layout = 'constrained')
    plt.imshow(train_x[11,0,:,:,i], cmap = 'gray')
    #fig = plt.figure(dpi=80, layout = 'constrained')
    #plt.imshow(train_y[14,:,:,i], cmap = 'gray')
    
    
print(' ')
folder_list.remove('T2')                                                # Menghapus folder dari path                                                         
folder_list.remove('T1')    
for folder in folder_list:                                             
    image_tmp = np.zeros(size['T1'])                                              # Menciptakan ndarray berisi nol dengan ukuran yg sama dengan array di T1                                                                             
    for sle_idx in list(range(size[folder][2])):                                  # Loop untuk sepanjang range z di data size T3 (berarti dari 0-33 )                                                                                    
        image_tmp[:,:,sle_idx] = skimage.transform.resize(image[folder][:,:,sle_idx], size['T1'][0:2], order=0, anti_aliasing=False)  # Meresize setiap gambar di setiap channel z di dalam T3 menjadi ukuran (x,y) T1 
        image[folder] = image_tmp                            
# Ini menghasilkan nilai pixel maksimum yang lebih kecil dari image asli (0,5 atau 0,6)
# Ini menghasilkan nilai pixel rata-rata yang sama dengan image asli
self.image_CLAHE = image   


modalitas = 'modalitas 1'
positive_path = os.path.join(subject_folder_path , modalitas, 'positive-'+ 'Heart' +'.npz')
if os.path.exists(positive_path):
    data_mentah = np.load(positive_path, allow_pickle=True)
    data_positive = data_mentah['arr_0']
    positive = data_positive
    print('    Selesai me-load positive = ')             
else :    
    ----Koordinat elemen 1 dari seg----
    positive = np.where(seg==1)                                            
    positive = list(zip(*positive))
    np.savez(os.path.join(subject_folder_path , modalitas, 'positive-'+ 'Heart' +'.npz'), positive)   
    print('    Selesai me-save positive = ')    
random.shuffle(positive)
sizeX, sizeY, sizeZ = seg.shape        


positive_mask_path = os.path.join(subject_folder_path , modalitas, 'positive_mask-'+ 'Heart' +'.npz')
if os.path.exists(positive_mask_path):
    data_mentah = np.load(positive_mask_path, allow_pickle=True)
    data_positive_mask = data_mentah['arr_0']
    positive_mask = data_positive_mask
    print('    Selesai me-load positive_mask = ')             
else :    
    ----Koordinat elemen 1 dari seg+mask----
    mask = case.masking['modalitas 1'] * np.invert(seg.astype(bool))
    positive_mask = np.where(mask==1)                                            
    positive_mask = list(zip(*positive_mask))
    np.savez(os.path.join(subject_folder_path , modalitas, 'positive_mask-'+ 'Heart' +'.npz'), positive_mask)   
    print('    Selesai me-save positive_mask = ') 
random.shuffle(positive_mask)   
'''