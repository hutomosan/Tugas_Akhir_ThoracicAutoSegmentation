# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:12:19 2023

@author: hutom
"""
import os                                                                       #It returns a dictionary having user’s environmental variable as key and their values as value
import numpy as np  
#from Output import*

def penyambung_data(name_1, name_2) :    
    training_dataset_path   = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\'
    f = np.load(os.path.join(training_dataset_path, name_1 + ".npz"), allow_pickle = True)
    data_1 = f['arr_0']

    f = np.load(os.path.join(training_dataset_path, name_2 + ".npz"), allow_pickle = True)
    data_2 = f['arr_0']

    data_loss_error_train = {}
    data_loss_error_val = {}
    data_epoch = 0
    data_iterasi = 0
    for i in range(len(data_1)) :
        if i == 0 or i == 1 :
            list_temp = []
            for j in range(len(data_1[i])) :
                if i == 0 :
                    list_temp = data_1[i][j] + data_2[i][j]
                    data_loss_error_train[j] = list_temp
                if i == 1 :
                    list_temp = data_1[i][j] + data_2[i][j]
                    data_loss_error_val[j] = list_temp
            continue
        if i == 2 :
            data_epoch = data_1[i] + data_2[i]
            continue
        if i == 3 :
            data_iterasi = data_1[i] + data_2[i]
            break
        
    data_akhir_7200 = [data_loss_error_train, data_loss_error_val, data_epoch, data_iterasi] 
    np.savez(os.path.join(training_dataset_path, 'Data_Akhir_Gabungan'+'.npz'), data_akhir_7200)
    
    return data_akhir_7200

training_dataset_path   = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\'
name_1 = "Data_Akhir_Gabungan"
name_2 = "Data_Akhir_3"

data_gabungan = penyambung_data(name_1, name_2)