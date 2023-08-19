# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:16:57 2023

@author: hutom
"""

from Output import*
training_dataset_path   = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\'
i = 0
data = {}
while i < 1 : 
    d = np.load(os.path.join(training_dataset_path, 'Data_Akhir_' + str(i+1) + '.npz'), allow_pickle = True)
    #d = np.load(os.path.join(training_dataset_path, 'Data_Akhir_Gabungan.npz'), allow_pickle = True)
    data[i] = d['arr_0']
    pembuat_kurva(data[i][0], data[i][1], data[i][2])
    i = i+1

def plot_img(img) :
    for i in range(img.shape[-1]) :    
        plt.figure(figsize=(6.4, 4.8), dpi=40)
        plt.imshow(img[:,:,i], cmap='gray');
        plt.axis('off')      
        plt.show()

# training_dataset_path   = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\'
# d = np.load(os.path.join(training_dataset_path, 'prediksi_1.npz'), allow_pickle = True)
# data = d['arr_0']
# plot_img(data)




