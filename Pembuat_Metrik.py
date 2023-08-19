# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:02:33 2023

@author: hutom
"""
import numpy as np
import os
import glob

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

def pengumpul_data(Path):
    data = {}
    i = 0;
    for file in Path :
        print(file)
        d = np.load(file, allow_pickle=True)
        data[i] = d['arr_0']
        i = i+1
    return data

def pengumpul_metrik(data) :
    dsc = ["DSC 3D"]
    for i in range(len(data)) :
        if data[i]==0 :
            continue
        dsc_avg = []
        for j in range(len(data[i])) :
            dsc_avg.append(data[i][j][2])
        avg = np.round(np.mean(dsc_avg), 3)
        sd = np.round(np.std(dsc_avg), 3)
        dsc_string = str(avg) + " " + u"\u00B1" + " " + str(sd)
        dsc.append(dsc_string)
    hd = ["HD95"]          
    for i in range(len(data)) :
        if data[i]==0 :
            continue
        hd_avg = []
        for j in range(len(data[i])) :
            hd_avg.append(data[i][j][3])    
        avg = np.round(np.mean(hd_avg)*0.01, 3)
        sd = np.round(np.std(hd_avg)*0.01, 3)
        hd_string = str(avg)+ " " + u"\u00B1" + " " + str(sd)
        hd.append(hd_string)
    metrik = [dsc, hd]
    return metrik

def get_Data(data_boxplot, data_boxplot_hd, key_s, metrik, judul) :   
    print(" ")
    mean = ['Means']
    for i in range(len(data_boxplot['means'])) :
        temp = data_boxplot['means'][i].get_data()
        mean.append(np.round(temp[1],3)[0])
    mean_hd = ['Means']
    for i in range(len(data_boxplot_hd['means'])) :
        temp = data_boxplot_hd['means'][i].get_data()
        mean_hd.append(np.round(temp[1],3)[0])
    print(mean)
    
    print(" ")
    terkecil = ['Terkecil']
    for i in range(0, len(data_boxplot['caps']), 2) :
        temp = data_boxplot['caps'][i].get_data()
        terkecil.append(np.round(temp[1],3)[0])
    terkecil_hd = ['Terkecil']
    for i in range(0, len(data_boxplot_hd['caps']), 2) :
        temp = data_boxplot_hd['caps'][i].get_data()
        terkecil_hd.append(np.round(temp[1],3)[0])
    print(terkecil)
    
    print(" ")
    kuartil_1 = ['Q1']
    for i in range(len(data_boxplot['boxes'])) :
        temp = data_boxplot['boxes'][i].get_data()
        kuartil_1.append(np.round(temp[1],3)[0])
    kuartil_1_hd = ['Q1']
    for i in range(len(data_boxplot_hd['boxes'])) :
        temp = data_boxplot_hd['boxes'][i].get_data()
        kuartil_1_hd.append(np.round(temp[1],3)[0])
    print(kuartil_1)
    
    print(" ")
    median = ['Median']
    for i in range(len(data_boxplot['medians'])) :
        temp = data_boxplot['medians'][i].get_data()
        median.append(np.round(temp[1],3)[0])
    median_hd = ['Median']
    for i in range(len(data_boxplot_hd['medians'])) :
        temp = data_boxplot_hd['medians'][i].get_data()
        median_hd.append(np.round(temp[1],3)[0])
    print(median)
    
    print(" ")
    kuartil_3 = ['Q3']
    for i in range(len(data_boxplot['boxes'])) :
        temp = data_boxplot['boxes'][i].get_data()
        kuartil_3.append(np.round(temp[1],3)[2])
    kuartil_3_hd = ['Q3']
    for i in range(len(data_boxplot_hd['boxes'])) :
        temp = data_boxplot_hd['boxes'][i].get_data()
        kuartil_3_hd.append(np.round(temp[1],3)[2])
    print(kuartil_3)
    
    print(" ")
    terbesar = ['Terbesar']
    for i in range(1, len(data_boxplot['caps']), 2) :
        temp = data_boxplot['caps'][i].get_data()
        terbesar.append(np.round(temp[1],3)[0])
    terbesar_hd = ['Terbesar']
    for i in range(1, len(data_boxplot_hd['caps']), 2) :
        temp = data_boxplot_hd['caps'][i].get_data()
        terbesar_hd.append(np.round(temp[1],3)[0])
    print(terbesar)
    
    print(" ")
    outliers = ["Outliers"]
    for i in range(len(data_boxplot['fliers'])) :
        temp = data_boxplot['fliers'][i].get_data()
        outliers.append(len(temp[1]))
    outliers_hd = ["Outliers"]
    for i in range(len(data_boxplot_hd['fliers'])) :
        temp = data_boxplot_hd['fliers'][i].get_data()
        outliers_hd.append(len(temp[1]))
    print(outliers)
    print(" ")
    # mydata = [patch, W_S_, 
    #           mean, terkecil, kuartil_1, median, kuartil_3, terbesar, outliers,
    #           pembatas,
    #           mean_hd, terkecil_hd, kuartil_1_hd, median_hd, kuartil_3_hd, terbesar_hd, outliers_hd]
    # tabel = tabulate(mydata, tablefmt="fancy_grid", stralign='center') 
    # print(tabel)    
    # print(" ")
    data_tabel = pd.DataFrame({'1': key_s[0], '2': key_s[1], 
                               '3': metrik[0], '4': mean,
                               '5': terkecil, '6': kuartil_1, 
                               '7': median, '8': kuartil_3, 
                               '9': terbesar, '10': outliers,
                               'p': key_s[2], 
                               '11': metrik[1], '12': mean_hd,
                               '13': terkecil_hd, '14': kuartil_1_hd, 
                               '15': median_hd,'16': kuartil_3_hd, 
                               '17': terbesar_hd, '18': outliers_hd, 
                               '19' : "Ketebalan Patch 12; W=Weight Loss; B=Batch;  = Weight class 0"})
    data_tabel = data_tabel.T
    datatoexcel = pd.ExcelWriter(judul)
    data_tabel.to_excel(datatoexcel)
    datatoexcel.save()
    print('DataFrame is written to Excel File successfully.')
    
def pembuat_boxplot_DSC_HD(jumlah, label, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, key_s, metrik, judul) :
    d_1 = []
    d_2 = []
    d_3 = []
    d_4 = []
    d_5 = []
    d_6 = []
    d_7 = []
    d_8 = []
    d_9 = []
    d_10 = []
    key_d = {0:d_1, 1:d_2, 2:d_3, 3:d_4, 4:d_5, 5:d_6, 6:d_7, 7:d_8, 8:d_9, 9:d_10}
    key_data = {0:data_1, 1:data_2, 2:data_3, 3:data_4, 4:data_5, 5:data_6, 6:data_7, 7:data_8, 8:data_9, 9:data_10}
    #fig = plt.figure(figsize=(5,7)) #Variasi Patch
    fig = plt.figure(figsize=(19,14)) #Variasi Batch
    ax_dsc = fig.add_axes([0, 0, 1, 1])
    data = []
    for i in range(jumlah) :
        for j in range(12) :
            key_d[i] = key_d[i] + key_data[i][j][0]
        #print("Data DSC : ", len(key_d[i]))
        data.append(np.asarray(key_d[i]))
    
    ax_dsc.set_xticklabels(label)
    bp_DSC = ax_dsc.boxplot(data, showmeans=True, meanline=True)
    plt.yticks((np.arange(0, 1.1, 0.1)), fontsize=23)
    plt.xticks(fontsize=21)
    plt.show()
    
    d_1 = []
    d_2 = []
    d_3 = []
    d_4 = []
    d_5 = []
    d_6 = []
    d_7 = []
    d_8 = []
    d_9 = []
    d_10 = []
    key_d = {0:d_1, 1:d_2, 2:d_3, 3:d_4, 4:d_5, 5:d_6, 6:d_7, 7:d_8, 8:d_9, 9:d_10}
    key_data = {0:data_1, 1:data_2, 2:data_3, 3:data_4, 4:data_5, 5:data_6, 6:data_7, 7:data_8, 8:data_9, 9:data_10}
    #fig_hd = plt.figure(figsize=(5,7)) #Variasi Patch
    fig_hd = plt.figure(figsize=(19,14)) #Variasi Batch
    ax_hd = fig_hd.add_axes([0, 0, 1, 1])
    data = []
    for i in range(jumlah) :
        for j in range(12) :
            key_d[i] = key_d[i] + key_data[i][j][1]
        #print("Data DSC : ", len(key_d[i]))
        data.append(np.asarray(key_d[i]))
    ax_hd.set_xticklabels(label)    
    bp_HD = ax_hd.boxplot(data, showmeans=True, meanline=True)
    plt.yticks((np.arange(0, 7, 1)), fontsize=23)
    plt.xticks(fontsize=21)
    plt.show()
    
    get_Data(bp_DSC, bp_HD, key_s, metrik, judul)    

'''
#J Terbaik
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_15_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_120_15\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_15_W=01\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_4 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_30\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_5 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_120_42_11_W=2.5\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_6 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_30_31_W=4\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
'''
'''
#J Variasi Patch dan sampel
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_15_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_120_15\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_15_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
'''
'''
# J Variasi Batch
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_15_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_30_31_W=10\\metrik_200",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_15_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_4 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_5 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_15_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_6 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_30_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_7 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_15_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_8 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
'''
'''
#J Variasi Weight
# data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_15_11_W=15\\metrik",
#                                                 'Data_Metrik_Pasien_*.npz')) )
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_15_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_15_11_W=5\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_15_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_4 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_96_15_11_W=01\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_8 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_15_11_W=01\\metrik",
                                               'Data_Metrik_Pasien_*.npz')) )
data_6 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_15_11_W=5\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_5 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_15_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_7 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Jantung\\Prediksi_J_144_15_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
'''
'''
#ESP Terbaik
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_120_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_30_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_30_11_W=2\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_4 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_18_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_5 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_18_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_6 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_15_11_W=2\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
'''

#ESP Variasi Weight Loss
# data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_30_11_W=20\\metrik",
#                                                 'Data_Metrik_Pasien_*.npz')) )
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_30_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_30_11_W=2\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_4 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_30_31_W=2\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_8 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_30_11_W=01_FTuned\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_7 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_30_11_W=2\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_6 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_30_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_5 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
# data_10 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_30_11_W=20\\metrik",
#                                                 'Data_Metrik_Pasien_*.npz')) )


'''
# ESP Variasi Patch
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_120_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
'''
'''
#ESP variasi sample
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_30_31_W=2\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_45_31_W=2\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_48_31_W=2_Balanced\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_4 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_30_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_5 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_45_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_6 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_48_31_W=10_Balanced\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
'''

'''
#ESP variasi sample 2
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_18_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_30_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_15_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_4 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_96_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_7 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_15_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_8 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_5 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_18_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_6 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\ESP\\Prediksi_ESP_144_30_31_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
'''
'''
#p_terbaik
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Paru\\Prediksi_P_96_15_11_W=10\\metrik",
                                               'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Paru\\Prediksi_P_120_15_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Paru\\Prediksi_P_144_15_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_4 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Paru\\Prediksi_P_144_30_11_W=10\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_5 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Paru\\Prediksi_P_96_30_31_W=4\\metrik_200",
                                                'Data_Metrik_Pasien_*.npz')) )
data_6 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Paru\\Prediksi_P_120_30_31_W=8\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_7 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Paru\\Prediksi_P_144_27_31_W=8\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
'''
'''
#p_variasi_patch
data_1 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Paru\\Prediksi_P_96_15\\metrik",
                                               'Data_Metrik_Pasien_*.npz')) )
data_2 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Paru\\Prediksi_P_120_15\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
data_3 = pengumpul_data(glob.glob(os.path.join("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\Paru\\Prediksi_P_144_15\\metrik",
                                                'Data_Metrik_Pasien_*.npz')) )
'''
#key_m = {0: data_1, 1: data_2, 2: data_3, 3: data_4, 4: data_5, 5: data_6, 6: data_7, 7: data_8, 8:data_9, 9:data_10}
#key_m = {0: data_1, 1: data_2, 2: data_3, 3: data_4, 4: data_5, 5: data_6, 6: data_7, 7: data_8, 8:data_9}
key_m = {0: data_1, 1: data_2, 2: data_3, 3: data_4, 4: data_5, 5: data_6, 6: data_7, 7: data_8, 8:0}
#key_m = {0: data_1, 1: data_2, 2: data_3, 3: data_4, 4: data_5, 5: data_6, 6: data_7, 7: 0}
#key_m = {0: data_1, 1: data_2, 2: data_3, 3: data_4, 4: data_5, 5: data_6, 6: 0, 7: 0, 8: 0, 9:0}
#key_m = {0: data_1, 1: data_2, 2: data_3, 3: data_4, 4: data_5, 5: 0, 6: 0, 7: 0}
#key_m = {0: data_1, 1: data_2, 2: data_3, 3: data_4, 4: 0, 5: 0, 6: 0}
#key_m = {0: data_1, 1: data_2, 2: data_3, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
#key_m = {0: data_1, 1: data_2, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
metrik = pengumpul_metrik(key_m)

judul = 'ESP_Variasi_WeightLoss_2.xlsx'
jumlah = 8
label = ['96,30,[1,11]', '96,30,[1,3.67]', '96,30,[1,3]', '96,30,[1,1]', '144,30,[1,11]', '144,30,[1,3.67]', '144,30,[1,3]', '144,30,[1,1.1]']
patch  = ['Ukuran Patch','96 X 96', '96 X 96', '96 X 96', '96 X 96', '144 X 144', '144 X 144', '144 X 144', '144 X 144']
W_S_ = ['W, B, ', '30,[1,11]', '30,[1,3.67]', '30,[1,3]', '30,[1,1]', '30,[1,11]', '30,[1,3.67]', '30,[1,3]', '30,[1,1.1]']
pembatas = [' ', ' ', ' ', ' ', ' ',' ',' ',' ',' ']

# judul = 'ESP_Variasi_WeightLoss_2.xlsx'
# jumlah = 10
# label = ['96,30,[1,21]', '96,30,[1,11]', '96,30,[1,3.67]', '96,30,[1,3]', '96,30,[1,1]', '144,30,[1,1.1]', '144,30,[1,3]', '144,30,[1,3.67]', '144,30,[1,11]', '144,30,[1,21]']
# patch  = ['Ukuran Patch', '96 X 96', '96 X 96', '96 X 96', '96 X 96', '96 X 96', '144 X 144', '144 X 144', '144 X 144', '144 X 144', '144 X 144']
# W_S_ = ['W, B, ', '30,[1,21]', '30,[1,11]', '30,[1,3.67]', '30,[1,3]', '30,[1,1]', '30,[1,1.1]', '30,[1,3]', '30,[1,3.67]', '30,[1,11]', '30,[1,21]']
# pembatas = [' ', ' ', ' ', ' ', ' ', ' ', ' ',' ',' ',' ',' ']

# judul = 'ESP_Terbaik_2.xlsx'
# jumlah = 6
# label = ['120,30,[1, 11]', '144,30,[1, 3]', '144,30,[1, 3.67]', '96,18,[1, 3.67]', '144,18,[1, 3.67]','144,15,[1, 3]']
# patch  = ['Ukuran Patch', '120 X 120', '144 X 144', '144 X 144', '96 X 96', '144 X 144', '144 X 144']
# W_S_ = ['W, B, ', '30,[1, 11]', '30,[1, 3]', '30,[1, 3.67]', '18,[1, 3.67]', '18,[1, 3.67]', '15,[1, 3]']
# pembatas = [' ', ' ', ' ', ' ', ' ', ' ', ' ']


# jumlah = 3
# judul = 'ESP_Variasi_Patch_1.xlsx'
# label = ['96,30,[1,11]', '120,30,[1,11]', '144,30,[1,11]']
# patch  = ['Ukuran Patch', '96 X 96', '120 X 120', '144 X 144']
# W_S_ = ['W, B, ', '30,[1,11]', '30,[1,11]', '30,[1,11]']
# pembatas = [' ', ' ', ' ', ' ']

# judul = 'ESP_Variasi_Batch_1.xlsx'
# jumlah = 8
# label = ['96,18,[1, 3.67]', '96,30,[1, 3.67]','96,15,[1, 11]', '96,30,[1, 11]', '144,18,[1, 3.67]', '144,30,[1, 3.67]', '144,15,[1, 11]', '144,30,[1, 11]']
# patch  = ['Ukuran Patch', '96 X 96', '96 X 96','96 X 96', '96 X 96', '144 X 144', '144 X 144', '144 X 144', '144 X 144']
# W_S_ = ['W, B', '18,[1, 3.67]', '30,[1, 3.67]', '15,[1, 11]', '30,[1, 11]', '18,[1, 3.67]', '30,[1, 3.67]', '15,[1, 11]', '30,[1, 11]']
# pembatas = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

# judul = 'ESP_Variasi_Sampel_1.xlsx'
# jumlah = 2
# label = ['144,10,15,11', '144,10,30,11']
# patch  = ['Ukuran Patch', '144 X 144', '144 X 144']
# W_S_ = ['W, B, ', '10,15,[1,1]', '10,30,[1,1]']
# pembatas = [' ', ' ', ' ']

# judul = 'J_Variasi_Patch.xlsx'
# jumlah = 3
# label = ['96,15,[1,11]', '120,15,[1,11]', '144,15,[1,11]']
# patch  = ['Ukuran Patch', '96 X 96', '120 X 120', '144 X 144']
# W_S_ = ['W, B, ', '15,[1,11]', '15,[1,11]', '15,[1,11]']
# pembatas = [' ', ' ', ' ', ' ']

# judul = 'J_Variasi_Batch_2.xlsx'
# jumlah = 8
# label = ['96,15,[1, 3.67]', '96,30,[1, 3.67]','96,15,[1,11]', '96,30,[1,11]', '144,15,[1,3.67]', '144,30,[1,3.67]', '144,15,[1,11]', '144,30,[1,11]']
# patch  = ['Ukuran Patch', '96 X 96', '96 X 96', '96 X 96', '96 X 96', '144 X 144', '144 X 144', '144 X 144', '144 X 144']
# W_S_ = ['W, B, ','15,[1, 3.67]','30,[1, 3.67]','15,[1, 11]','30,[1, 11]', '15,[1, 3.67]', '10,30,[1, 3.67]', '15,[1, 11]', '10,30,[1, 11]']
# pembatas = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

# judul = 'J_Variasi_Weight_1.xlsx'
# jumlah = 8
# label = ['96,15,[1,11]', '96,15,[1,6]', '96,15,[1,3.67]', '96,15,[1,1.1]', '144,15,[1,11]', '144,15,[1,6]','144,15,[1,3.67]', '144,15,[1,1.1]']
# patch  = ['Ukuran Patch', '96 X 96', '96 X 96', '96 X 96', '96 X 96', '144 X 144', '144 X 144', '144 X 144', '144 X 144']
# W_S_ = ['W, B, ', '15,[1,11]', '15,[1,6]', '15,[1, 3.7]', '15,[1, 1.1]', '15,[1,11]', '15,[1,6]', '15,[1,3.67]', '15,[1, 1.1]']
# pembatas = [' ',' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

# judul = 'J_Variasi_.xlsx'
# jumlah = 3
# label = ['96,10,30,13', '96,10,30,11', '144,10,30,31']
# patch  = ['Ukuran Patch', '96 X 96', '96 X 96', '96 X 96']
# W_S_ = ['W, B, ', '10,15,[1,3]', '10,15,[1,1]', '10,15,[3,1]']
# pembatas = [' ', ' ', ' ', ' ']

# judul = 'J_Terbaik.xlsx'
# jumlah = 6
# label = ['96,15,[1, 3.37]', '120,15,[1,11]', '144,15,[1, 1.1]', '144,30,[1,11]', '120,42,[1, 3.5]', '96,30,1,[1.7]']
# patch  = ['Ukuran Patch', '96 X 96', '120 X 120', '144 X 144', '144 X 144', '120 X 120', '96 X 96']
# W_S_ = ['W, B, ','15,[1, 3.37]', '10,15,[1,11]', '10,15,[1, 1.1]', '30,[1,11]', '42,[1, 3.5]', '30,1,[1.7]']
# pembatas = [' ', ' ', ' ', ' ', ' ', ' ', ' ']

# judul = 'J_Terbaik_1.xlsx'
# jumlah = 7
# label = ['96,15,[1, 3.37]', '120,15,[1,11]', '144,15,[1, 1.1]', '144,30,[1,11]', '120,42,[1, 3.5]', '96,30,[1, 1.67]', '96,30,[1, 3.67]']
# patch  = ['Ukuran Patch', '96 X 96', '120 X 120', '144 X 144', '144 X 144', '120 X 120', '96 X 96', '96 X 96']
# W_S_ = ['W, B, ','15,[1, 3.37]', '10,15,[1,11]', '10,15,[1, 1.1]', '30,[1,11]', '42,[1, 3.5]', '30,[1, 1.67]', '30,[1, 3.37]']
# pembatas = [' ',' ', ' ', ' ', ' ', ' ', ' ', ' ']

# judul = 'P_Terbaik.xlsx'
# jumlah = 7
# label = ['96,15,[1, 11]', '120,15,[1, 11]', '144,15,[1, 11]', '144,30,[1, 11]', '96,30,[1, 1.67]', '120,30,[1, 3]', '144,27,[1, 3]']
# patch  = ['Ukuran Patch', '96 X 96', '120 X 120', '144 X 144', '144 X 144', '96 X 96', '120 X 120', '144 X 144']
# W_S_ = ['W, B, ','15,[1,11]', '15,[1,11]', '15,[1,11]', '30,[1,11]', '30,[1, 1.67]', '30,[1, 3]', '27,[1, 3]']
# pembatas = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

# judul = 'P_Variasi_Patch_1.xlsx'
# jumlah = 3
# label = ['96,30,[1,11]', '120,30,[1,11]', '144,30,[1,11]']
# patch  = ['Ukuran Patch', '96 X 96', '120 X 120', '144 X 144']
# W_S_ = ['W, B, ', '30,[1,11]', '30,[1,11]', '30,[1,11]']
# pembatas = [' ', ' ', ' ', ' ']

key_s = {0:patch, 1:W_S_, 2:pembatas}
#pembuat_boxplot_DSC_HD(jumlah, label, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, key_s, metrik, judul)
#pembuat_boxplot_DSC_HD(jumlah, label, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9,0, key_s, metrik, judul)
pembuat_boxplot_DSC_HD(jumlah, label, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, 0,0, key_s, metrik, judul)
#pembuat_boxplot_DSC_HD(jumlah, label, data_1, data_2, data_3, data_4, data_5, data_6, data_7,0, 0, 0, key_s, metrik, judul)
#pembuat_boxplot_DSC_HD(jumlah, label, data_1, data_2, data_3, data_4, data_5, data_6, 0, 0, 0, 0, key_s, metrik, judul)
#pembuat_boxplot_DSC_HD(jumlah, label, data_1, data_2, data_3, data_4, data_5,0, 0, 0, 0, 0, key_s, metrik, judul)
#pembuat_boxplot_DSC_HD(jumlah, label, data_1, data_2, data_3, data_4, 0, 0, 0, 0, 0,0, key_s, metrik, judul)
#pembuat_boxplot_DSC_HD(jumlah, label, data_1, data_2, data_3, 0, 0, 0, 0, 0, 0, 0, key_s, metrik, judul)
#pembuat_boxplot_DSC_HD(jumlah, label, data_1, data_2, 0, 0, 0, 0, 0, 0, 0, key_s, metrik, judul)

'''
print(" ")
for i in range(len(data_boxplot['means'])) :
    print("means = ", data_boxplot['means'][i].get_data())
    
print(" ")
for i in range(len(data_boxplot['medians'])) :
    print("medians= ", data_boxplot['medians'][i].get_data())


print(" ")
for i in range(len(data_boxplot['boxes'])) :
    print("Kuartil= ", data_boxplot['boxes'][i].get_data())

print(" ")
for i in range(len(data_boxplot['caps'])) :
    print("Caps= ", data_boxplot['caps'][i].get_data())
'''