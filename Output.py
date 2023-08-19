# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:21:48 2023

@author: Wahyu Hutomo Nugroho (hutomonugroho@gmail.com)
"""
import os    
import numpy as np
import matplotlib.pyplot as plt

def pembuat_kurva(training, validation, epoch):
    def fungsi_kurva_Loss(string, x, y_train, y_valid, epoch) :
        plt.figure(figsize=(13.8, 15.6), dpi = 100)
        plt.plot(x, y_train, label = "Train"+string)
        plt.plot(x, y_valid, label = "Valid"+string)
        #plt.title(string+" Learning Curve", fontsize=30)
        plt.xlabel("Epoch", fontsize=30)
        plt.ylabel(string, fontsize=30)
        plt.xticks(np.arange(0, epoch+20, 40), fontsize=30)
        #plt.yticks(np.arange(0, 3.2, 0.2), fontsize=30)
        plt.yticks(np.arange(0, 1.3, 0.1), fontsize=30) 
        plt.legend()
        plt.grid()
        plt.show()
    def fungsi_kurva_Error(string, x, y_train, y_valid, epoch) :
        plt.figure(figsize=(13.8, 15.6), dpi = 100)
        plt.plot(x, y_train, label = "Train"+string)
        plt.plot(x, y_valid, label = "Valid"+string)
        #plt.title(string+" Learning Curve", fontsize=30)
        plt.xlabel("Epoch", fontsize=30)
        plt.ylabel(string, fontsize=30)
        plt.xticks(np.arange(0, epoch+5, 40), fontsize=30)
        #plt.yticks(np.arange(0, 31, 2), fontsize=30) 
        plt.yticks(np.arange(0, 18, 1), fontsize=30) 
        plt.legend()
        plt.grid()
        plt.show()
    
    def remover(data, k) :
        temp = np.asarray(data)
        data = temp * (temp < k)
        print(data)
        data = list(data)
        i = 0
        j = 0
        for d in data :
            if d == 0 :
                data[i] = k
                j = j+1
            i = i+1
        print("j = ", j)
        data = list(data)
        return data
    
    x = list(range(0, epoch))
    y_loss_train = training[0] 
    y_loss_valid = validation[0]
    # y_error_train = training[2]
    # y_error_valid = validation[2]
    # y_loss_train = remover(training[0], 0.45) 
    # y_loss_valid = remover(validation[0], 0.45)
    y_error_train = remover(training[2], 18)
    y_error_valid = remover(validation[2], 18)
    fungsi_kurva_Loss(string="Loss", x=x, y_train=y_loss_train, y_valid=y_loss_valid, epoch=epoch)
    fungsi_kurva_Error(string="Error", x=x, y_train=y_error_train, y_valid=y_error_valid, epoch=epoch)

def figure_maker(image, prediksi, segmentasi, dsc, hd, save_figure_path) :  
    plt.ioff() 
    key_fig = {0:image, 1:prediksi[0], 2:prediksi[1], 3:prediksi[2], 4:segmentasi}
    i = 0
    z = prediksi[i].shape[-1]
    #panjang = 654   
    panjang = 954   
    #panjang = 2237
    lebar = len(key_fig)*(panjang/z)*1  #1.25 (96X96X12) #1.333 (120X120X8) #1 (224X224X12)
    fig = plt.figure(figsize=(lebar, panjang), dpi=40, layout = None)
    sub_fig = fig.subfigures(z,1, squeeze=False, wspace = 0.07, hspace=6.0) #2.0
    for n in range(0, z) : 
        axes = sub_fig[n,0].subplots(1,5, squeeze=True, sharex = True, sharey = True)
        m = 0
        for ax in axes :
            if m == 0 :
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')
                ax.set_title('\nCT Image', {'fontsize':40}) #{'fontsize':60}
            elif m == 1 :
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')   
                ax.set_title('\n(96 X 96 X 12)' + "\n" + 'DSC = %s' %str(round(dsc[m-1][n], 3)) + "\n" + 'HD = %s' %str(round(hd[m-1][n], 3)),
                             {'fontsize':40})
            elif m == 2 :
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')   
                ax.set_title('\n(120 X 120 X 12)' + "\n" + 'DSC = %s' %str(round(dsc[m-1][n], 3)) + "\n" + 'HD = %s' %str(round(hd[m-1][n], 3)),
                             {'fontsize':40})
            elif m == 3 :
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')   
                ax.set_title('\n(224 X 224 X 12)' + "\n" + 'DSC = %s' %str(round(dsc[m-1][n], 3)) + "\n" + 'HD = %s' %str(round(hd[m-1][n], 3)),
                             {'fontsize':40}) 
            else:
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')   
                ax.set_title('\nG.Truth', {'fontsize':40})   
            ax.set_xticks(np.arange(0,512, step = 100)) #480 (120X120x8) 432 (224X224x12)
            ax.set_yticks(np.arange(0,512, step = 100)) #384 (96X96X12) #360 (120X120x8) 432 (224X224x12)
            ax.tick_params(axis='both', labelsize=30)
            m = m+1
        #sub_fig[n,0].suptitle('Slice-%s' %str(n+1), fontsize=80, fontweight = 'bold')
        sub_fig[n,0].suptitle('Slice-%s' %str(n+1), x=0.1, horizontalalignment='left', fontsize=40, fontweight = 'bold')
    fig.savefig(os.path.join(save_figure_path, "Pasien-%d.png" % (i+1)))
    plt.close(fig)

def figure_maker_var_loss(image, prediksi, segmentasi, dsc, hd, save_figure_path) :  
    plt.ioff() 
    key_fig = {0:image, 1:prediksi[2], 2:prediksi[3], 3:segmentasi}
    i = 0
    z = prediksi[i].shape[-1]
    #panjang = 654
    panjang = 954
    #panjang = 2237
    lebar = len(key_fig)*(panjang/z)*1  #1.25 (96X96X12) #1.333 (120X120X8) #1 (224X224X12)
    fig = plt.figure(figsize=(lebar, panjang), dpi=40, layout = None)
    sub_fig = fig.subfigures(z,1, squeeze=False, wspace = 0.07, hspace=6.0) #2.0
    for n in range(0, z) : 
        axes = sub_fig[n,0].subplots(1,4, squeeze=True, sharex = True, sharey = True)
        m = 0
        for ax in axes :
            if m == 0 :
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')
                ax.set_title('CT Image', {'fontsize':40})
            elif m == 1 :
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')   
                ax.set_title('(224 X 224 X 12)' + "\n" + "Weight loss = 10" + "\n" + 'DSC = %s' %str(round(dsc[m+1][n], 3)) + "\n" + 'HD = %s' %str(round(hd[m+1][n], 3)),
                             {'fontsize':40}) 
            elif m == 2 :
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')   
                ax.set_title('(224 X 224 X 12)' + "\n" + "Weight loss = 0.1" + "\n" + 'DSC = %s' %str(round(dsc[m+1][n], 3)) + "\n" + 'HD = %s' %str(round(hd[m+1][n], 3)),
                             {'fontsize':40}) 
            else:
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')   
                ax.set_title('G.Truth', {'fontsize':40})   
            ax.set_xticks(np.arange(0,512, step = 100)) #480 (120X120x8) 432 (224X224x12)
            ax.set_yticks(np.arange(0,512, step = 100)) #384 (96X96X12) #360 (120X120x8) 432 (224X224x12)
            ax.tick_params(axis='both', labelsize=30)
            m = m+1
        #sub_fig[n,0].suptitle('Slice-%s' %str(n+1), fontsize=80, fontweight = 'bold')
        sub_fig[n,0].suptitle('Slice-%s' %str(n+1),x=0.1, horizontalalignment='left', fontsize=40, fontweight = 'bold')
    fig.savefig(os.path.join(save_figure_path, "Pasien-%d.png" % (i+1)))
    plt.close(fig)

def figure_maker_var_sample(image, prediksi, segmentasi, dsc, hd, save_figure_path) :  
    plt.ioff() 
    key_fig = {0:image, 1:prediksi[2], 2:prediksi[4], 3:segmentasi}
    i = 0
    z = prediksi[i].shape[-1]
    #panjang = 654 
    panjang = 954
    #panjang = 2237
    lebar = len(key_fig)*(panjang/z)*1  #1.25 (96X96X12) #1.333 (120X120X8) #1 (224X224X12)
    fig = plt.figure(figsize=(lebar, panjang), dpi=40, layout = None)
    sub_fig = fig.subfigures(z,1, squeeze=False, wspace = 0.07, hspace=6.0) #2.0
    for n in range(0, z) : 
        axes = sub_fig[n,0].subplots(1,4, squeeze=True, sharex = True, sharey = True)
        m = 0
        for ax in axes :
            if m == 0 :
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')
                ax.set_title('CT Image', {'fontsize':40})
            elif m == 1 :
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')   
                ax.set_title('(224 X 224 X 12)' + "\n" + "Samples = 15" + "\n" + 'DSC = %s' %str(round(dsc[m+1][n], 3)) + "\n" + 'HD = %s' %str(round(hd[m+1][n], 3)),
                             {'fontsize':40}) 
            elif m == 2 :
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')   
                ax.set_title('(224 X 224 X 12)' + "\n" + "Samples = 30" + "\n" + 'DSC = %s' %str(round(dsc[m+2][n], 3)) + "\n" + 'HD = %s' %str(round(hd[m+2][n], 3)),
                             {'fontsize':40}) 
            else:
                ax.imshow(key_fig[m][:,:,n], cmap = 'gray')   
                ax.set_title('G.Truth', {'fontsize':40})   
            ax.set_xticks(np.arange(0,512, step = 100)) #480 (120X120x8) 432 (224X224x12)
            ax.set_yticks(np.arange(0,512, step = 100)) #384 (96X96X12) #360 (120X120x8) 432 (224X224x12)
            ax.tick_params(axis='both', labelsize=30)
            m = m+1
        #sub_fig[n,0].suptitle('Slice-%s' %str(n+1), fontsize=80, fontweight = 'bold')
        sub_fig[n,0].suptitle('Slice-%s' %str(n+1), x=0.1, horizontalalignment='left', fontsize=40, fontweight = 'bold')
    fig.savefig(os.path.join(save_figure_path, "Pasien-%d.png" % (i+1)))
    plt.close(fig)

'''_______Arsip_______
    
class konstruksi(object) :
    def __init__(self, data):
        self.m = 0  #Jumlah sampel dalam arah x
        self.n = 0  #Jumlah sampel dalam arah y
        self.key_temp_x = {}
        self.key_temp_y = {}
        self.key_xyz = {}
        self.stack_temp = None
        self.data = data
        
    def konstruksi_z(self, awal, akhir, axis) :
        self.stack_temp = self.data[awal]
        for i in range(awal+1, akhir) :    
            temp_z = np.concatenate(( self.stack_temp, self.data[i]), axis = axis)
            self.stack_temp = temp_z
            if i == akhir-1 :
                self.key_temp_x[self.m] = self.stack_temp
                print('key_temp_x', ' [',self.m,']', ' shape =', self.key_temp_x[self.m].shape, ' i = ', i)
                self.m = self.m+1
    
    def konstruksi_y(self, axis) :
        self.stack_temp = self.key_temp_x[0]
        for i in range(1, self.m) :
            temp_y = np.concatenate(( self.stack_temp, self.key_temp_x[i]), axis=axis)   
            self.stack_temp = temp_y
            if i == self.m-1:
                self.key_temp_y[self.n] = self.stack_temp
                print('    selesai merge baris x ke n = ', self.n, ' shape = ', self.key_temp_y[self.n].shape, ' di i = ', i)
                self.n = self.n + 1
                self.m = 0
                self.key_temp_x = {}  
       
    def konstruksi_x(self, axis) :  
        self.stack_temp = self.key_temp_y[0]
        print('key_temp_y 0 ', self.key_temp_y[0].shape)
        for i in range(1, len(self.key_temp_y)) :
            print('key_temp_y i ', self.key_temp_y[i].shape)
            stack_xy = np.concatenate(( self.stack_temp, self.key_temp_y[i]), axis = axis)
            self.stack_temp = stack_xy
            if i == len(self.key_temp_y)-1 :
                self.key_xyz[0] =  self.stack_temp
                print('    selesai merge xy menjadi volume xyz')
                print('shape xyz ', self.key_xyz[0].shape)
        
def konstruksi_volume(data, sizeZ) :
    dim = np.ndim(data)
    if dim == 5 :
        axis_z = 3
        axis_y = 2
        axis_x = 1
    else :
        axis_z = 2
        axis_y = 1
        axis_x = 0
        
    konstruk = konstruksi(data)
    maks = max(list(range(0, data.shape[0]+sizeZ, sizeZ)))
    print('maks = ', maks)
    for i in range(0, maks, sizeZ) :
        print('i = ', i)
        if (konstruk.m == 4 or i == maks) :     #konstruk.m == 7 (56X56X22) konstruk.m == 4 (96X96X12); konstruk.m == 3 (120X120X8); konstruk.m == 2 (224X224X12)
            if i == maks :
                print('   ------i==maks------')
                print('---------------------------------------------------------- i = ', i)  
                konstruk.konstruksi_z(i, data.shape[0], axis_z)     
                print('Selesai stack baris x, ', 'jumlah x = ', len(konstruk.key_temp_x), ' m = ', konstruk.m)
                konstruk.konstruksi_y(axis_y)
                break     
            else :
                print('----m==3----') #print('----m==7----')
                print('---------------------------------------------------------- i = ', i)   
                konstruk.konstruksi_z(i, i+sizeZ, axis_z)  
                print('Selesai stack baris x, ', 'jumlah x = ', len(konstruk.key_temp_x), ' m = ', konstruk.m)
                konstruk.konstruksi_y(axis_y)
                continue
        print('---------------------------------------------------------- i = ', i)
        konstruk.konstruksi_z(i, i+sizeZ, axis_z)      
    konstruk.konstruksi_x(axis_x)
    return konstruk.key_xyz[0]

dim = np.ndim(konstruk.key_xyz[0])
fig = plt.figure(figsize=(336, 392))
for i in range(0, konstruk.key_xyz[0].shape[-1]) :    
    fig.add_subplot(konstruk.key_xyz[0].shape[-1], 1, i+1)
    if dim == 4 : 
        plt.imshow(konstruk.key_xyz[0][0,:,:,i], cmap='gray');
    else:
        plt.imshow(konstruk.key_xyz[0][:,:,i], cmap='gray');
    plt.axis('off')

if (j == data[0].shape[0]-1 and temp_z.shape[3] < (22*sizeZ)):
    sisa = int(22*sizeZ - temp_z.shape[3])
    print('   ternyata z kurang = ', temp_z.shape[3], ' sisa ', sisa)
    temp_z = np.concatenate((temp_z, np.zeros((1, 56, 56, sisa))), axis = 3)
    
if konstruk.m < 6:
    print('   ternyata m < 6 ', 'm = ', konstruk.m, ' tambah zeros')
    while konstruk.m < 6 :
        konstruk.m = konstruk.m + 1
        konstruk.key_temp_x[konstruk.m] = np.zeros((1, 56, 56, (22*sizeZ))) 
        print('key_temp_x', ' [',konstruk.m,']', ' shape =', konstruk.key_temp_x[konstruk.m].shape, ' i = ', j)  
'''