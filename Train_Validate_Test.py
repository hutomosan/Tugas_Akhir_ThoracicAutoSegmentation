# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:09:50 2023

@author: Wahyu Hutomo Nugroho (hutomonugroho@gmail.com)
"""
#import os
#os.environ['THEANO_FLAGS']="device=cuda0,floatX=float32"
#import sys
#import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
from Input import *
from Output import *
from Image_Pre_and_Post_Processing import*

import pickle
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import *
#from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.nonlinearities import rectify
from lasagne.regularization import regularize_network_params, regularize_layer_params, l1, l2

import skimage
import skimage.segmentation
import medpy.metric
import codecs

import timeit
#import wandb
#os.environ["WANDB_API_KEY"] = "30d3220783ee4cbb8279f3f3b0971b58d5a2e850"
#os.environ["WANDB_MODE"] = "dryrun"

class LogisticRegression(object):
    def __init__(self, input_feature):
        self.batch_size, self.n_class, self.dim_x, self.dim_y, self.dim_z = input_feature.shape                                     
        self.input = input_feature.dimshuffle(0,2,3,4,1).reshape((self.batch_size*self.dim_x*self.dim_y*self.dim_z, self.n_class))   
                                                                                                                                    
                                                                                                                                   
        self.p_y_given_x = T.nnet.softmax(self.input)                                                                             
        self.score_map = self.p_y_given_x.reshape((self.batch_size, self.dim_x, self.dim_y, self.dim_z, self.n_class))[:,:,:,:,1]

    def negative_log_likelihood(self, label):
        y = label.reshape((self.batch_size*self.dim_x*self.dim_y*self.dim_z,))
        loss = -((T.log(self.p_y_given_x)))[T.arange(y.shape[0]), y] #provides us with a vector that contains the log likelihoods for each training example/class digit response pair. Hence this vector has length N
                                                                 # loss is 1-D vector contains log likelihoods
                                                                
        mask = y * 10
        weighted_loss = T.mean(loss + loss * mask)

        return weighted_loss
    
    def errors(self, label) :
        return T.mean(T.neq(T.round(self.score_map), label))*100


def build_model(input_var, batch_size):                                        
    net = {}                                                                   

    # Residual 1
    net['input'] = InputLayer((batch_size, 1, None, None, None), input_var=input_var)                                                       
    
    # Low Level 1
    net['conv1a'] = batch_norm(Conv3DLayer(net['input'], 64, (3,3,3), pad='same', nonlinearity=rectify))                                                                                                                                                                                       
    net['conv1b'] = batch_norm(Conv3DLayer(net['conv1a'], 64, (3,3,3), pad='same', nonlinearity=rectify))    
    net['conv1c'] = Conv3DLayer(net['conv1b'], num_filters=64, filter_size=(3,3,3), stride=(2,2,2), pad='same', nonlinearity=None)
    net['pool1'] = MaxPool3DLayer(net['conv1b'], pool_size=(2,2,2)) # 80,80,16
    
    # Residual 2
    net['res2'] = BatchNormLayer(net['conv1c'])                                 
    net['res2'] = NonlinearityLayer(net['res2'], nonlinearity=rectify)                                                                  
    net['res2'] = batch_norm(Conv3DLayer(net['res2'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res2'] = Conv3DLayer(net['res2'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res2'] = ElemwiseSumLayer([net['res2'], net['conv1c']])              
    
    # Residual 3
    net['res3'] = BatchNormLayer(net['res2'])
    net['res3'] = NonlinearityLayer(net['res3'], nonlinearity=rectify)
    net['res3'] = batch_norm(Conv3DLayer(net['res3'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res3'] = Conv3DLayer(net['res3'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res3'] = ElemwiseSumLayer([net['res3'], net['res2']])   
   
    net['bn3'] = BatchNormLayer(net['res3'])
    net['relu3'] = NonlinearityLayer(net['bn3'], nonlinearity=rectify)
    net['conv3a'] = Conv3DLayer(net['relu3'], num_filters=64, filter_size=(3,3,3), stride=(2,2,1), pad='same', nonlinearity=None)
    #Low Level 2
    net['pool2'] = MaxPool3DLayer(net['relu3'], pool_size=(2,2,1)) # 40,40,16

    # Residual 4
    net['res4'] = BatchNormLayer(net['conv3a'])
    net['res4'] = NonlinearityLayer(net['res4'], nonlinearity=rectify)
    net['res4'] = batch_norm(Conv3DLayer(net['res4'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res4'] = Conv3DLayer(net['res4'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res4'] = ElemwiseSumLayer([net['res4'], net['conv3a']])

    # Residual 5
    net['res5'] = BatchNormLayer(net['res4'])
    net['res5'] = NonlinearityLayer(net['res5'], nonlinearity=rectify)
    net['res5'] = batch_norm(Conv3DLayer(net['res5'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res5'] = Conv3DLayer(net['res5'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res5'] = ElemwiseSumLayer([net['res5'], net['res4']])
    
    net['bn5'] = BatchNormLayer(net['res5'])
    net['relu5'] = NonlinearityLayer(net['bn5'], nonlinearity=rectify)
    net['conv5a'] = Conv3DLayer(net['relu5'], num_filters=64, filter_size=(3,3,3), stride=(2,2,2), pad='same', nonlinearity=None)
    
    # Residual 6
    net['res6'] = BatchNormLayer(net['conv5a'])
    net['res6'] = NonlinearityLayer(net['res6'], nonlinearity=rectify)
    net['res6'] = batch_norm(Conv3DLayer(net['res6'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res6'] = Conv3DLayer(net['res6'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res6'] = ElemwiseSumLayer([net['res6'], net['conv5a']])

    # Residual 7 (High Level 1)
    net['res7'] = BatchNormLayer(net['res6'])
    net['res7'] = NonlinearityLayer(net['res7'], nonlinearity=rectify)
    net['res7'] = batch_norm(Conv3DLayer(net['res7'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res7'] = Conv3DLayer(net['res7'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res7'] = ElemwiseSumLayer([net['res7'], net['res6']])

    net['bn7'] = BatchNormLayer(net['res7'])
    net['relu7'] = NonlinearityLayer(net['bn7'], nonlinearity=rectify)
    net['conv8'] = batch_norm(Conv3DLayer(net['relu7'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))

    # upscale 1
    net['upscale1'] = Upscale3DLayer(net['conv8'], scale_factor=(2,2,2), mode='repeat')                             
    
    # Low Level 2 + High Level 1
    net['concat1'] = ConcatLayer([net['pool2'], net['upscale1']])                                                   
    net['upconv1a'] = batch_norm(Conv3DLayer(net['concat1'], 64, (1,1,1), pad='same', nonlinearity=rectify))
    net['upconv1b'] = batch_norm(Conv3DLayer(net['upconv1a'], 64, (3,3,3), pad='same', nonlinearity=rectify))

    # upscale 2 (High Level 2)
    net['upscale2'] = Upscale3DLayer(net['upconv1b'], scale_factor=(2,2,1), mode='repeat')
    
    # Low Level 1 + High Level 2
    net['concat2'] = ConcatLayer([net['pool1'], net['upscale2']])
    net['upconv2a'] = batch_norm(Conv3DLayer(net['concat2'], 64, (1,1,1), pad='same', nonlinearity=rectify))
    net['upconv2b'] = batch_norm(Conv3DLayer(net['upconv2a'], 64, (3,3,3), pad='same', nonlinearity=rectify))

    # upscale 3 (High Level 3)
    net['upscale3'] = Upscale3DLayer(net['upconv2b'], scale_factor=(2,2,2), mode='repeat')
    
    # Output
    net['upconv3a'] = batch_norm(Conv3DLayer(net['upscale3'], 64, (1,1,1), pad='same', nonlinearity=rectify))
    net['upconv3b'] = batch_norm(Conv3DLayer(net['upconv3a'], 64, (3,3,3), pad='same', nonlinearity=rectify))
    net['output'] = batch_norm(Conv3DLayer(net['upconv3b'], 2, (3,3,3), pad='same', nonlinearity=None))

    params = lasagne.layers.get_all_params(net['output'], trainable=True)       
                                                                               
    l2_penalty = regularize_network_params(net['output'], l2)                      

    return net, params, l2_penalty

def train_validate_model(results_path, batch_size, m_batch_size, patch_size, n_epochs, masking, fine_tune, Organ, base_lr=0.001):
    ftensor5 = T.TensorType('float32', (False,)*5)                             
    x = ftensor5()                                                              
    y = T.itensor4('y')                                                       

    network, params, l2_penalty = build_model(x, batch_size)                                    
    
    if fine_tune is True: # Fine tune the model if this flag is True
        with np.load(os.path.join(results_path,'params_5040.npz'), allow_pickle = True) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            set_all_param_values(network['output'], param_values[0])
            print('initialization done!')

    prediction = get_output(network['output'])                                  
    loss_layer = LogisticRegression(prediction)                                 
    cost_output = loss_layer.negative_log_likelihood(y)      
    error = loss_layer.errors(y)                  

    lamda=0.0001
    cost_penaltied = cost_output + lamda * l2_penalty
    updates = lasagne.updates.adadelta(cost_penaltied, params)
    train = theano.function([x, y], [cost_penaltied, cost_output, error], updates=updates)                                                                              
    validate = theano.function([x, y], [cost_penaltied, cost_output, error])   

    start_time = timeit.default_timer()
    print('-------------------------------------------------------')
    print('----------------------Training ML----------------------')
    file_name = results_path + "/log_loss_error.txt"
    fw = codecs.open(file_name, "w", "utf-8-sig")
    data_folder_train = "C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\LCTSC"
    
    train_cost_p_avg, train_cost_o_avg, train_error_avg = [], [], []
    valid_cost_p_avg, valid_cost_o_avg, valid_error_avg = [], [], []
    cost_p, cost_o, error = [], [], []
    
    key_train = {0:train_cost_p_avg, 1:train_cost_o_avg, 2:train_error_avg}
    key_valid = {0:valid_cost_p_avg, 1:valid_cost_o_avg, 2:valid_error_avg}
    #wandb.init()
    epoch = 0
    for train_x, train_y, case_num, itr in load_train(data_folder = data_folder_train, m_batch_size=m_batch_size, n_epochs=n_epochs, patchSize=patch_size, masking_status=masking, Organ=Organ):  
        n_train_batches = train_x.shape[0] / batch_size
        for minibatch_index in list(range(int(n_train_batches))):
            train_x_itr = train_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size,:,:,:]    
            train_y_itr = train_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size,:,:,:]
            #print('    memulai training...')
            cost_train_penaltied, cost_train_classify, error_train = train(train_x_itr.astype('float32'), train_y_itr)     
            cost_p.append(cost_train_penaltied) 
            cost_o.append(cost_train_classify)
            error.append(error_train )                                 
            #print(('    itr: {}, train loss overall: {}, train loss classify: {}').format(itr, train_cost_itr, train_cost_itr_classify))
        
        if itr % case_num == 0 :
            epoch = epoch+1
            '''-----Menghitung cost dan error training tiap epoch-----'''
            avg_cost_p = sum(cost_p)/len(cost_p)
            key_train[0].append(avg_cost_p)
            avg_cost_o = sum(cost_o)/len(cost_o)
            key_train[1].append(avg_cost_o)
            avg_error = sum(error)/len(error)
            key_train[2].append(avg_error)
            print(('Epoch: {},     train loss overall: {},                   train error: {}').format(epoch, avg_cost_o, avg_error), file=fw)
           
            print(' ')
            print('-------------------------------------------------------')
            print('----------------------Validasi ML----------------------')
            data_folder_validate = "C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Validasi\\LCTSC"
            cost_p, cost_o, error = [], [], []
            for validate_x, validate_y, case_num0, itr0 in load_train(data_folder = data_folder_validate, m_batch_size = m_batch_size, n_epochs = 1, patchSize=patch_size, masking_status=masking, Organ=Organ) : 
                n_validate_batches = validate_x.shape[0] / batch_size
                for minibatch_index in list(range(int(n_validate_batches))) :    
                    validate_x_itr = validate_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size,:,:,:]    
                    validate_y_itr = validate_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size,:,:,:]
                    #print('          memulai validasi')
                    cost_validate_penaltied, cost_validate_classify, error_validate = validate(validate_x_itr, validate_y_itr)
                    cost_p.append(cost_validate_penaltied) 
                    cost_o.append(cost_validate_classify)
                    error.append(error_validate)   
                    
                if itr0 % case_num0 == 0 :
                    '''-----Menghitung cost dan error validasi tiap epoch-----'''
                    avg_cost_p = sum(cost_p)/len(cost_p)
                    key_valid[0].append(avg_cost_p)
                    avg_cost_o = sum(cost_o)/len(cost_o)
                    key_valid[1].append(avg_cost_o)
                    avg_error = sum(error)/len(error)
                    key_valid[2].append(avg_error)
                    
                    cost_p, cost_o, error = [], [], []
                    print(('Epoch: {},     valid loss overal: {},                   valid error: {}').format(epoch, avg_cost_o, avg_error), file=fw)
            print('---------------------Selesai validasi------------------')
            print(' ')      
            if epoch % 5 == 0 and itr > 1 :
                np.savez(os.path.join(results_path, 'params_'+str(itr)+'.npz'), get_all_param_values(network['output'])) 
                np.savez(os.path.join(results_path, 'Data_Akhir_'+str(itr)+'.npz'), [key_train, key_valid, epoch, itr])
                print('  save model and data_akhir done ...')
                print(' ')             
    fw.close()
    end_time = timeit.default_timer()
    print("The code ran for %.1fs" % (end_time - start_time))
    data_return = {0:key_train, 1:key_valid, 2:epoch, 3:itr}
    return data_return
    
    
def predict_algorithm(batch_size, model_path, dataset_path, patchSize, masking_status, EandS, Organ):
    ftensor5 = T.TensorType('float32', (False,)*5)
    x = ftensor5()                                                                                                                    
    network, params, l2_penalty = build_model(x, batch_size)  

    parameters = np.load(model_path, allow_pickle = True)
    set_all_param_values(network['output'], parameters['arr_0'])
    
    prediction = get_output(network['output'])                                  
    loss_layer = LogisticRegression(prediction)
    predict = theano.function(
        inputs=[x], 
        outputs=loss_layer.score_map
    )
    data_folder = dataset_path
    files = os.listdir(data_folder) 
    files.remove('LICENSE')
    case_num = len(files) 
    print('Jumlah pasien = ', case_num)
    predict_dataset_path    = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test'
    pembagi = 6
    pembagi_z = 2
    patchX, patchY, patchZ = patchSize
    padding_x = int(((pembagi-1)*patchX)/pembagi)  #Penyebut
    padding_y = int(((pembagi-1)*patchY)/pembagi)
    padding_z = int((((pembagi_z-1)*patchZ)/pembagi_z))
    #print("Padding_xy = ", padding_x, ', ', padding_y,  " Padding_z = ", padding_z)
    start_time = timeit.default_timer()
    key_img = {}
    key_seg = {}
    key_prediction = {}
    for itr in list(range(case_num)): 
        folder = files[itr]   
        print(' ')
        print('Iterasi ', itr, ' Folder ', folder)
        case = dicomSubject(subject_folder_path=os.path.join(data_folder, folder), masking_status = masking_status, Organ=Organ)
        
        # mask and it's extreme body coordinates    
        max_slice = max(case.plane_idx) 
        min_slice = min(case.plane_idx)
        
        if EandS==True :                                     #(min_slice-padding_z):(max_slice+padding_z)]
            mask_img = case.masking['modalitas 1'][:,:,(min_slice-0):(max_slice+0)]  #(min_slice-0):(max_slice+0)]
            mask_img = image_processing_1(mask_img)
            img = case.image['modalitas 1'][:,:,(min_slice-0):(max_slice+0)] * mask_img #(min_slice-0):(max_slice+0)]
            
            mask_img = np.pad(mask_img, ((padding_x,padding_x), (padding_y,padding_y), (2*padding_z,2*padding_z)), 'constant', constant_values=((0,0), (0,0), (0,0)))  
            #print("Ukuran mask = ", mask_img.shape)
            
            img = np.pad(img, ((padding_x,padding_x), (padding_y,padding_y), (2*padding_z,2*padding_z)), 'constant', constant_values=((0,0), (0,0), (0,0))) - 0.2
            key_img[itr] = img
            #print("Ukuran Image = ", img.shape)
        else :
            mask_img = case.masking['modalitas 1'][:,:,(min_slice-padding_z):(max_slice+padding_z)]  #(min_slice-0):(max_slice+0)]
            mask_img = image_processing_1(mask_img)
            img = case.image['modalitas 1'][:,:,(min_slice-padding_z):(max_slice+padding_z)] * mask_img #(min_slice-0):(max_slice+0)]
        
            mask_img = np.pad(mask_img, ((padding_x,padding_x), (padding_y,padding_y), (1*padding_z,1*padding_z)), 'constant', constant_values=((0,0), (0,0), (0,0)))  
            #print("Ukuran mask = ", mask_img.shape)
            
            img = np.pad(img, ((padding_x,padding_x), (padding_y,padding_y), (1*padding_z,1*padding_z)), 'constant', constant_values=((0,0), (0,0), (0,0))) - 0.2
            key_img[itr] = img
            #print("Ukuran Image = ", img.shape)
            
        mask_coord = np.where(mask_img[:,:,:] == 1)
        max_coord_x = max(mask_coord[1]) 
        min_coord_x = min(mask_coord[1])                                                                                    
        max_coord_y = max(mask_coord[0]) 
        min_coord_y = min(mask_coord[0])
        
        if EandS==True :
            max_coord_z = mask_img.shape[-1] - padding_z  #2*padding_z
            min_coord_z = padding_z                   #2*padding_z
            boundary_right = (max_coord_x + padding_x) - (144) ##
        else :
            max_coord_z = mask_img.shape[-1] - 2*padding_z
            min_coord_z = 2*padding_z                  
            boundary_right = (max_coord_x + padding_x)
        
        boundary_bottom = max_coord_y + padding_y 
        boundary_behind = max_coord_z + padding_z
        
        sizeX, sizeY, sizeZ = img.shape                                       
        volume_img = np.zeros([sizeX, sizeY, sizeZ])                                      
        
        #predicted samples
        if EandS==True :
            tepi_x = (min_coord_x + ((1/pembagi)*patchX) - (patchX/2)) + (144/2) ## #Jika 1/2 = 0; Jika 1/3 = 1/6; Jika 1/4 = 1/4; Jika 1/6 = 1/3; Jika 1/12 = 11/24
            tepi_y = (min_coord_y + ((1/pembagi)*patchY) - (patchY/2)) + (144/2) ##
        else :
            tepi_x = (min_coord_x + ((1/pembagi)*patchX) - (patchX/2)) ## #Jika 1/2 = 0; Jika 1/3 = 1/6; Jika 1/4 = 1/4; Jika 1/6 = 1/3; Jika 1/12 = 11/24
            tepi_y = (min_coord_y + ((1/pembagi)*patchY) - (patchY/2)) ##    
        
        tepi_z = min_coord_z + ((1/pembagi_z)*patchZ) - (patchZ/2)
        temp_img = np.zeros([1, 1, patchX, patchY, patchZ])
        
        centre_y = int(tepi_y)
        lacak_y = centre_y + int(patchY/2)
        while lacak_y < boundary_bottom :
            centre_x = int(tepi_x) #pembagi -> 6; 4 -> 4
            lacak_x = centre_x + int(patchX/2) 
            while lacak_x < boundary_right :
                centre_z = int(tepi_z)#
                lacak_z = centre_z + int(patchZ/2)
                while lacak_z < boundary_behind :
                    #print("                     Centre = ", centre_x, ', ', centre_y, ', ', centre_z)
                    temp_img[0,0,:,:,:] = img[int(centre_x - patchX/2):int(centre_x + patchX/2), 
                                              int(centre_y - patchY/2):int(centre_y + patchY/2), 
                                              int(centre_z - patchZ/2):int(centre_z + patchZ/2)]    
                    #print("     temp_img = ", temp_img.shape)
                    prediksi_temp = predict(temp_img[:,:,:,:,:].astype('float32'))
                    #print("     prediksi_img = ", prediksi_temp.shape)
                    volume_img[int(centre_x - patchX/2):int(centre_x + patchX/2), 
                               int(centre_y - patchY/2):int(centre_y + patchY/2), 
                               int(centre_z - patchZ/2):int(centre_z + patchZ/2)] = volume_img[int(centre_x - patchX/2):int(centre_x + patchX/2), 
                                                                                               int(centre_y - patchY/2):int(centre_y + patchY/2), 
                                                                                               int(centre_z - patchZ/2):int(centre_z + patchZ/2)] + prediksi_temp[0,:,:,:]
                    #print("     Volume_img = ", volume_img.shape)
                    centre_z = centre_z + (patchZ/pembagi_z)
                    lacak_z = lacak_z + (patchZ/pembagi_z)
                    #print("                 Centre = ", centre_x, ", ", centre_y, ", ", centre_z)
                centre_x = centre_x + (patchX/pembagi)
                lacak_x = lacak_x + (patchX/pembagi)
                #print("Centre = ", centre_x, ", ", centre_y, ", ", centre_z)
            centre_y = centre_y + (patchY/pembagi)
            lacak_y = lacak_y + (patchY/pembagi)
            #print("         Centre = ", centre_x, ", ", centre_y, ", ", centre_z)
        #print(" ")  
        key_prediction[itr] = volume_img[:,:,(2*padding_z):(volume_img.shape[-1]-(2*padding_z)+1)] * mask_img[:,:,(2*padding_z):(volume_img.shape[-1]-(2*padding_z)+1)]
        np.savez(os.path.join(predict_dataset_path, 'Prediksi_ESP_96_15_11_W=10_%d' %int(itr) + '.npz'), key_prediction[itr])
        #print("Prediksi = ", key_prediction[itr].shape, ", Min = ", np.min(key_prediction[itr]), ", Max= ", np.max(key_prediction[itr]))
    end_time = timeit.default_timer()
    print("The code ran for %.1fs" % (end_time - start_time))
    return key_img, key_seg, key_prediction

    
if __name__ == '__main__':
    training_dataset_path   = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training'
    model_path              = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\params_7200.npz'
    predict_dataset_path    = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Test\\LCTSC'
    #save_figure_path        = 'C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\Lab_Comp\\On_Mask\\Percobaan 2\\Jantung\\Gambar Prediksi'
    if os.path.exists(model_path):  #Jika sudah ada file model 
        image, segmentation, prediction = predict_algorithm(
                batch_size = 1, 
                model_path = model_path, 
                dataset_path = predict_dataset_path, 
                patchSize = [96, 96, 12], 
                masking_status = True,
                EandS = True,
                Organ = 'ESP')
        '''
        for i in range(len(prediction)) :    
            np.savez(os.path.join(predict_dataset_path, 'Prediksi_J_120_(6,2)_15_%d' %int(i) + '.npz'), prediction[i])
        '''
        '''
        for i in range(0, prediction[0].shape[-1]) : 
            plt.figure(figsize=(12.8, 9.6), dpi=30)
            plt.imshow(prediction[0][:,:,i], cmap='gray');
            plt.axis('off')      
            plt.show()
        '''
    else: #Jika belum ada file model
        data_akhir = train_validate_model(
                results_path = training_dataset_path, 
                batch_size = 3, 
                m_batch_size = 15, 
                patch_size = [48,48,6], 
                n_epochs = 200, 
                masking = True, fine_tune=False,
                Organ = "ESP")
        
        pembuat_kurva(data_akhir[0], data_akhir[1], data_akhir[2])
    
        #Menyambung data
        #data_1 = np.load(os.path.join(training_dataset_path, "Data_Akhir_2160.npz"), allow_pickle=True)['arr_0']
        #data_2 = np.load(os.path.join(training_dataset_path, "Data_Akhir_5040.npz"), allow_pickle=True)['arr_0']
        #data_akhir = penyambung_data(data_1 = data_1, data_2 = data_2)
    
  
'''_______Arsip_______

f = np.load(os.path.join(training_dataset_path,'masking.npz'), allow_pickle = True)
masking = [f['arr_%d' % i] for i in range(len(f.files))]
masking_ed = image_processing_1(masking[0])
print("masking_ed = ", masking_ed.shape)
fig_akhir = plt.figure(dpi=50)
for i in range(0, masking_ed.shape[-1]) :    
    plt.imshow(masking_ed[:,:,i], cmap='gray');
    plt.axis('off')      
    plt.show()


def konstruksi(hasil_prediksi, data_asli_seg, data_asli_img, sizeZ) :
    fig_prediksi = {}
    fig_segmentasi = {}
    fig_image = {}
    i = 0
    while i < len(hasil_prediksi) :
        fig_prediksi[i] = konstruksi_volume(np.round(hasil_prediksi[i]), sizeZ[i])
        fig_segmentasi[i] = konstruksi_volume(data_asli_seg[i], sizeZ[i])
        fig_image[i] = konstruksi_volume(data_asli_img[i], sizeZ[i])
        i = i+1
    return fig_prediksi, fig_segmentasi, fig_image

def test_model(batch_size, model_path, predict_dataset_path, patch_size, masking) :
    ftensor5 = T.TensorType('floatpembagi2', (False,)*5)
    x = ftensor5()                                                              
    y = T.itensor4('y')                                                         
    network, params, l2_penalty = build_model(x, batch_size)  

    parameters = np.load(model_path, allow_pickle = True)
    set_all_param_values(network['output'], parameters['arr_0'])
    
    prediction = get_output(network['output'])                                  
    loss_layer = LogisticRegression(prediction)
    predict = theano.function(
        inputs=[x], 
        outputs=loss_layer.score_map
    )
    
    key_hasil_prediksi = {}
    key_data_asli_img = {}
    key_data_asli_seg = {}
    key_size_z = {}
    m = 0                                        #Patch Tidak dikali 2#
    for test_x, test_y, sizeZ in load_test(patchSize=patch_size, dataset_path=predict_dataset_path, masking_status=masking):
        print(' ')
        n_test_batches = test_x.shape[0]
        prediksi_temp = predict(test_x[0*batch_size:(0+1)*batch_size, :, :, :, :].astype('floatpembagi2'))
        print('shape prediksi =', prediksi_temp.shape)
        for minibatch_index in list(range(1, n_test_batches)) :
            temp = np.append(prediksi_temp, predict(test_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :, :, :, :].astype('float32')), axis = 0)
            prediksi_temp = temp
            if minibatch_index == n_test_batches-1:
                key_hasil_prediksi[m] = prediksi_temp
        print('selesai testing...')
        print('Jml hasil_prediksi ', key_hasil_prediksi[m].shape)
        key_data_asli_img[m] = test_x
        key_data_asli_seg[m] = test_y
        key_size_z[m] = sizeZ
        m = m+1                    
    return key_hasil_prediksi, key_data_asli_seg, key_data_asli_img, key_size_z

hasil_prediksi, data_asli_seg, data_asli_img, sizeZ = test_model(batch_size = 1, 
                                                                 model_path = model_path, 
                                                                 predict_dataset_path=predict_dataset_path, 
                                                                 patch_size=[96, 96, 12], 
                                                                 masking = False)

fig_prediksi, fig_segmentasi, fig_image = konstruksi(
    hasil_prediksi, 
    data_asli_seg, 
    data_asli_img, 
    sizeZ)     

def figure_maker(fig, kode) :
    i = 0
    dim = np.ndim(fig[0])
    while i < 1 :
        fig_akhir = plt.figure(figsize=(336, 448), dpi=125)
        m = 0
        if dim == 4 :
            for n in range(0, fig[i].shape[-1]) :    
                fig_akhir.add_subplot(fig[i].shape[-1], 1, m+1)
                plt.imshow(fig[i][0,:,:,n], cmap='gray');
                plt.title('Slice-%d' % (n+1))
                plt.axis('off')
                m = m+1 
            plt.close(fig_akhir)
        else :
            for n in range(0, fig[i].shape[-1]) :    
                fig_akhir.add_subplot(fig[i].shape[-1], 1, m+1)
                plt.imshow(fig[i][:,:,n], cmap='gray');
                plt.title('Slice-%d' % (n+1))
                plt.axis('off')
                m = m+1 
            plt.close(fig_akhir)
    
        if kode == 0 :
            fig_akhir.savefig("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\Percobaan 3\\Paru-paru\\Gambar Prediksi\\Image-%d.png" % (i+1), bbox_inches='tight')
        elif kode == 1 :
            fig_akhir.savefig("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\Percobaan 3\\Paru-paru\\Gambar Prediksi\\Prediksi-%d.png" % (i+1), bbox_inches='tight')
        elif kode == 2 :
            fig_akhir.savefig("C:\\Users\\hutom\\OneDrive\\Documents\\Dicom_Data_Training\\Percobaan 3\\Paru-paru\\Gambar Prediksi\\Segmentasi-%d.png" % (i+1), bbox_inches='tight')
        plt.show()
        i = i+1;
        
 if not os.path.exists(results_path):
     os.makedirs(results_path)
     print('make folder', results_path)
     
fig_akhir = plt.figure(figsize=(512, 512), dpi=127)
for i in range(0, 32) :    
    fig_akhir.add_subplot(64, 2, (2*i)+1)
    plt.imshow(fig_img[0,:,:,i], cmap='gray');
    plt.axis('off')
    fig_akhir.add_subplot(64, 2, (2*i)+2)
    plt.imshow(fig_prediksi[:,:,i], cmap='gray');
    plt.axis('off')        
'''
    