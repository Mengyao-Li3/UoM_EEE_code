#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os

from scipy import signal
import tensorflow as tf
#import tensorflow_ranking as tfr
from tensorflow.keras.models import Model, load_model #Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input # Concatenate,
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import KFold, StratifiedKFold
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import time
import cv2
import shutil
import random
#from focal_loss import BinaryFocalLoss

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the reshaped recordings and labels.
    if verbose >= 1:
        print('Extracting reshaped recordings and labels from the Challenge data...')

    recordings = list()
    patient_features = list()
    outcomes = list()
    cpcs = list()
    patients = list() # created for cross validation

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # Extract reshaped recordings.
        b,a = signal.butter(4, [0.02,0.26], 'bandpass') # bandpass filter
        current_recordings, meta_features = get_recordings(patient_metadata, recording_metadata, recording_data,b,a)
        recordings.append(current_recordings)
        patient_features.append(meta_features)

        # Extract labels and patients' ids for each recording.
        current_outcome = get_outcome(patient_metadata) # 0: good, 1: poor
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata) # 1,2,3,4,5
        cpcs.append(current_cpc)
        patients.append(patient_id)

    flag = 1 # 1: no segmenting; 0: with segmenting
    if flag == 0:
        #data segmenting and short-time Fourier transform
        if verbose >= 1:
            print('Segmenting data and employing short-time Fourier transform...')

        X_all = list() # peprocessed recordings
        y1_all = list() # preprocessed outcomes
        y2_all = list() # preprocessed cpcs
        z_all = list() # preprocessed patients' ids
        frame_size = int(30000) # size of each segment
        hop_size = int(30000) # non-overlapping
        X_all, y1_all, y2_all, z_all = get_frames(recordings, frame_size, hop_size, outcomes, cpcs, patients, model_folder)

    else:
        #short-time Fourier transform
        if verbose >= 1:
            print('Employing short-time Fourier transform...')

        frames = list()
        for j in range(len(recordings)):

            x_train = recordings[j]
            STFT = list()
            i = 0
            img = list()

            for m in range(18): # number of channels

                if m == 2: # the 3rd channel
                    img = STFT1(x_train[:,m],1, model_folder, j, i, outcomes[j], cpcs[j], patients[j])
                    # dsize = output_width, output_height
                    STFT.append(cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR)) # interpolation = cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
                    #filename = os.path.join(data_folder, 'resized image.png')
                    #cv2.imwrite(filename, cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR))
                else:
                    img = STFT1(x_train[:,m],0, model_folder, j, i, outcomes[j], cpcs[j], patients[j])
                    STFT.append(cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR)) # interpolation = cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
            
            frames.append(np.dstack((STFT[0], STFT[1], STFT[2], STFT[3], STFT[4], STFT[5], STFT[6], STFT[7], STFT[8], STFT[9], STFT[10], STFT[11], STFT[12], STFT[13], STFT[14], 
                                        STFT[15], STFT[16], STFT[17])))
        
        # bring the segment into a better shape
        X_all = np.asarray(frames).reshape(-1, np.shape(STFT[0])[0], np.shape(STFT[0])[1],18)
        feature_all = np.asarray(patient_features).reshape(-1,len(patient_features[0]))
        y1_all = np.asarray(outcomes)
        y2_all = np.asarray(cpcs)

    # Generate outcome model and cpc model
    outcome_model = generate_cnn(2, X_all, feature_all) # 2 labels: good, poor
    cpc_model = generate_cnn(5, X_all, feature_all) # 5 labels: 1, 2, 3, 4, 5

    # set early stopping criteria
    pat = 5 #10 # the number of epochs with no improvement after which the training will stop
    early_stopping =  EarlyStopping(monitor='val_loss',patience=pat, verbose=1)

    selector = 2 # 1: KFold, 2: StratifiedKFold
    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(os.path.join(model_folder, '10-fold CV'), exist_ok=True)

    dt = list()
    t1 = time.time()

    n_folds = 5#10
    epochs = 300
    batch_size = 64#1#2#4#8#16#32#64

    if selector == 1:
        kf = KFold(n_splits=n_folds, shuffle=True)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    group = None

    # outcome model training
    if verbose >= 1:
        print('Training the Challenge outcome model on the Challenge data...')

    k = 0
    outcome_model_history = list()
    outcome_res = list()
    # for outcome model
    for train_index, cv_index in kf.split(X_all,y1_all,groups=group):
        k = k + 1
        print("Training on Fold: ", k)
        X_train, X_cv = X_all[train_index], X_all[cv_index]    
        y1_train, y1_cv = y1_all[train_index], y1_all[cv_index]
        fea_train, fea_cv = feature_all[train_index], feature_all[cv_index]
        if len(fea_train) == 0:
            print('wrong')
        else:
            pass
        # one-hot convertion
        y1_train = tf.keras.utils.to_categorical(y1_train)
        y1_cv = tf.keras.utils.to_categorical(y1_cv)   
        
        if len(y1_cv[0]) == 2:
            pass
        else:
            A = (2-len(y1_cv[0]))*[0]
            y1_cv_new = np.concatenate([y1_cv[0],A])
            y1_cv_new = y1_cv_new.reshape((1,2))
            y1_cv_new = y1_cv_new.astype(np.float32)
        
            for i in range(1,len(y1_cv)):
                temp = np.concatenate([y1_cv[i],A])
                temp = temp.reshape((1,2))
                temp = temp.astype(np.float32)
                y1_cv_new = np.concatenate([y1_cv_new,temp], axis = 0)       
            y1_cv = y1_cv_new    

        # save the models as physical files
        if selector == 1:
            outcome_filename = os.path.join(model_folder, '10-fold CV', 'Fold_' + str(k) + '_{epoch:03d}_{val_recall:.4f}' + '_outcome_KFold.h5')
        else:
            outcome_filename = os.path.join(model_folder, '10-fold CV', 'Fold_' + str(k) + '_{epoch:03d}_{val_recall:.4f}' + '_outcome_StratifiedKFold.h5')
        outcome_model_checkpoint = ModelCheckpoint(filepath = outcome_filename, monitor='val_recall', verbose=1, save_best_only=False, save_weights_only=False, mode='max')
        outcome_result, cv_score = fit_and_eval(X_train, X_cv, fea_train, fea_cv ,y1_train, y1_cv, outcome_model, epochs, batch_size,early_stopping,outcome_model_checkpoint)
        outcome_res.append(cv_score[1])
        outcome_model_history.append(outcome_result)

    ##### plot training and validation accuracy and loss for outcome model
    plot_figures('outcome', n_folds, outcome_model_history, model_folder)

    # save the optimal model
    new_file_name = os.path.join(model_folder, '10-fold CV', 'Optimal_outcome_model.h5')
    if selector == 1:
        fold_best_model = list()
        fold_best_metric = list()
        for i in range(n_folds):
            temp = list()
            temp = 1 - np.asarray(outcome_model_history[i].history['val_macro_double_soft_f1'])#1 - np.asarray(outcome_model_history[i].history['val_loss'])#outcome_model_history[i].history['val_recall']
            temp2 = list()
            temp2 = 1-np.asarray(outcome_model_history[i].history['val_recall'])#1 - np.asarray(outcome_model_history[i].history['val_macro_double_soft_tpr'])#outcome_model_history[i].history['val_loss']
            temp_ind = list()
            temp3 = list()
            for j in range(len(temp)):
                if temp[j] == np.max(temp) or temp[j] >= 0.95:
                    temp_ind.append(j)
                    temp3.append(temp2[j])
                else:
                    pass
            final_ind = temp_ind[np.argmin(temp3)]
            fold_best_metric.append([temp[final_ind],temp2[final_ind]])
            print("epoch, val_fpr, val_recall:", final_ind + 1, 1-temp[final_ind], 1-temp2[final_ind])
            fold_best_model.append(os.path.join(model_folder, '10-fold CV', 'Fold_'+str(i+1)+'_'+str(format(final_ind + 1, '03d'))+'_'+str(format(1-temp2[final_ind],'.4f'))+'_outcome_KFold.h5'))
            #fold_best_model.append(os.path.join(model_folder, '10-fold CV', 'Fold_'+str(i+1)+'_'+str(format(final_ind + 1, '03d'))+'_'+str(format(outcome_model_history[i].history['val_recall'][final_ind],'.4f'))+'_outcome_KFold.h5'))
        Ind = list()
        Temp = list()
        for k in range(len(fold_best_model)):
            if fold_best_metric[k][0] == np.max(fold_best_metric,axis=0)[0] or fold_best_metric[k][0] >= 0.95:
                Ind.append(k)
                Temp.append(fold_best_metric[k][1])
            else:
                pass
        shutil.copyfile(fold_best_model[Ind[np.argmin(Temp)]], new_file_name)
    else:
        fold_best_model = list()
        fold_best_metric = list()
        for i in range(n_folds):
            temp = list()
            temp = 1 - np.asarray(outcome_model_history[i].history['val_macro_double_soft_f1'])#1 - np.asarray(outcome_model_history[i].history['val_loss'])#outcome_model_history[i].history['val_recall']
            temp2 = list()
            temp2 = 1-np.asarray(outcome_model_history[i].history['val_recall'])#1 - np.asarray(outcome_model_history[i].history['val_macro_double_soft_tpr'])#outcome_model_history[i].history['val_loss']
            temp_ind = list()
            temp3 = list()
            for j in range(len(temp)):
                if temp[j] == np.max(temp) or temp[j] >= 0.95:
                    temp_ind.append(j)
                    temp3.append(temp2[j])
                else:
                    pass
            final_ind = temp_ind[np.argmin(temp3)]
            fold_best_metric.append([temp[final_ind],temp2[final_ind]])
            print("epoch, val_fpr, val_recall:", final_ind + 1, 1-temp[final_ind], 1-temp2[final_ind])
            fold_best_model.append(os.path.join(model_folder, '10-fold CV', 'Fold_'+str(i+1)+'_'+str(format(final_ind + 1, '03d'))+'_'+str(format(1-temp2[final_ind],'.4f'))+'_outcome_StratifiedKFold.h5'))
            #fold_best_model.append(os.path.join(model_folder, '10-fold CV', 'Fold_'+str(i+1)+'_'+str(format(final_ind + 1, '03d'))+'_'+str(format(outcome_model_history[i].history['val_recall'][final_ind],'.4f'))+'_outcome_StratifiedKFold.h5'))
        Ind = list()
        Temp = list()
        for k in range(len(fold_best_model)):
            if fold_best_metric[k][0] == np.max(fold_best_metric,axis=0)[0] or fold_best_metric[k][0] >= 0.95:
                Ind.append(k)
                Temp.append(fold_best_metric[k][1])
            else:
                pass
        shutil.copyfile(fold_best_model[Ind[np.argmin(Temp)]], new_file_name)
    print('Select final model from Fold_' + str(Ind[np.argmin(Temp)]+1))
    print('Save the optimal outcome model finished.')
    print("======="*12, end="\n\n")  

    # cpc model training
    if verbose >= 1:
        print('Training the Challenge cpc model on the Challenge data...')

    cpc_res = list()
    cpc_model_history = list()
    k = 0
    # for cpc model
    for train_index, cv_index in kf.split(X_all,y2_all,groups=group):
        k = k + 1
        print("Training on Fold: ", k)
        X_train, X_cv = X_all[train_index], X_all[cv_index]    
        y2_train, y2_cv = y2_all[train_index], y2_all[cv_index]
        fea_train, fea_cv = feature_all[train_index], feature_all[cv_index]
        # one-hot convertion
        y2_train = tf.keras.utils.to_categorical(y2_train-1)  
        y2_cv = tf.keras.utils.to_categorical(y2_cv-1)  
        
        if len(y2_cv[0]) == 5:
            pass
        else:
            A = (5-len(y2_cv[0]))*[0]
            y2_cv_new = np.concatenate([y2_cv[0],A])
            y2_cv_new = y2_cv_new.reshape((1,5))
            y2_cv_new = y2_cv_new.astype(np.float32)
        
            for i in range(1,len(y2_cv)):
                temp = np.concatenate([y2_cv[i],A])
                temp = temp.reshape((1,5))
                temp = temp.astype(np.float32)
                y2_cv_new = np.concatenate([y2_cv_new,temp], axis = 0)       
            y2_cv = y2_cv_new  

        # save the models as physical files
        if selector == 1:
            cpc_filename = os.path.join(model_folder, '10-fold CV', 'Fold_' + str(k) + '_{epoch:03d}_{val_mse:.4f}' + '_cpc_KFold.h5')
        else:
            cpc_filename = os.path.join(model_folder, '10-fold CV', 'Fold_' + str(k) + '_{epoch:03d}_{val_mse:.4f}' + '_cpc_StratifiedKFold.h5')
        cpc_model_checkpoint = ModelCheckpoint(filepath = cpc_filename, monitor='val_mse', verbose=1, save_best_only=False, save_weights_only=False, mode='min')
        cpc_result, cv_score = fit_and_eval(X_train, X_cv, fea_train, fea_cv,y2_train, y2_cv, cpc_model, epochs, batch_size,early_stopping,cpc_model_checkpoint)
        cpc_res.append(cv_score[1])
        cpc_model_history.append(cpc_result)

    ##### plot training and validation accuracy and loss for cpc model
    plot_figures('cpc', n_folds, cpc_model_history, model_folder)
            
    # save the optimal model
    new_file_name = os.path.join(model_folder, '10-fold CV', 'Optimal_cpc_model.h5')
    if selector == 1:
        fold_best_model = list()
        fold_best_metric = list()
        for i in range(n_folds):
            temp = list()
            temp = cpc_model_history[i].history['val_loss']
            fold_best_metric.append(np.min(temp))
            print("epoch, val_mse:", np.argmin(temp) + 1, np.min(temp))
            fold_best_model.append(os.path.join(model_folder, '10-fold CV', 'Fold_'+str(i+1)+'_'+str(format(np.argmin(temp) + 1, '03d'))+'_'+str(format(np.min(temp),'.4f'))+'_cpc_KFold.h5'))
        shutil.copyfile(fold_best_model[np.argmin(fold_best_metric)], new_file_name)
    else:
        fold_best_model = list()
        fold_best_metric = list()
        for i in range(n_folds):
            temp = list()
            temp = cpc_model_history[i].history['val_loss']
            fold_best_metric.append(np.min(temp))
            print("epoch, val_mse:", np.argmin(temp) + 1, np.min(temp))
            fold_best_model.append(os.path.join(model_folder, '10-fold CV', 'Fold_'+str(i+1)+'_'+str(format(np.argmin(temp) + 1, '03d'))+'_'+str(format(np.min(temp),'.4f'))+'_cpc_StratifiedKFold.h5'))
        shutil.copyfile(fold_best_model[np.argmin(fold_best_metric)], new_file_name)
    print('Select final model from Fold_' + str(np.argmin(fold_best_metric)+1))
    print('Save the optimal cpc model finished.')
    print("======="*12, end="\n\n")
    
    t2 = time.time()
    dt = t2 - t1
    print("Running time(s): ", dt) 

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):

    outcome_filename = os.path.join(model_folder, '10-fold CV', 'Optimal_outcome_model.h5')
    cpc_filename = os.path.join(model_folder, '10-fold CV', 'Optimal_cpc_model.h5')
    return [load_model(outcome_filename, custom_objects={'macro_double_soft_f1_loss': macro_double_soft_f1_loss, 'macro_double_soft_f1': macro_double_soft_f1}), load_model(cpc_filename)] #, 'macro_double_soft_tpr': macro_double_soft_tpr
    # return [load_model(outcome_filename, custom_objects={'f1_loss': f1_loss, 'f1': f1}), load_model(cpc_filename)]
    #return [load_model(outcome_filename, custom_objects={'custom_loss': custom_loss}), load_model(cpc_filename)]

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):

    # load models
    outcome_model = models[0]
    cpc_model = models[1]

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # get signal quality scores
    quality_score = get_quality_scores(recording_metadata)
    while np.nan in quality_score:
        quality_score.remove(np.nan)

    # Extract reshaped recording.
    current_recording = list()
    patient_features = list()
    b,a = signal.butter(4, [0.02,0.26], 'bandpass')
    current_recording.append(get_recordings(patient_metadata, recording_metadata, recording_data,b,a)[0])
    patient_features.append(get_recordings(patient_metadata, recording_metadata, recording_data,b,a)[1])

    if len(current_recording[0]) == 0:

        print('No data was provided for patient ' + patient_id)
        return random.randint(0,1), 0.5, random.randint(1,5)

    else:

        flag = 1 # 1: no segmenting; 0: with segmenting
        if flag == 0:

            # Extract labels and patient's id
            current_outcome = list()
            current_outcome.append('Nan') # 0: good, 1: poor
            current_cpc = list()
            current_cpc.append('Nan') # 1,2,3,4,5
            current_patient = list()
            current_patient.append(patient_id)

            frame_size = int(30000) # size of each segment
            hop_size = int(30000) # non-overlapping
            x_test, y1_test, y2_test, z_test = get_frames(current_recording, frame_size, hop_size, current_outcome, current_cpc, current_patient, data_folder)

            # Apply models to test data.
            y1_pred = outcome_model.predict(X_test)
            outcome_probability = list()
            for i in range(len(y1_pred)):
                outcome_probability.append(y1_pred[i][1])
            y1_pred = np.argmax(y1_pred, axis = 1)

            y2_pred = cpc_model.predict(X_test)
            y2_pred = np.argmax(y2_pred, axis = 1)

            # calculate weights of decisions
            w = list()
            for i in range(len(quality_score)):
                w.append(quality_score[i]/np.sum(quality_score))

            return  np.sum(list(map(lambda e,f:e*f, w,y1_pred))), np.sum(list(map(lambda e,f:e*f, w,outcome_probability))), np.sum(list(map(lambda e,f:e*f, w,y2_pred+1)))#Counter(np.asarray(y1_pred)).most_common(1)[0][0], np.mean(outcome_probability), np.mean(y2_pred+1)
        
        else:

            #short-time Fourier transform
            x_test = current_recording[0]
            STFT = list()
            frames = list()
            j = 0
            i = 0
            img = list()

            for m in range(18): # number of channels
                if m == 2: # the 3rd channel
                    img = STFT1(x_test[:,m],1, data_folder, j, i, 'Nan', 'Nan', patient_id)
                    # dsize = output_width, output_height
                    STFT.append(cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR)) # interpolation = cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
                    #filename = os.path.join(data_folder, 'resized image.png')
                    #cv2.imwrite(filename, cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR))
                else:
                    img = STFT1(x_test[:,m],0, data_folder, j, i, 'Nan', 'Nan', patient_id)
                    STFT.append(cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR)) # interpolation = cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
            
            frames.append(np.dstack((STFT[0], STFT[1], STFT[2], STFT[3], STFT[4], STFT[5], STFT[6], STFT[7], STFT[8], STFT[9], STFT[10], STFT[11], STFT[12], STFT[13], STFT[14], 
                                                STFT[15], STFT[16], STFT[17])))
                
            # bring the frames into a better shape
            X_test = np.asarray(frames).reshape(-1, np.shape(STFT[0])[0], np.shape(STFT[0])[1],18)
            X_features = np.asarray(patient_features).reshape(-1, len(patient_features[0]))

            # Apply models to test data.
            y1_pred = outcome_model.predict([X_test,X_features])
            print(y1_pred)
            outcome_probability = y1_pred[0][1]
            print(outcome_probability)
            y1_pred = np.argmax(y1_pred)
            print(y1_pred)

            y2_pred = cpc_model.predict([X_test,X_features])
            #y2_pred = cpc_model.predict(X_test)
            y2_pred = np.argmax(y2_pred)

            return  y1_pred, outcome_probability, y2_pred+1 #Counter(np.asarray(y1_pred)).most_common(1)[0][0], np.mean(outcome_probability), np.mean(y2_pred+1)

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

########## data transformation: short-time Fourier transform
def STFT1(X,flag, figure_folder, j, i, y1, y2, z):
        
    f,t,Zxx = signal.stft(X,100) # sampling rate = 100 Hz
    Zxx = np.abs(Zxx)
    
    # if flag ==1:
    #     plt.clf()
    #     c=plt.pcolormesh(t, f, Zxx)
    #     cb = plt.colorbar(c)
    #     cb.set_label('Power/Frequency [dB/Hz]')
    #     plt.title('STFT Diagram')
    #     plt.ylabel('Frequency [Hz]')
    #     plt.xlabel('Time [sec]')
    #     os.makedirs(os.path.join(figure_folder, 'STFT figures'), exist_ok=True)
    #     c3_filename = os.path.join(figure_folder, 'STFT figures', str(y1) + '_' + str(y2) + '_' + str(z) + '_' + str(j) + '_' + str(i) + '_' + 'c3_stft.png')
    #     plt.savefig(c3_filename)
    #     plt.close()
    # else:
    #     pass
    
    return Zxx

########## data segment
def get_frames(x, frame_size, hop_size, y1, y2, z, figure_folder):
    
    frames = list()
    outcomes = list()
    cpcs = list()
    patients = list()

    for j in range(len(x)):

        x_train = x[j]
    
        for i in range(0, np.shape(x_train)[0] - frame_size + 1, hop_size): 
        
            c1 = x_train[i:i+frame_size,0]
            c2 = x_train[i:i+frame_size,1]
            c3 = x_train[i:i+frame_size,2]
            c4 = x_train[i:i+frame_size,3]
            c5 = x_train[i:i+frame_size,4]
            c6 = x_train[i:i+frame_size,5]
            c7 = x_train[i:i+frame_size,6]
            c8 = x_train[i:i+frame_size,7]
            c9 = x_train[i:i+frame_size,8]
            c10 = x_train[i:i+frame_size,9]
            c11 = x_train[i:i+frame_size,10]
            c12 = x_train[i:i+frame_size,11]
            c13 = x_train[i:i+frame_size,12]
            c14 = x_train[i:i+frame_size,13]
            c15 = x_train[i:i+frame_size,14]
            c16 = x_train[i:i+frame_size,15]
            c17 = x_train[i:i+frame_size,16]
            c18 = x_train[i:i+frame_size,17]
        
            y = list()
            for k in range(frame_size): # frame_size
                y.append(k/100)
            # plt.clf()
            # plt.title('Time Series Diagram')
            # plt.plot(y,c3)
            # plt.xticks([0,60,120,180,240,300]) # should be changed according to frame_size
            # plt.ylabel('Voltage [uV]')
            # plt.xlabel('Time [sec]')
            # os.makedirs(os.path.join(figure_folder, 'STFT figures'), exist_ok=True)
            # c3_filename = os.path.join(figure_folder, 'STFT figures', str(y1[j]) + '_' + str(y2[j]) + '_' + str(z[j]) + '_' + str(j) + '_' + str(i) + '_' + 'c3.png')
            # plt.savefig(c3_filename)
            # plt.close()

            STFT = list()
            L = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18]
            for m in range(len(L)):
                if L[m] is c3:
                    STFT.append(STFT1(L[m],1, figure_folder, j, i, y1[j], y2[j], z[j]))
                else:
                    STFT.append(STFT1(L[m],0, figure_folder, j, i, y1[j], y2[j], z[j]))

            frames.append(np.dstack((STFT[0], STFT[1], STFT[2], STFT[3], STFT[4], STFT[5], STFT[6], STFT[7], STFT[8], STFT[9], STFT[10], STFT[11], STFT[12], STFT[13], STFT[14], 
                                     STFT[15], STFT[16], STFT[17])))
            
            outcomes.append(y1[j])
            cpcs.append(y2[j])
            patients.append(z[j])
    
    # bring the segment into a better shape
    frames = np.asarray(frames).reshape(-1, np.shape(STFT[0])[0], np.shape(STFT[0])[1], 18)
    outcomes = np.asarray(outcomes)
    cpcs = np.asarray(cpcs)
    patients = np.asarray(patients)
    
    return frames, outcomes, cpcs, patients

# custom loss function
def custom_loss(y_true, y_pred):

    log_y_pred = tf.math.log(y_pred)
    elements = -tf.math.multiply_no_nan(x=log_y_pred, y=y_true)
    loss_ce = tf.reduce_mean(tf.reduce_sum(elements,axis=1))

    label_true = tf.math.argmax(y_true, axis = 1)
    label_pred = tf.math.argmax(y_pred, axis = 1)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(label_true)):

        if label_pred[i] == 1 and label_true[i] == 1:

            tp += 1

        elif label_pred[i] == 1 and label_true[i] == 0:

            fp += 1

        elif label_pred[i] == 0 and label_true[i] == 0:

            tn += 1

        else:

            fn += 1

    loss_FPR = float(fp) / float(fp + tn)

    return loss_ce + loss_FPR

# focal loss function 
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.keras.sum(alpha * tf.keras.pow(1 - pt_1, gamma) * tf.keras.log(pt_1)) - tf.keras.sum((1 - alpha) * tf.keras.pow(pt_0, gamma) * tf.keras.log(1. - pt_0))

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def macro_double_soft_f1_loss(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)

    # soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)#1.25*tp / (1.25*tp + 0.25*fn + fp + 1e-16)
    # soft_f1_class0 = 2*tn / (2*tn + fn + 1*fp + 1e-16) #2*tn / (2*tn + fn + fp + 1e-16)
    # cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    # cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    # cost = 0.5 * (2*cost_class1 + 0*cost_class0) # take into account both class 1 and class 0
    # macro_cost = tf.reduce_mean(cost) # average on all labels

    #macro_cost = tf.reduce_mean(0.5 * (100*fp/(fp+tn+1e-16) + 1 - tp/(tp+fn+1e-16)))

    fpr = fp/(fp+tn+1e-16)
    tpr = tp/(tp+fn+1e-16)
    macro_cost = tf.reduce_mean(fpr-tpr+1)
    # if tf.cast(tf.reduce_mean(fpr), tf.float32) > 0.05:
    #     macro_cost = tf.reduce_mean(fpr)# + 1 - tpr)
    # else:
    #     macro_cost = tf.reduce_mean(1 - tpr)

    return macro_cost

def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        1-cost(scalar Tensor: value of the cost function for the batch)
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)

    # soft_f1_class1 = 1.25*tp / (1.25*tp + 0.25*fn + fp + 1e-16)#1.04*tp / (1.04*tp + 0.04*fn + fp + 1e-16)#1.0625*tp / (1.0625*tp + 0.0625*fn + fp + 1e-16)#1.01*tp / (1.01*tp + 0.01*fn + fp + 1e-16)#10/9*tp / (10/9*tp + 1/9*fn + fp + 1e-16)# 1.25*tp / (1.25*tp + 0.25*fn + fp + 1e-16) # 2*tp / (2*tp + fn + fp + 1e-16)
    # soft_f1_class0 = 2*tn / (2*tn + fn + 1*fp + 1e-16)#101*tn / (101*tn + fn + 100*fp + 1e-16)#10*tn / (10*tn + fn + 9*fp + 1e-16)# 5*tn / (5*tn + fn + 4*fp + 1e-16) # 2*tn / (2*tn + fn + fp + 1e-16)
    # cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    # cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    # cost = 0.5 * (2*cost_class1 + 0*cost_class0) # take into account both class 1 and class 0
    # macro_cost = tf.reduce_mean(cost) # average on all labels

    #macro_cost = tf.reduce_mean(0.5 * (100*fp/(fp+tn+1e-16) + 1 - tp/(tp+fn+1e-16)))

    fpr = fp/(fp+tn+1e-16)
    #tpr = tp/(tp+fn+1e-16)
    macro_cost = tf.reduce_mean(fpr)
    # if tf.cast(tf.reduce_mean(fpr), tf.float32) > 0.05:
    #     macro_cost = tf.reduce_mean(fpr)# + 1 - tpr) # 2*fpr - tpr #tf.reduce_mean(1 * (2*(1 + fpr) + 1*(1 - tpr)))
    # else:
    #     macro_cost = tf.reduce_mean(1 - tpr)

    return macro_cost # 1-macro_cost

def macro_double_soft_tpr(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        1-cost(scalar Tensor: value of the cost function for the batch)
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)

    # soft_f1_class1 = 1.25*tp / (1.25*tp + 0.25*fn + fp + 1e-16)#1.04*tp / (1.04*tp + 0.04*fn + fp + 1e-16)#1.0625*tp / (1.0625*tp + 0.0625*fn + fp + 1e-16)#1.01*tp / (1.01*tp + 0.01*fn + fp + 1e-16)#10/9*tp / (10/9*tp + 1/9*fn + fp + 1e-16)# 1.25*tp / (1.25*tp + 0.25*fn + fp + 1e-16) # 2*tp / (2*tp + fn + fp + 1e-16)
    # soft_f1_class0 = 2*tn / (2*tn + fn + 1*fp + 1e-16)#101*tn / (101*tn + fn + 100*fp + 1e-16)#10*tn / (10*tn + fn + 9*fp + 1e-16)# 5*tn / (5*tn + fn + 4*fp + 1e-16) # 2*tn / (2*tn + fn + fp + 1e-16)
    # cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    # cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    # cost = 0.5 * (2*cost_class1 + 0*cost_class0) # take into account both class 1 and class 0
    # macro_cost = tf.reduce_mean(cost) # average on all labels

    #macro_cost = tf.reduce_mean(0.5 * (100*fp/(fp+tn+1e-16) + 1 - tp/(tp+fn+1e-16)))

    #fpr = fp/(fp+tn+1e-16)
    tpr = tp/(tp+fn+1e-16)
    macro_cost = tf.reduce_mean(tpr)
    # if tf.cast(tf.reduce_mean(fpr), tf.float32) > 0.05:
    #     macro_cost = tf.reduce_mean(fpr)# + 1 - tpr) # 2*fpr - tpr #tf.reduce_mean(1 * (2*(1 + fpr) + 1*(1 - tpr)))
    # else:
    #     macro_cost = tf.reduce_mean(1 - tpr)

    return macro_cost # 1-macro_cost

########## Generate CNN Model
def generate_cnn(k, X_all, feature_all):
    
    # model = Sequential()
    # model.add(Conv2D(16, (3,3), padding = 'same', activation='relu', input_shape=(np.shape(X_all)[1],np.shape(X_all)[2],np.shape(X_all)[3]), kernel_initializer='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    # model.add(Conv2D(32, (3,3), padding = 'same', activation='relu', kernel_initializer='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    # #model.add(Dropout(0.1))
    # model.add(Conv2D(64, (3,3), padding = 'same', activation='relu', kernel_initializer='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    # #model.add(Dropout(0.1))
    # model.add(Conv2D(128, (3,3), padding = 'same', activation='relu', kernel_initializer='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    # model.add(Conv2D(256, (3,3), padding = 'same', activation='relu', kernel_initializer='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    # # model.add(Conv2D(128, (3,3), padding = 'same', activation='relu', kernel_initializer='he_normal'))
    # # model.add(BatchNormalization(axis=-1))
    # # model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    # # model.add(Dropout(0.25))
    # # # add one layer
    # # model.add(Conv2D(256, (3,3), padding = 'same', activation='relu', kernel_initializer='he_normal'))
    # # model.add(BatchNormalization(axis=-1))
    # # model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))#model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.75))
    # model.add(Dense(k, activation='softmax', kernel_initializer='he_normal'))

    EEG_input = Input(shape=(np.shape(X_all)[1],np.shape(X_all)[2],np.shape(X_all)[3]))

    model = Conv2D(8, (3,3), padding = 'same', activation='relu', kernel_initializer='he_normal')(EEG_input)
    #model = tf.keras.layers.LeakyReLU(alpha=0.01)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPool2D(pool_size=(2,2), padding = 'same')(model)

    model = Conv2D(16, (3,3), padding = 'same', activation='relu', kernel_initializer='he_normal')(model)
    #model = tf.keras.layers.LeakyReLU(alpha=0.01)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPool2D(pool_size=(2,2), padding = 'same')(model)
    model = Dropout(0.1)(model)

    model = Conv2D(32, (3,3), padding = 'same', activation='relu', kernel_initializer='he_normal')(model)
    #model = tf.keras.layers.LeakyReLU(alpha=0.01)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPool2D(pool_size=(2,2), padding = 'same')(model)
    model = Dropout(0.1)(model)

    # model = Conv2D(128, (3,3), padding = 'same', activation='relu', kernel_initializer='lecun_uniform')(model)
    # #model = tf.keras.layers.LeakyReLU(alpha=0.01)(model)
    # model = BatchNormalization(axis=-1)(model)
    # model = MaxPool2D(pool_size=(2,2), padding = 'same')(model)

    # model = Conv2D(256, (3,3), padding = 'same', activation='relu', kernel_initializer='lecun_uniform')(model)
    # #model = tf.keras.layers.LeakyReLU(alpha=0.01)(model)
    # model = BatchNormalization(axis=-1)(model)
    # model = MaxPool2D(pool_size=(2,2), padding = 'same')(model)

    # model = Conv2D(512, (3,3), padding = 'same', activation='relu', kernel_initializer='lecun_uniform')(model)
    # #model = tf.keras.layers.LeakyReLU(alpha=0.01)(model)
    # model = BatchNormalization(axis=-1)(model)
    # model = MaxPool2D(pool_size=(2,2), padding = 'same')(model)

    model = Flatten()(model)

    # # patient features
    new_input = Input(shape=(np.shape(feature_all)[1],))
    # model = Concatenate(axis=1)([model,new_input])

    model = Dense(32, activation='relu', kernel_initializer='he_normal')(model)
    #model = tf.keras.layers.LeakyReLU(alpha=0.01)(model)
    #model = BatchNormalization()(model)
    model = Dropout(0.2)(model)

    # model = Dense(1024, activation='relu', kernel_initializer='lecun_uniform')(model)
    #model = tf.keras.layers.LeakyReLU(alpha=0.01)(model)

    res = Dense(k, activation='softmax', kernel_initializer='he_normal')(model)

    new_model = Model(inputs=[EEG_input,new_input], outputs=res) 

    new_model.summary()

    #Model compiler settings
    sgd =tf.keras.optimizers.legacy.SGD(lr=0.001) #, decay=1e-6, momentum=0.9 #lr=0.000001
    #RMSprop = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001)#tf.keras.optimizers.experimental.RMSprop(lr=0.001)
    if k ==2:
        new_model.compile(optimizer = sgd,#RMSprop,#tf.keras.optimizers.Adam(0.001),
              loss=macro_double_soft_f1_loss,#f1_loss,#BinaryFocalLoss(gamma=2, pos_weight=0.15),#focal_loss(), #tfr.keras.losses.PairwiseHingeLoss(),#custom_loss(), #'categorical_crossentropy',
              metrics=[macro_double_soft_f1,tf.keras.metrics.Recall()]) # macro_double_soft_tpr, #, tf.keras.metrics.AUC() # tf.keras.metrics.SensitivityAtSpecificity(specificity=0.95),  #['accuracy'])
    else:
        new_model.compile(optimizer = sgd,#tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error', #'categorical_crossentropy',
              metrics=['mse']) #['accuracy'])        
    
    return new_model

def fit_and_eval(X_train, X_val,fea_train, fea_val,y_train, y_val,model,epochs,batch_size,early_stopping,model_checkpoint):

    cnn_model = None
    cnn_model = model
    results = cnn_model.fit([X_train, fea_train], y_train, epochs=epochs,batch_size=batch_size,callbacks = [model_checkpoint], verbose=1, validation_data = ([X_val, fea_val],y_val)) # early_stopping,
    val_score = cnn_model.evaluate([X_val, fea_val], y_val,batch_size=batch_size)
    print("Val_score: ", val_score)
    
    return results,val_score  

def plot_figures(A, n_folds,model_history, model_folder):

    # # plot training and validation accuracy
    # n_folds = n_folds
    # for i in range(n_folds):
    #     plt.clf()
    #     plt.title('Fold ' + str(i+1) + ': Training accuracy v.s. Validation accuracy for ' + A + ' model with nested 10-fold cross validation')
    #     plt.ylabel('Accuracy')
    #     plt.xlabel('Number of epoch')
    #     plt.plot(model_history[i].history['accuracy'], label = 'Training Accuracy')
    #     plt.plot(model_history[i].history['val_accuracy'], label = 'Validation Accuracy', linestyle = 'dashdot')
    #     plt.legend()
    #     fig_name = os.path.join(model_folder, 'Fold ' + str(i+1) + ' Training Accuracy v.s. Validation Accuracy for ' + A + ' model with nested 10-fold cv.png')
    #     plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    #     plt.close()

    # plot training and validation loss
    n_folds = n_folds
    for i in range(n_folds):
        plt.clf()
        #plt.title('Fold ' + str(i+1) + ': Training Loss v.s. Validation Loss for ' + A + ' model with nested 10-fold cross validation')
        #plt.ylabel('Loss')
        plt.xlabel('Number of epoch')
        plt.plot(model_history[i].history['loss'], label = 'Training Loss')
        plt.plot(model_history[i].history['val_loss'], label = 'Validation Loss', linestyle = 'dashdot')
        if A == 'outcome':
            plt.plot(model_history[i].history['recall'], label = 'Training recall')
            plt.plot(model_history[i].history['val_recall'], label = 'Validation recall', linestyle = 'dashdot')
            plt.plot(model_history[i].history['macro_double_soft_f1'], label = 'Training fpr')
            plt.plot(model_history[i].history['val_macro_double_soft_f1'], label = 'Validation fpr', linestyle = 'dashdot')
            # plt.plot(model_history[i].history['macro_double_soft_tpr'], label = 'Training tpr')
            # plt.plot(model_history[i].history['val_macro_double_soft_tpr'], label = 'Validation tpr', linestyle = 'dashdot')
            fig_name = os.path.join(model_folder, 'Fold ' + str(i+1) + ' Training metrics v.s. Validation metrics for ' + A + ' model with nested 10-fold cv.png')
        else:
            fig_name = os.path.join(model_folder, 'Fold ' + str(i+1) + ' Training Loss v.s. Validation Loss for ' + A + ' model with nested 10-fold cv.png')
        plt.legend()
        plt.savefig(fig_name, dpi=200, bbox_inches='tight')
        plt.close()

#def get_features(patient_metadata, recording_metadata, recording_data):
# Extract reshaped recordings from the data.
def get_recordings(patient_metadata, recording_metadata, recording_data,b,a): 
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    #num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            # reserve three common frequency bands (delta, 1–4 Hz; theta, 4–8 Hz and alpha, 8–13 Hz)
            filted_data = signal_data
            for i in range(np.shape(signal_data)[0]):
                filted_data[i,:] = signal.filtfilt(b,a,signal_data[i,:])
            available_signal_data.append(filted_data.T) # size: num_samples * num_channels
    if len(available_signal_data) > 0:
        available_signal_data = np.vstack(available_signal_data) 
    else:
        pass

    # if len(available_signal_data) > 0:
    #     available_signal_data = np.hstack(available_signal_data)
    #     signal_mean = np.nanmean(available_signal_data, axis=1)
    #     signal_std  = np.nanstd(available_signal_data, axis=1)
    # else:
    #     signal_mean = float('nan') * np.ones(num_channels)
    #     signal_std  = float('nan') * np.ones(num_channels)

    # # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # # recent recording.
    # index = None
    # for i in reversed(range(num_recordings)):
    #     signal_data, sampling_frequency, signal_channels = recording_data[i]
    #     if signal_data is not None:
    #         index = i
    #         break

    # if index is not None:
    #     signal_data, sampling_frequency, signal_channels = recording_data[index]
    #     signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

    #     delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
    #     theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
    #     alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
    #     beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

    #     delta_psd_mean = np.nanmean(delta_psd, axis=1)
    #     theta_psd_mean = np.nanmean(theta_psd, axis=1)
    #     alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
    #     beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

    #     quality_score = get_quality_scores(recording_metadata)[index]
    # else:
    #     delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
    #     quality_score = float('nan')

    # recording_features = np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    # # Combine the features from the patient metadata and the recording data and metadata.
    # features = np.hstack((patient_features, recording_features))

    return available_signal_data, patient_features#, features
