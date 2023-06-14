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

#import pickle
from scipy import signal
import tensorflow as tf
#import tensorflow_ranking as tfr
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
#import time
import cv2
import shutil
import random
#from collections import Counter

# import resource
# soft, hard = resource.getrlimit(resource.RLIMIT_AS)
# resource.setrlimit(resource.RLIMIT_AS, (68719476736, hard)) # set the maximum memory usage: 64 GB

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
    outcomes = list()
    cpcs = list()
    patients = list() # created for cross validation
    #patient_infos = list() # save patient metadata

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # Extract reshaped recordings.
        b,a = signal.butter(4, [0.02,0.26], 'bandpass')
        current_recordings = get_recordings(patient_metadata, recording_metadata, recording_data,b,a)
        recordings.append(current_recordings)
        #patient_infos.append(patient_info)

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
                    STFT.append(cv2.resize(img, (128, 128), interpolation = cv2.INTER_LINEAR)) # interpolation = cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
                    #filename = os.path.join(data_folder, 'resized image.png')
                    #cv2.imwrite(filename, cv2.resize(img, (128, 128), interpolation = cv2.INTER_LINEAR))
                else:
                    img = STFT1(x_train[:,m],0, model_folder, j, i, outcomes[j], cpcs[j], patients[j])
                    STFT.append(cv2.resize(img, (128, 128), interpolation = cv2.INTER_LINEAR)) # interpolation = cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
            
            frames.append(np.dstack((STFT[0], STFT[1], STFT[2], STFT[3], STFT[4], STFT[5], STFT[6], STFT[7], STFT[8], STFT[9], STFT[10], STFT[11], STFT[12], STFT[13], STFT[14], 
                                        STFT[15], STFT[16], STFT[17])))
        
        # bring the segment into a better shape
        X_all = np.asarray(frames).reshape(-1, np.shape(STFT[0])[0], np.shape(STFT[0])[1],18)
        y1_all = np.asarray(outcomes)
        y2_all = np.asarray(cpcs)
        z_all = np.asarray(patients)

    # Generate outcome model and cpc model
    outcome_model = generate_cnn(2, X_all) # 2 labels: good, poor
    cpc_model = generate_cnn(5, X_all) # 5 labels: 1, 2, 3, 4, 5

    # set early stopping criteria
    pat = 5#10 # the number of epochs with no improvement after which the training will stop
    early_stopping =  EarlyStopping(monitor='val_loss',patience=pat, verbose=1, restore_best_weights=True)

    res = list()
    res_cpc = list()
    outcome_model_history = list()
    cpc_model_history = list()
    num_trial = 1
    for T in range(num_trial): # run 5 times to save memory

        # save the models as physical files
        os.makedirs(os.path.join(model_folder, 'Trial ' + str(T)), exist_ok=True) # Create a folder for the Challenge outputs if it does not already exist.
        outcome_filename = os.path.join(model_folder, 'Trial ' + str(T), '{epoch:03d}_{val_recall:.4f}_' + 'outcome_model.h5')
        cpc_filename = os.path.join(model_folder, 'Trial ' + str(T), 'cpc_model.h5')
        oucome_model_checkpoint = ModelCheckpoint(filepath = outcome_filename, verbose=1, save_best_only=False)
        cpc_model_checkpoint = ModelCheckpoint(filepath = cpc_filename, verbose=1, save_best_only=True)

        # model training
        if verbose >= 1:
            print('Trial '+ str(T) + ': Training the Challenge models on the Challenge data...')

        # dt = list()
        # t1 = time.time()
        epochs = 150#300#200#120
        batch_size = 64#32#64

        X_train = X_all
        y1_train = y1_all
        y2_train = y2_all
        X_train = np.asarray(X_train).reshape(-1, np.shape(X_train[0])[0],np.shape(X_train[0])[1],np.shape(X_train[0])[2])
        y1_train = np.asarray(y1_train)
        y2_train = np.asarray(y2_train)
    
        # one-hot convertion
        y1_train = tf.keras.utils.to_categorical(y1_train)
        y2_train = tf.keras.utils.to_categorical(y2_train-1)   

        # train models   
        outcome_results, outcome_val_score= fit_and_eval(int(0),X_train, y1_train, z_all, outcome_model, epochs, batch_size, early_stopping, oucome_model_checkpoint)
        res.append(outcome_val_score[1])
        outcome_model_history.append(outcome_results)
        cpc_results, cpc_val_score = fit_and_eval(int(1),X_train, y2_train, z_all, cpc_model, epochs, batch_size, early_stopping, cpc_model_checkpoint)
        res_cpc.append(cpc_val_score[1])
        cpc_model_history.append(cpc_results)

        # outcome_train_label = list()
        # outcome_val_label = list()
        # outcome_train_patient = list()
        # outcome_val_patient = list()
        # for p in range(X_train.shape[0]):
        #     if p in outcome_indices[0]:
        #         outcome_train_label.append(y1_all[p])
        #         outcome_train_patient.append(z_all[p])
        #     else:
        #         outcome_val_label.append(y1_all[p])
        #         outcome_val_patient.append(z_all[p])

        # cpc_train_label = list()
        # cpc_val_label = list()
        # cpc_train_patient = list()
        # cpc_val_patient = list()
        # for q in range(X_train.shape[0]):
        #     if q in cpc_indices[0]:
        #         cpc_train_label.append(y2_all[q])
        #         cpc_train_patient.append(z_all[q])
        #     else:
        #         cpc_val_label.append(y2_all[q])
        #         cpc_val_patient.append(z_all[q])
    
        # t2 = time.time()
        # dt = t2 - t1
        # print("Running time(s): ", dt)

        # # save results
        # results = {'Running_time_s': dt, 'Outcome_validation_accuracy': outcome_val_score, 'Cpc_validation accuracy': cpc_val_score}#, 
        #         #    'outcome_train_label': outcome_train_label, 'outcome_train_patient': outcome_train_patient, 'outcome_val_label': outcome_val_label, 
        #         #    'outcome_val_patient': outcome_val_patient, 'cpc_train_label': cpc_train_label, 'cpc_train_patient': cpc_train_patient, 'cpc_val_label': cpc_val_label, 
        #         #    'cpc_val_patient': cpc_val_patient}
        # file_name = os.path.join(model_folder, 'Trial ' + str(T), 'final_results_nested_holdout.pkl')
        # pickle.dump(results,open(file_name,"wb"))

    ##### plot training and validation accuracy and loss for outcome model and cpc model
    plot_figures('outcome', num_trial, outcome_model_history, model_folder)
    plot_figures('cpc', num_trial, cpc_model_history, model_folder)

    # save the optimal model
    os.makedirs(os.path.join(model_folder, 'Optimal'), exist_ok=True)
    outcome_name = os.path.join(model_folder, 'Optimal', 'outcome_model.h5')
    cpc_name = os.path.join(model_folder, 'Optimal', 'cpc_model.h5')
    fold_best_model = list()
    fold_best_metric = list()
    for i in range(num_trial):
        temp = list()
        temp = 1 - np.asarray(outcome_model_history[i].history['val_custom_fpr'])#1 - np.asarray(outcome_model_history[i].history['val_loss'])#outcome_model_history[i].history['val_recall']
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
        fold_best_model.append(os.path.join(model_folder, 'Trial ' + str(i), str(format(final_ind + 1, '03d'))+'_'+str(format(1-temp2[final_ind],'.4f'))+'_'+'outcome_model.h5'))
        #fold_best_model.append(os.path.join(model_folder, '10-fold CV', 'Fold_'+str(i+1)+'_'+str(format(final_ind + 1, '03d'))+'_'+str(format(outcome_model_history[i].history['val_recall'][final_ind],'.4f'))+'_outcome_KFold.h5'))
    Ind = list()
    Temp = list()
    for k in range(len(fold_best_model)):
        if fold_best_metric[k][0] == np.max(fold_best_metric,axis=0)[0] or fold_best_metric[k][0] >= 0.95:
            Ind.append(k)
            Temp.append(fold_best_metric[k][1])
        else:
            pass
    shutil.copyfile(fold_best_model[Ind[np.argmin(Temp)]], outcome_name)
    shutil.copyfile(os.path.join(model_folder, 'Trial ' + str(np.argmin(res_cpc)), 'cpc_model.h5'), cpc_name)
    print('Save the optimal model finished.')

    # index = np.argmax(res)
    # new_folder_name = os.path.join(model_folder, 'Optimal')
    # shutil.copytree(os.path.join(model_folder, 'Trial ' + str(index)), new_folder_name)
    # print('Save the optimal model finished.')

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):

    outcome_filename = os.path.join(model_folder, 'Optimal', 'outcome_model.h5')
    cpc_filename = os.path.join(model_folder, 'Optimal', 'cpc_model.h5')
    return [load_model(outcome_filename, custom_objects={'custom_loss': custom_loss,'custom_fpr': custom_fpr}), load_model(cpc_filename)]

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):

    # load models
    outcome_model = models[0]
    cpc_model = models[1]

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # Extract reshaped recording.
    current_recording = list()
    b,a = signal.butter(4, [0.02,0.26], 'bandpass')
    current_recording.append(get_recordings(patient_metadata, recording_metadata, recording_data,b,a))

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
            X_test = np.asarray(x_test).reshape(-1, np.shape(x_test[0])[0],np.shape(x_test[0])[1],np.shape(x_test[0])[2])

            # Apply models to test data.
            y1_pred = outcome_model.predict(X_test)
            outcome_probability = list()
            for i in range(len(y1_pred)):
                outcome_probability.append(y1_pred[i][1])
            y1_pred = np.argmax(y1_pred, axis = 1)

            y2_pred = cpc_model.predict(X_test)
            y2_pred = np.argmax(y2_pred, axis = 1)

            # get signal quality scores
            quality_score = get_quality_scores(recording_metadata)
            while np.nan in quality_score:
                quality_score.remove(np.nan)

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
                    STFT.append(cv2.resize(img, (128, 128), interpolation = cv2.INTER_LINEAR)) # interpolation = cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
                    #filename = os.path.join(data_folder, 'resized image.png')
                    #cv2.imwrite(filename, cv2.resize(img, (128, 128), interpolation = cv2.INTER_LINEAR))
                else:
                    img = STFT1(x_test[:,m],0, data_folder, j, i, 'Nan', 'Nan', patient_id)
                    STFT.append(cv2.resize(img, (128, 128), interpolation = cv2.INTER_LINEAR)) # interpolation = cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
            
            frames.append(np.dstack((STFT[0], STFT[1], STFT[2], STFT[3], STFT[4], STFT[5], STFT[6], STFT[7], STFT[8], STFT[9], STFT[10], STFT[11], STFT[12], STFT[13], STFT[14], 
                                                STFT[15], STFT[16], STFT[17])))
                
            # bring the frames into a better shape
            X_test = np.asarray(frames).reshape(-1, np.shape(STFT[0])[0], np.shape(STFT[0])[1],18)

            # # Apply models to test data.
            y1_pred = outcome_model.predict(X_test)
            print(y1_pred)
            outcome_probability = y1_pred[0][1]
            print(outcome_probability)
            y1_pred = np.argmax(y1_pred)
            print(y1_pred)

            y2_pred = cpc_model.predict(X_test)
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
def custom_loss(y, y_hat):

    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)

    fpr = fp/(fp+tn+1e-16)
    tpr = tp/(tp+fn+1e-16)
    macro_cost = tf.reduce_mean(fpr-tpr+1)# tf.reduce_mean(fpr)

    return macro_cost

# custom metric function
def custom_fpr(y, y_hat):

    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)

    fpr = fp/(fp+tn+1e-16)
    macro_cost = tf.reduce_mean(fpr)

    return macro_cost

########## Generate CNN Model
def generate_cnn(k, X_all):
    
    model = Sequential()
    model.add(Conv2D(8, (3,3), padding = 'same', activation='relu', kernel_initializer='lecun_uniform', input_shape=(np.shape(X_all)[1],np.shape(X_all)[2],np.shape(X_all)[3])))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))

    model.add(Conv2D(16, (3,3), padding = 'same', kernel_initializer='lecun_uniform', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))

    model.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer='lecun_uniform', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    #model.add(Dropout(0.1))

    model.add(Conv2D(64, (3,3), padding = 'same', kernel_initializer='lecun_uniform', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    #model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding = 'same', kernel_initializer='lecun_uniform', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))

    # model.add(Conv2D(256, (3,3), padding = 'same', kernel_initializer='lecun_uniform', activation='relu'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))

    model.add(Flatten())

    model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
    model.add(Dropout(0.75))

    # model.add(Dense(512, activation='relu', kernel_initializer='lecun_uniform'))

    model.add(Dense(k, activation='softmax', kernel_initializer='lecun_uniform'))
    model.summary()

    #Model compiler settings
    if k ==2:
        model.compile(optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.001),#tf.keras.optimizers.Adam(0.0005),
              loss=custom_loss, #tfr.keras.losses.ApproxNDCGLoss(), #'categorical_crossentropy',
              metrics=[custom_fpr,tf.keras.metrics.Recall()]) #['accuracy'])
    else:
        model.compile(optimizer = tf.keras.optimizers.Adam(0.01),
              loss='mean_squared_error', #'categorical_crossentropy',
              metrics=['mse']) #['accuracy'])        
    
    return model

def fit_and_eval(flag,X,y,z,model,epochs,batch_size,early_stopping,model_checkpoint):
    
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, stratify=y) # randomly select 80% for training, the rest 20% for validation 
    # indices = np.arange(X.shape[0])
    # X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X,y,indices,test_size=0.2, stratify=y)

    cnn_model = model
    
    if flag == 0:
        results = cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks = [model_checkpoint], verbose=1, validation_data = (X_val,y_val))
    else:
         results = cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks = [early_stopping,model_checkpoint], verbose=1, validation_data = (X_val,y_val))

    val_score = cnn_model.evaluate(X_val,y_val,batch_size=batch_size)
    
    print("Val_score: ", val_score)
    
    return results, val_score#, [indices_train, indices_val]

def plot_figures(A, n_folds,model_history, model_folder):

    # # plot training and validation accuracy
    # plt.clf()
    # plt.title('Training Accuracy v.s. Validation Accuracy for ' + A + ' model with nested holdout')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Number of epoch')
    # n_folds = n_folds
    # for i in range(n_folds):
    #     plt.plot(model_history[i].history['accuracy'], label = 'Training Accuracy')
    #     plt.plot(model_history[i].history['val_accuracy'], label = 'Validation Accuracy', linestyle = 'dashdot')
    #     plt.legend()
    #     fig_name = os.path.join(model_folder, 'Trial ' + str(T), 'Training Accuracy v.s. Validation Accuracy for ' + A + ' model with nested holdout.png')
    #     plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    #     plt.close()

    # plot training and validation loss
    n_folds = n_folds
    for i in range(n_folds):
        plt.clf()
        #plt.title('Training Loss v.s. Validation Loss for ' + A + ' model with nested holdout')
        #plt.ylabel('Loss')
        plt.xlabel('Number of epoch')
        plt.plot(model_history[i].history['loss'], label = 'Training Loss')
        plt.plot(model_history[i].history['val_loss'], label = 'Validation Loss', linestyle = 'dashdot')
        if A == 'outcome':
            plt.plot(model_history[i].history['recall'], label = 'Training recall')
            plt.plot(model_history[i].history['val_recall'], label = 'Validation recall', linestyle = 'dashdot')     
            fig_name = os.path.join(model_folder, 'Trial ' + str(i+1) + ' Training metrics v.s. Validation metrics for ' + A + ' model with nested 10-fold cv.png')
        else:
            fig_name = os.path.join(model_folder, 'Trial ' + str(i+1) + ' Training Loss v.s. Validation Loss for ' + A + ' model with nested 10-fold cv.png')   
        plt.legend()
        #fig_name = os.path.join(model_folder, 'Trial ' + str(T), 'Training Loss v.s. Validation Loss for ' + A + ' model with nested holdout.png')
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

    return available_signal_data#, features
