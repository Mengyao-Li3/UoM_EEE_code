#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os#, sys
import mne
#from sklearn.impute import SimpleImputer
#from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#import joblib

#import pickle
from scipy import signal
import tensorflow as tf
#import tensorflow_ranking as tfr
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
#import time
import cv2
import shutil
import random
#from collections import Counter

import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (68719476736, hard)) # set the maximum memory usage: 64 GB

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

    EEG_recordings = list()
    ECG_recordings = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Extract reshaped recordings
        EEG_current_recording, ECG_curent_recording, EEG_sampling_frequency, ECG_sampling_frequency = get_recordings(data_folder, patient_ids[i])
        EEG_recordings.append(EEG_current_recording)
        ECG_recordings.append(ECG_curent_recording)

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    #reshape data and labels
    if verbose >= 1:
       print('Employing short-time Fourier transform...')

    X_all, y1_all, y2_all = data_reshape(model_folder, EEG_recordings, ECG_recordings, outcomes, cpcs, EEG_sampling_frequency, ECG_sampling_frequency)     

    # Generate outcome model and cpc model
    outcome_model = generate_cnn(2, X_all) # 2 labels: good, poor
    cpc_model = generate_cnn(5, X_all) # 5 labels: 1, 2, 3, 4, 5

    # set early stopping criteria
    pat = 10#5#10 # the number of epochs with no improvement after which the training will stop
    early_stopping =  EarlyStopping(monitor='val_loss',patience=pat, verbose=1, restore_best_weights=True)

    res = list()
    res_cpc = list()
    outcome_model_history = list()
    cpc_model_history = list()
    num_trial = 5#1
    for T in range(num_trial): # run 5 times to save memory

        # save the models as physical files
        os.makedirs(os.path.join(model_folder, 'Trial ' + str(T)), exist_ok=True) # Create a folder for the Challenge outputs if it does not already exist.

        # outcome_filename = os.path.join(model_folder, 'Trial ' + str(T), '{epoch:03d}_{val_recall:.4f}_' + 'outcome_model.h5')
        # oucome_model_checkpoint = ModelCheckpoint(filepath = outcome_filename, verbose=1, save_best_only=False)
        outcome_filename = os.path.join(model_folder, 'Trial ' + str(T), 'outcome_model.h5')
        oucome_model_checkpoint = ModelCheckpoint(filepath = outcome_filename, verbose=1, save_best_only=True)

        cpc_filename = os.path.join(model_folder, 'Trial ' + str(T), 'cpc_model.h5')
        cpc_model_checkpoint = ModelCheckpoint(filepath = cpc_filename, verbose=1, save_best_only=True)

        # model training
        if verbose >= 1:
            print('Trial '+ str(T) + ': Training the Challenge models on the Challenge data...')

        # dt = list()
        # t1 = time.time()
        epochs = 150#300#200#120
        batch_size = 32#64#32#64

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

        # outcome_results, outcome_val_score= fit_and_eval(int(0),X_train, y1_train, outcome_model, epochs, batch_size, early_stopping, oucome_model_checkpoint)
        outcome_results, outcome_val_score= fit_and_eval(int(1),X_train, y1_train, outcome_model, epochs, batch_size, early_stopping, oucome_model_checkpoint)

        res.append(outcome_val_score[1])
        outcome_model_history.append(outcome_results)
        cpc_results, cpc_val_score = fit_and_eval(int(1),X_train, y2_train, cpc_model, epochs, batch_size, early_stopping, cpc_model_checkpoint)
        res_cpc.append(cpc_val_score[1])
        cpc_model_history.append(cpc_results)

    ##### plot training and validation accuracy and loss for outcome model and cpc model
    plot_figures('outcome', num_trial, outcome_model_history, model_folder)
    plot_figures('cpc', num_trial, cpc_model_history, model_folder)

    # save the optimal model
    os.makedirs(os.path.join(model_folder, 'Optimal'), exist_ok=True)
    outcome_name = os.path.join(model_folder, 'Optimal', 'outcome_model.h5')
    cpc_name = os.path.join(model_folder, 'Optimal', 'cpc_model.h5')

    # fold_best_model = list()
    # fold_best_metric = list()
    # for i in range(num_trial):
    #     temp = list()
    #     temp = 1 - np.asarray(outcome_model_history[i].history['val_custom_fpr'])#1 - np.asarray(outcome_model_history[i].history['val_loss'])#outcome_model_history[i].history['val_recall']
    #     temp2 = list()
    #     temp2 = 1-np.asarray(outcome_model_history[i].history['val_recall'])#1 - np.asarray(outcome_model_history[i].history['val_macro_double_soft_tpr'])#outcome_model_history[i].history['val_loss']
    #     temp_ind = list()
    #     temp3 = list()
    #     for j in range(len(temp)):
    #         if temp[j] == np.max(temp) or temp[j] >= 0.95:
    #             temp_ind.append(j)
    #             temp3.append(temp2[j])
    #         else:
    #             pass
    #     final_ind = temp_ind[np.argmin(temp3)]
    #     fold_best_metric.append([temp[final_ind],temp2[final_ind]])
    #     print("epoch, val_fpr, val_recall:", final_ind + 1, 1-temp[final_ind], 1-temp2[final_ind])
    #     fold_best_model.append(os.path.join(model_folder, 'Trial ' + str(i), str(format(final_ind + 1, '03d'))+'_'+str(format(1-temp2[final_ind],'.4f'))+'_'+'outcome_model.h5'))
    #     #fold_best_model.append(os.path.join(model_folder, '10-fold CV', 'Fold_'+str(i+1)+'_'+str(format(final_ind + 1, '03d'))+'_'+str(format(outcome_model_history[i].history['val_recall'][final_ind],'.4f'))+'_outcome_KFold.h5'))
    # Ind = list()
    # Temp = list()
    # for k in range(len(fold_best_model)):
    #     if fold_best_metric[k][0] == np.max(fold_best_metric,axis=0)[0] or fold_best_metric[k][0] >= 0.95:
    #         Ind.append(k)
    #         Temp.append(fold_best_metric[k][1])
    #     else:
    #         pass
    # shutil.copyfile(fold_best_model[Ind[np.argmin(Temp)]], outcome_name)
    shutil.copyfile(os.path.join(model_folder, 'Trial ' + str(np.argmax(res)), 'outcome_model.h5'), outcome_name)

    shutil.copyfile(os.path.join(model_folder, 'Trial ' + str(np.argmin(res_cpc)), 'cpc_model.h5'), cpc_name)
    print('Trial ' + str(np.argmax(res)))
    print('Trial ' + str(np.argmin(res_cpc)))
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

    # Reshape recordings.
    EEG_current_recording = list()
    ECG_curent_recording = list()

    current_recording_1, curent_recording_2, EEG_sampling_frequency, ECG_sampling_frequency = get_recordings(data_folder, patient_id)
    EEG_current_recording.append(current_recording_1)
    ECG_curent_recording.append(curent_recording_2)

    # if float('nan') in EEG_current_recording[0] and float('nan') in ECG_curent_recording[0]:
    #     print('No data was provided for patient ' + patient_id)
    #     return random.randint(0,1), 0.5, random.randint(1,5)
    # else:
    #     X_test = data_reshape(data_folder, EEG_current_recording, ECG_curent_recording, ['Nan'], ['Nan'], EEG_sampling_frequency, ECG_sampling_frequency)
    if np.isnan(EEG_current_recording[0]).all() and np.isnan(ECG_curent_recording[0]).all():
        print('No data was provided for patient ' + patient_id)
        return random.randint(0,1), 0.5, random.randint(1,5)
    else:
        X_test = data_reshape(data_folder, EEG_current_recording, ECG_curent_recording, ['Nan'], ['Nan'], EEG_sampling_frequency, ECG_sampling_frequency)

        # Apply models to test data.
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

# Save your trained model.
# def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
#     d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
#     filename = os.path.join(model_folder, 'models.sav')
#     joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency

# Extract reshaped recordings.
def get_recordings(data_folder, patient_id):
    # Load patient data.
    #patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    reduced_recording_ids = list()
    for i in recording_ids:
        #print(i)
        if int(i[9:])<72:
            reduced_recording_ids.append(i)
        else:
            pass

    # num_recordings = len(recording_ids)
    num_recordings = len(reduced_recording_ids)

    # Extract patient features.
    #patient_features = get_patient_features(patient_metadata)

    # Extract EEG recordings.
    EEG_data = list()
    EEG_sampling_frequency = 500

    # eeg_channels = ['F3', 'P3', 'F4', 'P4']
    eeg_channels = ['F3', 'T3', 'P3', 'F4', 'T4', 'P4']
    
    group = 'EEG'

    if num_recordings > 0:
        # recording_id = recording_ids[-1]
        recording_id = reduced_recording_ids[-1]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            if all(channel in channels for channel in eeg_channels):
                data, channels = reduce_channels(data, channels, eeg_channels)
                data, EEG_sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)

                # EEG_data = np.array([data[0, :] - data[1, :], data[2, :] - data[3, :]]) # Convert to bipolar montage: F3-P3 and F4-P4
                EEG_data = np.array([data[0, :] - data[1, :], data[1, :] - data[2, :], data[0, :] - data[2, :], data[3, :] - data[4, :], data[4, :] - data[5, :], data[3, :] - data[5, :]]) # Convert to bipolar montage: F3-T3, T3-P3, F3-P3, F4-T4, T4-P4, and F4-P4

                # data size: num_channels * num_samples 
                #eeg_features = get_eeg_features(data, sampling_frequency).flatten()
            else:
                #eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel
                #num_channels, num_samples = np.shape(data)
                print('NAN 1')
                # EEG_data = float('nan') * np.ones((2, 30000)) # 2 channels * 500 Hz * 60 s
                EEG_data = float('nan') * np.ones((6, 30000)) # 6 channels * 500 Hz * 60 s
        else:
            #eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel
            print('NAN 2')
            # EEG_data = float('nan') * np.ones((2, 30000)) # 2 channels * 500 Hz * 60 s
            EEG_data = float('nan') * np.ones((6, 30000)) # 6 channels * 500 Hz * 60 s
    else:
        #eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel
        print('NAN 3')
        # EEG_data = float('nan') * np.ones((2, 30000)) # 2 channels * 500 Hz * 60 s
        EEG_data = float('nan') * np.ones((6, 30000)) # 6 channels * 500 Hz * 60 s

    # Extract ECG recordings.
    ECG_data = list()
    DATA = list()
    ECG_sampling_frequency = 500
    ecg_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    group = 'ECG'

    if num_recordings > 0:
        # recording_id = recording_ids[0]
        recording_id = reduced_recording_ids[0]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            data, channels = reduce_channels(data, channels, ecg_channels)
            DATA, ECG_sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
            ECG_data = ECG_expand_channels(DATA, channels, ecg_channels)
            #features = get_ecg_features(data)
            #ecg_features = expand_channels(features, channels, ecg_channels).flatten()

        else:
            #ecg_features = float('nan') * np.ones(10) # 5 channels * 2 features / channel
            #num_channels, num_samples = np.shape(data)
            ECG_data = float('nan') * np.ones((5, 30000)) # 5 channels * 500 Hz * 60 s
    else:
        #ecg_features = float('nan') * np.ones(10) # 5 channels * 2 features / channel
        ECG_data = float('nan') * np.ones((5, 30000)) # 5 channels * 500 Hz * 60 s

    # Extract features.
    #return np.hstack((patient_features, eeg_features, ecg_features))
    return EEG_data.T, ECG_data.T, EEG_sampling_frequency, ECG_sampling_frequency

def ECG_expand_channels(current_data, current_channels, requested_channels):
    if current_channels == requested_channels:
        expanded_data = current_data
    else:
        num_current_channels, num_samples = np.shape(current_data)
        num_requested_channels = len(requested_channels)
        expanded_data = np.zeros((num_requested_channels, num_samples))
        for i, channel in enumerate(requested_channels):
            if channel in current_channels:
                j = current_channels.index(channel)
                expanded_data[i, :] = current_data[j, :]
            else:
                expanded_data[i, :] = float('nan') * np.ones((1, num_samples))
    return expanded_data

# short-time Fourier transform
def STFT1(X,flag,figure_folder,j,i,y1,y2,sampling_frequency,A):
        
    f,t,Zxx = signal.stft(X,sampling_frequency)
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
    #     filename = os.path.join(figure_folder, 'STFT figures', A+'_'+str(y1)+'_'+str(y2)+'_'+str(j)+'_'+str(i)+'.png')
    #     plt.savefig(filename)
    #     plt.close()
    # else:
    #     pass
    
    return Zxx

#reshape data and labels
def data_reshape(model_folder,EEG_recordings, ECG_recordings, outcomes, cpcs, EEG_sampling_frequency, ECG_sampling_frequency):

    frames = list()

    for j in range(len(EEG_recordings)):
        STFT = list()

        EEG_x_train = EEG_recordings[j]
        img = list()
        # for m in range(2):
        for m in range(6):
            img = STFT1(EEG_x_train[:,m],1,model_folder,j,m,outcomes[j],cpcs[j],EEG_sampling_frequency,'EEG')
            # dsize = output_width, output_height
            STFT.append(cv2.resize(img, (128, 128), interpolation = cv2.INTER_LINEAR)) # interpolation = cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4

        # ECG_x_train = ECG_recordings[j]
        # img = list()
        # for m in range(5):
        #     img = STFT1(ECG_x_train[:,m],1,model_folder,j,m,outcomes[j],cpcs[j],ECG_sampling_frequency,'ECG')
        #     # dsize = output_width, output_height
        #     STFT.append(cv2.resize(img, (128, 128), interpolation = cv2.INTER_LINEAR)) # interpolation = cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4    

        # frames.append(np.dstack((STFT[0], STFT[1], STFT[2], STFT[3], STFT[4], STFT[5], STFT[6])))
        # frames.append(np.dstack((STFT[0], STFT[1])))
        frames.append(np.dstack((STFT[0], STFT[1], STFT[2], STFT[3], STFT[4], STFT[5])))

    # bring the segment into a better shape
    # X_all = np.asarray(frames).reshape(-1, np.shape(STFT[0])[0], np.shape(STFT[0])[1],7)
    #X_all = np.asarray(frames).reshape(-1, np.shape(STFT[0])[0], np.shape(STFT[0])[1],2)
    X_all = np.asarray(frames).reshape(-1, np.shape(STFT[0])[0], np.shape(STFT[0])[1],6)

    if outcomes[0] == 'Nan':
        return X_all
    else:
        y1_all = np.asarray(outcomes)
        y2_all = np.asarray(cpcs)
        return X_all, y1_all, y2_all

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
    model.add(Conv2D(16, (3,3), padding = 'same', activation='linear', kernel_initializer='lecun_uniform', input_shape=(np.shape(X_all)[1],np.shape(X_all)[2],np.shape(X_all)[3])))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))

    model.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer='lecun_uniform', activation='linear'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))

    model.add(Conv2D(64, (3,3), padding = 'same', kernel_initializer='lecun_uniform', activation='linear'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    #model.add(Dropout(0.1))

    model.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer='lecun_uniform', activation='linear'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))
    #model.add(Dropout(0.25))

    model.add(Conv2D(256, (3,3), padding = 'same', kernel_initializer='lecun_uniform', activation='linear'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))

    model.add(Conv2D(512, (3,3), padding = 'same', kernel_initializer='lecun_uniform', activation='linear'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), padding = 'same'))

    model.add(Flatten())

    model.add(Dense(1024, activation='linear', kernel_initializer='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))

    model.add(Dense(1024, activation='linear', kernel_initializer='lecun_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(k, activation='softmax', kernel_initializer='lecun_uniform'))
    model.summary()

    #Model compiler settings
    if k ==2:
        model.compile(optimizer = tf.keras.optimizers.Adam(0.001),#tf.keras.optimizers.legacy.SGD(learning_rate=0.01),#tf.keras.optimizers.Adam(0.0005),
              loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5,apply_class_balancing=False),#tf.keras.losses.BinaryCrossentropy(),#'binary_crossentropy',#tf.keras.losses.SparseCategoricalCrossentropy(),#custom_loss, #tfr.keras.losses.ApproxNDCGLoss(), #'categorical_crossentropy',
              metrics=[tf.keras.metrics.AUC()])  # custom_fpr,tf.keras.metrics.Recall()  #['accuracy'])
    else:
        model.compile(optimizer = tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error', #'categorical_crossentropy',
              metrics=['mse']) #['accuracy'])        
    
    return model

def fit_and_eval(flag,X,y,model,epochs,batch_size,early_stopping,model_checkpoint):
    
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
            plt.plot(model_history[i].history['auc'], label = 'Training AUC')
            plt.plot(model_history[i].history['val_auc'], label = 'Validation AUC', linestyle = 'dashdot')  
            # plt.plot(model_history[i].history['recall'], label = 'Training Recall')
            # plt.plot(model_history[i].history['val_recall'], label = 'Validation Recall', linestyle = 'dashdot')      
            fig_name = os.path.join(model_folder, 'Trial ' + str(i) + ' Training metrics v.s. Validation metrics for ' + A + ' model with nested 10-fold cv.png')
        else:
            fig_name = os.path.join(model_folder, 'Trial ' + str(i) + ' Training Loss v.s. Validation Loss for ' + A + ' model with nested 10-fold cv.png')   
        plt.legend()
        #fig_name = os.path.join(model_folder, 'Trial ' + str(T), 'Training Loss v.s. Validation Loss for ' + A + ' model with nested holdout.png')
        plt.savefig(fig_name, dpi=200, bbox_inches='tight')
        plt.close()

# Extract patient features from the data.
# def get_patient_features(data):
#     age = get_age(data)
#     sex = get_sex(data)
#     rosc = get_rosc(data)
#     ohca = get_ohca(data)
#     shockable_rhythm = get_shockable_rhythm(data)
#     ttm = get_ttm(data)

#     sex_features = np.zeros(2, dtype=int)
#     if sex == 'Female':
#         female = 1
#         male   = 0
#         other  = 0
#     elif sex == 'Male':
#         female = 0
#         male   = 1
#         other  = 0
#     else:
#         female = 0
#         male   = 0
#         other  = 1

#     features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

#     return features

# # Extract features from the EEG data.
# def get_eeg_features(data, sampling_frequency):
#     num_channels, num_samples = np.shape(data)

#     if num_samples > 0:
#         delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
#         theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
#         alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
#         beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

#         delta_psd_mean = np.nanmean(delta_psd, axis=1)
#         theta_psd_mean = np.nanmean(theta_psd, axis=1)
#         alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
#         beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
#     else:
#         delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

#     features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T

#     return features

# # Extract features from the ECG data.
# def get_ecg_features(data):
#     num_channels, num_samples = np.shape(data)

#     if num_samples > 0:
#         mean = np.mean(data, axis=1)
#         std  = np.std(data, axis=1)
#     elif num_samples == 1:
#         mean = np.mean(data, axis=1)
#         std  = float('nan') * np.ones(num_channels)
#     else:
#         mean = float('nan') * np.ones(num_channels)
#         std = float('nan') * np.ones(num_channels)

#     features = np.array((mean, std)).T

#     return features
