from __future__ import print_function

import os

from keras.layers import Dense, Dropout, Activation, GRU
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import adam
from sklearn.svm import LinearSVC

from configs import max_features, resource_dir_path


def create_LSTM(max_words, nb_classes):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_words))
    model.add(GRU(512))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model


def create_slc_model(max_words, nb_classes):
    print(nb_classes)
    print(max_words)
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model


def create_svm():
    model = LinearSVC(loss='l2', penalty='l2',
                      dual=False, tol=1e-3, C=0.3, class_weight='balanced', verbose=True)
    return model


def create_perceptron_model(max_words, nb_classes):
    model = Sequential([
        Dense(512, input_dim=max_words),
        Activation('relu'),
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax'),
    ])
    return model


def get_latest_epoch_timestamp(algorithm):
    file_names = next(os.walk(resource_dir_path + '/models/checkpoints/'))[2]
    print("filenames: " + str(file_names))
    file_names = [file_name for file_name in file_names if algorithm in file_name]
    latest_epoch = 0
    latest_timestamp = 0
    for file_name in file_names:
        _, epoch, timestamp = file_name.split('.')[0].split('-')
        epoch = int(epoch)
        timestamp = int(timestamp)
        print("epoch: " + str(epoch))
        print("timestamp: " + str(timestamp))
        if timestamp > latest_timestamp:
            latest_timestamp = timestamp
            latest_epoch = epoch
        elif timestamp == latest_timestamp and latest_epoch < epoch:
            latest_epoch = epoch

    print("latest timestamp: " + str(latest_timestamp) + " latest epoch: " + str(latest_epoch))
    return latest_epoch, latest_timestamp


def load_trained_model(model, epoch, timestamp, algorithm):
    model.load_weights(resource_dir_path +
                       '/models/checkpoints/' + algorithm + "-" + "%02d" % int(epoch) + "-" + str(timestamp) + ".hdf5")
    loss = 'categorical_crossentropy'
    optimizer = adam(lr=0.01)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def get_model(nb_classes, algorithm, mode='historic'):
    (epoch, timestamp) = get_latest_epoch_timestamp(algorithm)
    if algorithm == 'nn':
        model = create_slc_model(max_features, nb_classes)
    elif algorithm == 'perc':
        model = create_perceptron_model(max_features, nb_classes)
    elif algorithm == 'lstm':
        model = create_LSTM(max_features, nb_classes=nb_classes)
    else:
        model = None

    if mode == 'historic':
        return model
    elif (timestamp == 0 and mode == 'predict') or model is None:
        print('no model defined.. bailing out')
        exit()
    else:
        model = load_trained_model(model, epoch, timestamp, algorithm)
    return model
