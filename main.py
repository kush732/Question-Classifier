import pickle
import time

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn import metrics

import dataset_service
import model_factory
from configs import label_file, max_features, train_file, algorithm, test_file, resource_dir_path, vectorizer_file


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


def extract_features(train_file):
    total_counts, data, labels, raw, class_to_ind, ind_to_class = dataset_service.load_data(path=train_file)
    tokenizer = Tokenizer(num_words=50, lower=True, split=" ")
    tokenizer.fit_on_texts(data)
    save_vectorizer(tokenizer)
    dataset = tokenizer.texts_to_sequences(data)
    pad_dataset = sequence.pad_sequences(dataset, maxlen=max_features)
    return pad_dataset, labels, ind_to_class


def save_vectorizer(vectorizer):
    with open(vectorizer_file, "wb") as fout:
        pickle.dump(vectorizer, fout)


def load_vectorizer():
    with open(vectorizer_file, "rb") as fin:
        vectorizer = pickle.load(fin)
    return vectorizer


def predict(texts, ind2class, algorithm):
    tokenizer = load_vectorizer()
    text_dataset = tokenizer.texts_to_sequences(texts)
    pad_dataset = sequence.pad_sequences(text_dataset, maxlen=max_features)
    model = model_factory.get_model(len(ind2class), algorithm, mode="predict")
    predictions = model.predict(pad_dataset)
    return predictions


def test(filename, algorithm):
    dataset = pd.read_csv(filename)
    analyse_test(dataset['Message'], dataset['Class'], algorithm)


def analyse_test(X_test, Y_test, algorithm):
    ind2class = dataset_service.load_labels(label_file)
    y_pred = predict(X_test, ind2class, algorithm)
    actual_labels = np.array(Y_test)
    predicted_labels = np.array([ind2class[x] for x in categorical_probas_to_classes(y_pred)])
    mask = np.max(y_pred, axis=1) > 0.1
    selected_test_labels = actual_labels[mask]
    selected_pred = predicted_labels[mask]
    score = metrics.accuracy_score(selected_test_labels, selected_pred)
    print("Accuracy Score: " + str(score))
    print("Classification Report: ")
    print(metrics.classification_report(selected_test_labels, selected_pred))
    print("Confusion Matrix: ")
    print(metrics.confusion_matrix(selected_test_labels, selected_pred))
    df = pd.DataFrame({'Message': X_test, 'Predicted Class': selected_pred, 'True Class': selected_test_labels})
    df.to_csv(resource_dir_path + "/test_set_pred.csv", index=False)


def train(train_file, algorithm):
    X_train, y_train, ind_to_class = extract_features(train_file)
    model = model_factory.get_model(len(ind_to_class), algorithm)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(patience=3, verbose=1)
    checkpointer = ModelCheckpoint(
        filepath=resource_dir_path + "/models/checkpoints/" + algorithm + "-" + "{epoch:02d}-" + str(
            int(round(time.time() * 1000))) + ".hdf5", verbose=1, save_best_only=True)
    model.fit(X_train, np_utils.to_categorical(y_train, len(ind_to_class)), validation_split=0.2, verbose=2,
              callbacks=[checkpointer, early_stopping])


if __name__ == "__main__":
    train(train_file, algorithm)
    test(test_file, algorithm)
