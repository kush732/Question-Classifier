import re
from random import shuffle

import numpy as np
import pandas as pd
import progressbar
from django.utils.encoding import smart_str
from sklearn.cross_validation import StratifiedShuffleSplit as Strat

from configs import NB_TEST, class_label, message_label, resource_dir_path


def get_training_data(train):
    classes = train[class_label]
    texts = train[message_label]
    return classes, texts


def load_data(threshold=20, stratified=True, path=resource_dir_path + "/train_set.csv"):
    train = pd.read_csv(path)
    classes, texts = get_training_data(train)

    data = []
    raw = []
    labels = []

    current_counts = {}

    bar = progressbar.ProgressBar(maxval=len(texts),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    total_counts = {}

    for idx, line in enumerate(texts):
        if classes[idx] == classes[idx] and len(smart_str(line)) >= 3:
            if classes[idx] in total_counts:
                total_counts[str(classes[idx])] += 1
            else:
                total_counts[str(classes[idx])] = 1
                current_counts[str(classes[idx])] = 0

    classes_set = {x for x in total_counts if total_counts[x] > threshold}

    ind_to_class = list(classes_set)
    class_to_ind = dict(zip(classes_set, range(len(classes_set))))

    bar.start()
    for idx, line in enumerate(texts):
        bar.update(idx + 1)
        if classes[idx] == classes[idx] and line == line:
            line = re.sub(r'\\u[0-9a-z]{4}', '', line)
            if len(smart_str(line)) >= 10 and total_counts[str(classes[idx])] > threshold and classes[
                idx] in classes_set:
                data.append(line)
                raw.append(smart_str(line))
                labels.append(class_to_ind[classes[idx]])
                current_counts[smart_str(classes[idx])] += 1
        else:
            if "nan" not in current_counts:
                current_counts["nan"] = 0
            else:
                current_counts["nan"] += 1
    bar.finish()

    data_shuf = []
    labels_shuf = []
    raw_shuf = []
    index_shuf = range(len(data))

    if stratified:
        split = Strat(labels, 1, test_size=NB_TEST, random_state=0)
        for train_index, test_index in split:
            index_shuf = np.append(train_index, test_index)
    else:
        shuffle(index_shuf)

    print('Shuffling Training Dataset ...')
    for i in index_shuf:
        data_shuf.append(smart_str(data[i].lower().replace('\n', ' ').replace('\r', '')))
        raw_shuf.append(raw[i])
        labels_shuf.append(labels[i])

    for key, value in total_counts.items():
        print("total clazz: " + str(key) + " : " + str(value))

    for key, value in current_counts.items():
        print("selected clazz: " + str(key) + " : " + str(value))

    print("length of selected clazz: " + str(len(classes_set)))

    from configs import label_file
    save_labels(ind_to_class, label_file)

    return total_counts, data_shuf, labels_shuf, raw_shuf, class_to_ind, ind_to_class


def save_labels(ind_to_class, label_file):
    with open(label_file, 'w') as f_label:
        for i in range(len(ind_to_class)):
            f_label.write(ind_to_class[i] + "\n")


def load_labels(label_file):
    ind_to_class = []
    with open(label_file, 'r') as f_label:
        for elem in f_label:
            ind_to_class.append(elem.strip())
    return ind_to_class
