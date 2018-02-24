from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from configs import resource_dir_path

df = pd.read_csv(resource_dir_path + "/data.tsv", sep='\t')

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
X = np.asarray([x.strip() for x in df['Message']])
y = np.asarray([x.strip() for x in df['Class']])
print(Counter(y))

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(Counter(y_train))
    print(Counter(y_test))
    pd.DataFrame({"Message": X_train, "Class": y_train}).to_csv(resource_dir_path + "/train_set.csv", index=False)
    pd.DataFrame({"Message": X_test, "Class": y_test}).to_csv(resource_dir_path + "/test_set.csv", index=False)
