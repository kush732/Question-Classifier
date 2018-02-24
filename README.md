# Question-Classifier
Identify Question Type: Given a question, the aim is to identify the category it belongs to. The four categories to handle for this assignment are : Who, What, When, Affirmation(yes/no).
Label any sentence that does not fall in any of the above four as "Unknown" type.

**Examples:**
1. What is your name? Type: What
2. When is the show happening? Type: When
3. Is there a cab available for airport? Type: Affirmation
There are ambiguous cases to handle as well like:
What time does the train leave(this looks like a what question but is actually a When type)

**Steps to run:**

1. `Creating train and test set` - Executing _create_datasets.py_ will read a _data.tsv_ file and create train and test sets with StratifiedShuffleSplit to ensure class ratio is maintained in the train and test sets.
2. `Training and testing the classifier` - Executing _main.py_ file will train a model based on the training set (reading the train file from under resources/train_set.csv) and save the vectorizer and model inside models folder. Followed by this a prediction will be triggered for the test set (reading the test file from under resources/test_set.csv). Finally the prediction results are stored in the csv file (resources/test_set_pred.csv).


**Results**<br />
(achieved on the dataset provided in this repo)

Class distribution of the complete dataset

what : 487<br />
when : 77<br />
affirmation : 83<br />
who : 321<br />
unknown : 218<br />

Training file size - 1186 samples.<br />
Testing file size - 297 samples.

Train on 948 samples, validate on 238 samples.

Epoch 00000: val_loss improved from inf to 1.09322, saving model to resources//models/checkpoints/lstm-00-1519470669212.hdf5
406s - loss: 1.3967 - acc: 0.4937 - val_loss: 1.0932 - val_acc: 0.6303<br /><br />
Epoch 2/10
Epoch 00001: val_loss improved from 1.09322 to 0.48674, saving model to resources//models/checkpoints/lstm-01-1519470669212.hdf5
390s - loss: 0.7791 - acc: 0.7226 - val_loss: 0.4867 - val_acc: 0.8445<br /><br />
Epoch 3/10
Epoch 00002: val_loss improved from 0.48674 to 0.35189, saving model to resources//models/checkpoints/lstm-02-1519470669212.hdf5
430s - loss: 0.4000 - acc: 0.8724 - val_loss: 0.3519 - val_acc: 0.9076<br /><br />
Epoch 4/10
Epoch 00003: val_loss did not improve
431s - loss: 0.4247 - acc: 0.8734 - val_loss: 0.5792 - val_acc: 0.8361<br /><br />
Epoch 5/10
Epoch 00004: val_loss improved from 0.35189 to 0.34211, saving model to resources//models/checkpoints/lstm-04-1519470669212.hdf5
388s - loss: 0.3798 - acc: 0.8776 - val_loss: 0.3421 - val_acc: 0.9118<br /><br />
Epoch 6/10
Epoch 00005: val_loss improved from 0.34211 to 0.27564, saving model to resources//models/checkpoints/lstm-05-1519470669212.hdf5
405s - loss: 0.2501 - acc: 0.9314 - val_loss: 0.2756 - val_acc: 0.9286<br /><br />
Epoch 7/10
Epoch 00006: val_loss improved from 0.27564 to 0.24086, saving model to resources//models/checkpoints/lstm-06-1519470669212.hdf5
414s - loss: 0.1994 - acc: 0.9430 - val_loss: 0.2409 - val_acc: 0.9454<br /><br />
Epoch 8/10
Epoch 00007: val_loss improved from 0.24086 to 0.22432, saving model to resources//models/checkpoints/lstm-07-1519470669212.hdf5
401s - loss: 0.1777 - acc: 0.9599 - val_loss: 0.2243 - val_acc: 0.9454<br /><br />
Epoch 9/10
Epoch 00008: val_loss improved from 0.22432 to 0.21152, saving model to resources//models/checkpoints/lstm-08-1519470669212.hdf5
357s - loss: 0.1620 - acc: 0.9610 - val_loss: 0.2115 - val_acc: 0.9538<br /><br />
Epoch 10/10
Epoch 00009: val_loss improved from 0.21152 to 0.20210, saving model to resources//models/checkpoints/lstm-09-1519470669212.hdf5
369s - loss: 0.1421 - acc: 0.9641 - val_loss: 0.2021 - val_acc: 0.9538<br /><br />

**Report on final model's performance on test set -**<br />
Accuracy Score: _0.9562_<br />

Classification Report: <br />
             _precision     recall  f1-score   support_<br />
<br />
_affirmation_       0.89      0.81      0.85        21<br />
    _unknown_       0.92      0.91      0.92        54<br />
       _what_       0.96      0.99      0.98       122<br />
       _when_       0.95      0.95      0.95        19<br />
        _who_       0.99      0.98      0.98        81<br />

_avg / total_       0.96      0.96      0.96       297<br />

Confusion Matrix: <br />
[[ 17   4   0   0   0]<br />
 [  2  49   2   1   0]<br />
 [  0   0 121   0   1]<br />
 [  0   0   1  18   0]<br />
 [  0   0   2   0  79]]<br />