## contains the configs used for training and other parameters

NB_TEST = 0.2

resource_dir_path = "resources/"

label_file = resource_dir_path + '/models/labels.txt'
train_file = resource_dir_path + "/train_set.csv"
test_file = resource_dir_path + "/test_set.csv"
vectorizer_file = resource_dir_path + '/models/vectorizer.pkl'

class_label = 'Class'
message_label = 'Message'
max_features = 2048

algorithm = "lstm"
