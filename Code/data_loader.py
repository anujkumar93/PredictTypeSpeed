import numpy as np
import pickle


class DataLoader:
    def __init__(self, batch_size, file_path, debug=False):
        self.debug = debug
        self.data_file = file_path
        self.data = pickle.load(open(file_path, 'rb'), encoding='iso-8859-1')
        self.train_data = self.data['Xtr']
        self.train_labels = self.data['ytr']
        self.training_set_size = self.train_data.shape[0]
        self.train_data = self.train_data.reshape((self.training_set_size, -1)).astype(np.float32)
        self.test_data = self.data['Xte']
        self.test_labels = self.data['yte']
        self.test_set_size = self.test_data.shape[0]
        self.test_data = self.test_data.reshape((self.test_set_size, -1)).astype(np.float32)
        self.training_batch_counter = 0
        self.test_batch_counter = 0
        self.batch_size = batch_size
        sum_train = 0
        for elem in self.train_data:
            sum_train += elem
        self.mean_train = sum_train / float(self.training_set_size)
        assert self.batch_size == int(self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_training_set_size(self):
        return self.training_set_size

    def shuffle_training_set(self):
        self.training_batch_counter = 0
        if not self.debug:
            random_indxs = np.random.permutation(np.arange(self.training_set_size))
            self.train_data = self.train_data[random_indxs]
            self.train_labels = self.train_labels[random_indxs]

    def get_next_training_batch(self):
        X, y = [], []
        for i in range(self.training_batch_counter * self.batch_size,
                       min(self.training_set_size, (self.training_batch_counter + 1) * self.batch_size)):
            data = self.train_data[i]
            # data -= self.mean_train
            X.append(data)
            label = self.train_labels[i]
            y.append(label)
        self.training_batch_counter += 1
        return np.array(X), np.array(y)

    def get_test_data_file_names(self):
        return self.test_data

    def get_test_set_size(self):
        return self.test_set_size

    def reset_test_batch_counter(self):
        self.test_batch_counter = 0

    def get_next_test_batch(self):
        X, y = [], []
        for i in range(self.test_batch_counter * self.batch_size,
                       min(self.test_set_size, (self.test_batch_counter + 1) * self.batch_size)):
            data = self.test_data[i]
            # data -= self.mean_train
            X.append(data)
            label = self.test_labels[i]
            y.append(label)
        self.test_batch_counter += 1
        return np.array(X), np.array(y)
