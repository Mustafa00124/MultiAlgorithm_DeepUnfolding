"""
Old school numpy loader
"""
import numpy as np
import matplotlib.pyplot as plt

class Loader:
    def __init__(self, path, time_steps, load_only=-1, flatten=True, scale=False):
        '''
        :param path: file path, data format: [time step, index, dimensions of one sample]
        :param load_only: load a limited number of samples, -1 if load all.
        :param flatten: Flatten all frames (images) to one-directional arrays
        :param scale: Scale 8-bit images of range 0-255 to range 0-1
        '''
        self.data = np.load(path).astype('float32')
        print('original data shape {}'.format(self.data.shape[2:]))
        assert load_only != 0, 'load_only should be either -1 (load all) or a positive number'
        assert load_only >= -1, 'load_only should be either -1 (load all) or a positive number'
        assert time_steps <= self.data.shape[0], 'time_steps should be smaller than the number of frames'
        if load_only > 0:
            self.data = self.data[:, :load_only]
        if time_steps < self.data.shape[0]:
            self.data = self.data[:time_steps]
        self.num_frames, self.num_samples, self.size = self.data.shape[0], self.data.shape[1], self.data.shape[2:]

        if flatten:
            self.data = self.data.reshape([self.num_frames, self.num_samples, -1])

        if scale:
            self.data = self.data / 255.

        self.train_cutoff = int(self.num_samples * 0.8)
        self.validation_cutoff = self.train_cutoff + int(self.num_samples * 0.1)
        self.train = self.data[:, :self.train_cutoff, ...]
        self.validation = self.data[:, self.train_cutoff: self.validation_cutoff, ...]
        self.test = self.data[:, self.validation_cutoff:, ...]
        self.current_idx_train = 0
        self.current_idx_validation = 0
        self.current_idx_test = 0

        print('data loaded, training/validation/testing: {}/{}/{}'.format(self.train.shape[1], self.validation.shape[1],
                                                                    self.test.shape[1]))

    def shuffle(self):
        '''
        Like np.random.shuffle but along the second axis
        '''
        indices = np.random.permutation(self.train_cutoff)
        self.train = self.train[:, indices, ...]

    def load_batch_train(self, batch_size):
        if self.current_idx_train + batch_size > self.train_cutoff:
            self.shuffle()
            self.current_idx_train = 0

        batch = self.train[:, self.current_idx_train:self.current_idx_train + batch_size, ...]
        self.current_idx_train += batch_size
        return batch

    def load_batch_validation(self, batch_size):
        if self.current_idx_validation + batch_size > self.validation.shape[1]:
            self.current_idx_validation = 0
            return []
        batch = self.validation[:, self.current_idx_validation: self.current_idx_validation + batch_size, ...]
        self.current_idx_validation += batch_size
        return batch

    def load_batch_test(self, batch_size):
        if self.current_idx_test + batch_size > self.test.shape[1]:
            self.current_idx_test = 0
            return []
        batch = self.test[:, self.current_idx_test: self.current_idx_test + batch_size, ...]
        self.current_idx_test += batch_size
        return batch


class Moving_MNIST_Loader(Loader):
    def __init__(self, path, time_steps=20, load_only=-1, flatten=True, scale=False):
        '''
        :param path: moving mnist file path
        '''
        super(Moving_MNIST_Loader, self).__init__(path, time_steps, load_only, flatten, scale)
        self.time_steps = time_steps

    def visualize(self, start=0, end=1):
        for i in range(start, end):
            clip = self.data[:, i, :, :]
            clip = 255 - clip
            plt.figure(1)
            plt.clf()
            plt.title('our method')
            for j in range(self.time_steps):
                img = clip[j]
                plt.imshow(img, cmap='gray')
                plt.pause(0.05)
                plt.draw()

class Moving_MNIST_RPCA_Loader():
    def __init__(self, path_data, path_fg, path_bg, time_steps=20, load_only=-1, flatten=True, scale=False):
        self.data = Moving_MNIST_Loader(path_data, time_steps, load_only, flatten, scale)
        self.foreground = Moving_MNIST_Loader(path_fg, time_steps, load_only, flatten, scale)
        self.background = Moving_MNIST_Loader(path_bg, time_steps, load_only, flatten, scale)
        self.train_samples = self.data.train.shape[1]
        self.eval_samples = self.data.validation.shape[1]
        self.test_samples = self.data.test.shape[1]

    def load_batch_train(self, batch_size):
        '''rewrite due to shuffling, need to make sure after shuffling all indices of foreground, background and full are equal '''
        if self.data.current_idx_train + batch_size > self.data.train_cutoff:
            indices = np.random.permutation(self.data.train_cutoff)
            self.data.train = self.data.train[:, indices, ...]
            self.foreground.train = self.foreground.train[:, indices, ...]
            self.background.train = self.background.train[:, indices, ...]
            self.data.current_idx_train = 0
            self.foreground.current_idx_train = 0
            self.background.current_idx_train = 0

        batch_data = self.data.train[:, self.data.current_idx_train:self.data.current_idx_train + batch_size, ...]
        batch_foreground = self.foreground.train[:, self.foreground.current_idx_train:self.foreground.current_idx_train + batch_size, ...]
        batch_background = self.background.train[:, self.background.current_idx_train:self.background.current_idx_train + batch_size, ...]
        self.data.current_idx_train += batch_size
        self.foreground.current_idx_train += batch_size
        self.background.current_idx_train += batch_size
        return batch_data, batch_foreground, batch_background

    def shuffle(self):
        shuffle_seed = 2021
        np.random.seed(shuffle_seed)
        self.data.shuffle()
        np.random.seed(shuffle_seed)
        self.foreground.shuffle()
        np.random.seed(shuffle_seed)
        self.background.shuffle()

    def load_batch_validation(self, batch_size):
        return self.data.load_batch_validation(batch_size), \
               self.foreground.load_batch_validation(batch_size), \
               self.background.load_batch_validation(batch_size)

    def load_batch_test(self, batch_size):
        return self.data.load_batch_test(batch_size), \
               self.foreground.load_batch_test(batch_size), \
               self.background.load_batch_test(batch_size)

class Caltech_Loader(Loader):
    def __init__(self, path, time_steps=128, load_only=-1, flatten=False, scale=False):
        '''
        :param path: Caltech256 file path
        '''
        super(Caltech_Loader, self).__init__(path, time_steps, load_only, flatten, scale)

    def visualize(self, start=0, end=100):
        for i in range(start, end):
            img = self.train[:, i, :]
            plt.figure(1)
            plt.clf()
            plt.title('img {}'.format(i))
            plt.imshow(img, cmap='gray')
            plt.pause(.01)
            plt.draw()

if __name__ == '__main__':

    path = 'moving_mnist/seq_r15.npy'
    loader2 = Moving_MNIST_Loader(path, time_steps=20, flatten=False, scale=False)
    loader2.visualize(29, 30)
