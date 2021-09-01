import os
import glob

from enum import Enum

import torch
import torch.utils.data
import numpy as np

from PIL import Image
from sklearn.decomposition import PCA


class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'Data_train'
        TEST = 'Data_test'
        VAL = 'Data_val'

    def __init__(self, path_to_data_dir, mode):
        super(Dataset, self).__init__()
        assert mode in Dataset.Mode
        self._mode = mode

        self._img_dir = os.path.join(path_to_data_dir, self._mode.value)
        self._img_paths = sorted(glob.glob(os.path.join(self._img_dir, '*', '*.png')), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self._classes = {'Carambula': 0, 'Lychee': 1, 'Pear': 2}
        self._data = self._pca(self._img_paths)

    def __len__(self):
        return self._data.shape[0]

    def _pca(self, img_paths):
        pca = PCA(n_components=2)
        data = []
        for img_path in img_paths:
            img = Image.open(img_path)
            img = np.asarray(img)[:, :, 0]
            img = np.true_divide(img, 255)
            img = np.reshape(img, (1, -1))
            try:
                data = np.concatenate((data, img), axis=0)
            except ValueError:
                data = img
        pca_data = pca.fit_transform(data)
        return pca_data

    def __getitem__(self, index):
        principal_component = self._data[index, :]
        label = self.parse_label(self._img_paths[index])
        sample = {'principal_component': principal_component,
                  'label': label}
        return sample

    def parse_label(self, path):
        name = path.split('/')[2]
        return self._classes[name]


if __name__ == '__main__':
    test_dataset = Dataset('data', Dataset.Mode.TRAIN)
    print(len(test_dataset))
