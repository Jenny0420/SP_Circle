import os
import torch
import pickle
import numpy as np
import albumentations as A
from pprint import pprint
from torch.utils.data import Dataset
from random import random,choice, shuffle

def data_set(folder):
    train_set = TemplateSet(folder, 'train', True)
    test_set = TemplateSet(folder, 'test', False)

    return train_set, test_set

class TemplateSet(Dataset):
    def __init__(self, folder_path, mode, augment):
        self.folder_path = folder_path
        self.img_path = os.path.join(self.folder_path, 'img')
        self.label_path = os.path.join(self.folder_path, 'label')
        self.aug = augment
        self.mode = mode
        self.stride = 8
        self.current_buffer = []
        self._nums_check()
        self._generate_kernel(5, 2)
        self._set_aug()

    def _nums_check(self):
        self.files = [int(a[:-4]) for a in os.listdir(self.img_path) if a.endswith('.npy')]
        self.files.sort()
        if self.mode == 'train':
            self.files = self.files[: int(len(self.files) * 0.8)]
        else:
            self.files = self.files[int(len(self.files) * 0.8) :]

        self.sample  = len(self.files) * 30

        print('{} data nums: {}'.format(self.mode, self.sample))

    def _set_aug(self):
        if self.aug:
            self._trans = []
            self._trans.append(A.MotionBlur(p=0.5))
            self._trans.append(A.Flip(p=0.5))
            self._trans.append(A.ShiftScaleRotate(p=0.5))
            self._trans.append(A.Resize(480, 640))
        else:
            self._trans = []
            self._trans.append(A.Resize(480, 640))

        self.trans = A.Compose(self._trans, keypoint_params={'format':'xy'})
        pprint(self._trans)

    def _generate_kernel(self, kernel_size, std):
        self.kernel_size = kernel_size = int(kernel_size // 2 * 2 + 1)
        self.half_kernel = half_kernel = kernel_size // 2
        self.kernel = np.zeros([kernel_size, kernel_size])
        for i in range(-half_kernel, half_kernel + 1):
            for j in range(-half_kernel, half_kernel + 1):
                z = (i ** 2 + j ** 2) / std ** 2
                self.kernel[i + half_kernel, j + half_kernel] = np.exp(-z)

    def _load_data(self):
        tgt = choice(self.files)
        data = np.load('{}/{}.npy'.format(self.img_path, tgt), allow_pickle=True)
        with open('{}/{}.pt'.format(self.label_path, tgt), 'rb') as fp:
            label = pickle.load(fp)
        self.current_buffer = [data, label]

    def apply(self, img, label):
        augmented = self.trans(image=img, keypoints=label)
        img_ = augmented['image']
        label_ = augmented['keypoints']

        tensor_img = np.array(img_, dtype=np.float32) / 255.
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0)
        Labels = []
        for x, y in label_:
            Labels.append((int(x), int(y)))

        return tensor_img, label_

    def _appy_kernel(self, labels_np, i, j):
        ch, cw = labels_np.shape[-2:]
        kernel = self.kernel
        i1, i2 = i - self.half_kernel, i + self.half_kernel + 1
        j1, j2 = j - self.half_kernel, j + self.half_kernel + 1
        if i1 < 0:
            kernel = kernel[-i1:]
            i1 = 0
        if j1 < 0:
            kernel = kernel[:, -j1:]
            j1 = 0
        if i2 > ch:
            kernel = kernel[:ch - i2]
            i2 = ch
        if j2 > cw:
            kernel = kernel[:, :cw - j2]
            j2 = cw

        labels_np[0, i1:i2, j1:j2] = np.maximum(kernel, labels_np[0, i1:i2, j1:j2])
        return labels_np

    def _label_transform(self, labels, hw):
        oh, ow = np.array(hw) // self.stride
        labels_np = np.zeros([3, oh, ow], dtype=np.float32)

        for label in labels:
            try:
                j = (label[0] // self.stride).astype(int)
                i = (label[1] // self.stride).astype(int)
                offj = label[0] % self.stride / self.stride
                offi = label[1] % self.stride / self.stride

                offi = (offi - 0.5) * 0.8 + 0.5
                offj = (offj - 0.5) * 0.8 + 0.5
                labels_np[-2:, i, j] = offi, offj
                labels_np = self._appy_kernel(labels_np, i ,j)
            except:
                pass
        return torch.from_numpy(labels_np)

    def _label_transform_back(self, label_np):
        labels = []
        scores = label_np[:-2]
        xys = label_np[-2:]
        _, I, J = np.where(scores > 0.999)
        S = scores[_, I, J]
        for c, i, j, s in zip(_, I, J, S):
            offi, offj = xys[..., i, j]
            i, j = i + offi, j + offj
            labels.append(self.stride * np.array([i, j]))
        return labels

    def __len__(self):
        return self.sample

    def __getitem__(self, item):
        if not len(self.current_buffer):
            self._load_data()
        else:
            if not len(self.current_buffer[0]):
                self._load_data()

        current_data = self.current_buffer[0][0]
        current_label = self.current_buffer[1][0]
        self.current_buffer[0] = np.delete(self.current_buffer[0], 0, 0)
        self.current_buffer[1].pop(0)
        tensor_data, labels = self.apply(current_data, current_label)
        tensor_label = self._label_transform(labels, (480, 640))

        return tensor_data, tensor_label


# if __name__ == "__main__":
#     folder = "/mnt/data/soft/pycharm/project/SPP_circle/Data_folder"
#
#     train_set, test_set = data_set(folder)
#     train_set.__getitem__(1)

