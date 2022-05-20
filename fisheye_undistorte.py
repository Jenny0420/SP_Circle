import os
import cv2
import json
import numpy as np
from  tqdm import tqdm

class Undistorte:
    def __init__(self, path):
        self.folder_path = path
        self.img_path = os.path.join(path, 'images')
        self.json_path = os.path.join(path, 'labels')
        self.k = np.array([[468.079, 0, 1007.39],
                  [0, 467.944, 775.047],
                  [0, 0, 1]])
        self.d = np.array([-0.0167991, 0.000930856, -0.00246083, -0.00246083])
        self.start_xy = (384, 544)
        self.p = np.array([[324.20117669, 0, 478.05352078],
                           [0, 345.7148511, 384.88061861],
                           [0, 0, 1]])

    def undis_img(self, name):
        img = cv2.imread(name)
        full_img = np.zeros((1536, 2048, 3))
        full_img[384:1152, 544:1504, :] += img
        img_shape = full_img.shape[:2][::-1]
        full_img = cv2.resize(full_img, img_shape).astype(np.uint8)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.k, self.d, np.eye(3), self.p, (960, 768), cv2.CV_16SC2)
        unidstorted_img = cv2.remap(full_img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
        new_name = name.replace('images', 'images_undis')
        cv2.imwrite(new_name, unidstorted_img)

    def undis_label(self, name):
        with open(name, 'r') as fp:
            obj = json.load(fp)
        n = len(obj['shapes'])

        for j, each_shape in enumerate(obj['shapes'][::-1]):
            flag = True
            point = np.zeros((2, 1, 2))

            for i in range(len(each_shape['points'])):
                point[i][0][0] = each_shape['points'][i][0] + 544
                point[i][0][1] = each_shape['points'][i][1] + 384
            after_point = cv2.fisheye.undistortPoints(point, self.k, self.d, None, self.p).squeeze().tolist()

            for each in after_point:
                if each[0] < 200 or each[0] > 750 or each[1] < 0 or each[1] > 768:
                    obj['shapes'].pop(n - j - 1)
                    flag = False
                    break
            if flag:
                each_shape['points'] = after_point
            else:
                flag = True
        new_name = name.replace('labels', 'labels_undis')
        with open(new_name, 'w') as fp:
            json.dump(obj, fp ,indent=4)

    def run(self):
        json_list = os.listdir(self.json_path)
        for each in tqdm(json_list):
            name = os.path.join(self.json_path, each)
            self.undis_label(name)
        print('ok')
        # img_list = os.listdir(self.img_path)
        # for each in tqdm(img_list):
        #     name = os.path.join(self.img_path, each)
        #     self.undis_img(name)
        # print('ok')

    def run_one_img(self):
        name = '/home/fr1511b/桌面/0.jpeg'
        img = cv2.imread(name)
        cut_img = img[384:1152, 544:1504, :]

        full_img = np.zeros((1536, 2048, 3))
        full_img[384:1152, 544:1504, :] += cut_img
        img_shape = full_img.shape[:2][::-1]
        full_img = cv2.resize(full_img, img_shape).astype(np.uint8)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.k, self.d, np.eye(3), self.p, (960, 768), cv2.CV_16SC2)
        unidstorted_img = cv2.remap(full_img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)

        cv2.imshow('0', cut_img)
        cv2.imshow('1', unidstorted_img)
        cv2.waitKey(0)

def Remake_img():
    k = np.array([[468.079, 0, 1007.39],
                  [0, 467.944, 775.047],
                  [0, 0, 1]])
    d = np.array([-0.0167991, 0.000930856, -0.00246083, -0.00246083])
    p = np.array([[324.20117669, 0, 478.05352078],
                  [0, 345.7148511, 384.88061861],
                  [0, 0, 1]])
    folder_name = '/home/fr1511b/桌面/test_img'
    path = os.path.join(folder_name, '192.168.0.189')
    name_list = os.listdir(path)
    for each in name_list:
        name = os.path.join(path, each)
        img = cv2.imread(name)
        cut_img = img[384:1152, 544:1504, :]

        # full_img = np.zeros((1536, 2048, 3))
        # full_img[384:1152, 544:1504, :] += cut_img
        # img_shape = full_img.shape[:2][::-1]
        # full_img = cv2.resize(full_img, img_shape).astype(np.uint8)
        # map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), p, (960, 768), cv2.CV_16SC2)
        # unidstorted_img = cv2.remap(full_img, map1, map2, interpolation=cv2.INTER_LINEAR,
        #                             borderMode=cv2.BORDER_CONSTANT)

        cv2.imwrite('{}/189/dis/{}'.format(folder_name, each), cut_img)
    print('')




if __name__ == "__main__":
    Remake_img()
    path = '/home/fr1511b/桌面/undistorte_imgs/labeled_all'
    obj = Undistorte(path)
    obj.run()
