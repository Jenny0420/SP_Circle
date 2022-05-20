import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import TemplateSet
from SPP_head import SPP_head
from SPP_model import SuperPointNet

def nms(points, threshhold):
    n = len(points)
    cal_m = np.ones((n ,n))
    cal_m *= np.inf
    for i in range(n):
        for j in range(i, n):
            y0, x0, s0 = points[i][0], points[i][1], points[i][2]
            y1, x1, s1 = points[j][0], points[j][1], points[j][2]
            d = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            if d == 0.0:
                pass
            else:
                cal_m[i, j] = d
    for each in cal_m:
        np.where(each < threshhold, np.inf, each)
        print('')


class Inference:
    def __init__(self, folder):
        self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        self.folder = folder
        self._init_model()

    def _init_model(self):
        self.SPP_model = SuperPointNet()
        self.head = SPP_head(0.85)
        self.SPP_model.load_state_dict(torch.load('weight.pth', map_location=self.device))
        self.head.load_state_dict(torch.load('models/epoch2-ap98.854.pth', map_location=self.device))
        self.SPP_model = self.SPP_model.to(self.device)
        self.head = self.head.to(self.device)

    def _load_data(self):
        self.dataset = TemplateSet(self.folder, 'test', False)
        self.loader = DataLoader(dataset=self.dataset, batch_size=1)

    def run_Templateset(self):
        self._load_data()
        for data in self.loader:
            sample, target = data
            img = sample.detach().numpy().squeeze()
            img = (img * 255.).astype(np.uint8)
            label = target.detach().numpy().squeeze()
            print('')


    def run(self):
        name_list = os.listdir(self.folder)
        for each in tqdm(name_list):
            name = os.path.join(self.folder, each)
            img = cv2.imread(name, 0)
            img = cv2.resize(img, (640, 480)).astype(np.float32)
            input = torch.from_numpy(img / 255.).unsqueeze(0).unsqueeze(0)
            input = input.to(self.device)
            heat_map = self.SPP_model(input)
            pred = self.head(heat_map[0])
            pred_points = self.head._post_processing(pred)
            # pred_points = nms(pred_points[0], 50)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for point in pred_points[0]:
                cv2.circle(img, (int(point[1]), int(point[0])), 5, (0, 0, 255))
            cv2.imwrite('test/{}'.format(each), img)


if __name__ == "__main__":
    from random import seed
    seed(0)
    folder = 'img'
    obj = Inference(folder)
    obj.run()
