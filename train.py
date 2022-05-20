import torch
import numpy as np
from tqdm import tqdm
from itertools import chain
from torch.utils.data import DataLoader

from Utils import Calculate_AP
from dataset import data_set
from SPP_model import SuperPointNet
from SPP_head import SPP_head, Focal_Loss

class MainProcess:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        self._init_model()
        self._data_loader()

    def _init_model(self):
        self.backbone = SuperPointNet()
        # self.backbone.load_state_dict(torch.load('weight.pth', map_location=self.device))
        self.backbone = self.backbone.to(self.device)
        print('SPP model init ok')
        self.head = SPP_head(0.8)
        self.head = self.head.to(self.device)

        self.optimizer = torch.optim.Adam(params=chain(self.head.parameters(), self.backbone.parameters()), weight_decay=3e-5, lr=5e-4)
        self.loss = Focal_Loss()
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [2, 4, 6, 8, 10, 12], 0.8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [10, 15, 20, 25, 30], 0.8)

    def _data_loader(self):
        train_set, val_set = data_set(self.data_folder)
        self.train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=8)
        self.val_loader = DataLoader(dataset=val_set, batch_size=4, shuffle=True, drop_last=True)

        print("train data: {}  |  val data: {}".format(train_set.sample, val_set.sample))
        print("train iter: {}  |  val iter: {}".format(len(self.train_loader), len(self.val_loader)))

    def _label_transform_back(self, tgt):
        label_np = tgt.detach().cpu().numpy()
        labels = []
        scores = label_np[:-2]
        xys = label_np[-2:]
        _, I, J = np.where(scores > 0.999)
        S = scores[_, I, J]
        for c, i, j, s in zip(_, I, J, S):
            offi, offj = xys[..., i, j]
            i, j = i + offi, j + offj
            labels.append(8 * np.array([i, j]))
        return labels

    def train(self, epoch):
        # self.backbone.eval()
        self.backbone.train()
        self.head.train()
        calculate_ap_train = Calculate_AP('train', 25, self._label_transform_back, 1000)
        pbar = tqdm(self.train_loader)
        precision = []
        epoch_loss = 0
        for i, data in enumerate(pbar):
            sample, target = data
            sample = sample.to(self.device)
            target = target.to(self.device)
            heat_map = self.backbone(sample)
            pred = self.head(heat_map[0])

            loss, losses = self.loss(pred, target)
            epoch_loss += loss.item()

            pred_points = self.head._post_processing(pred)
            calculate_ap_train.add_buffer(pred_points, target)
            for k, v in calculate_ap_train.map_eval.items():
                precision.append(v.average_precision)

            loss.backward()
            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.step()
            pbar.set_description("train -- epoch:{} | loss:{} | AP:{} | lr:{}".
                                 format(epoch, epoch_loss / (i + 1), sum(precision) / len(precision) * 100, lr))
        self.scheduler.step()

    def test(self, epoch):
        with torch.no_grad():
            self.backbone.eval()
            self.head.eval()
            calculate_ap_test = Calculate_AP('test', 50, self._label_transform_back, 1000)
            pbar = tqdm(self.val_loader)
            precision = []
            epoch_loss = 0
            for i, data in enumerate(pbar):
                sample, target = data
                sample = sample.to(self.device)
                target = target.to(self.device)
                heat_map = self.backbone(sample)
                pred = self.head(heat_map[0])
                loss, losses = self.loss(pred, target)
                epoch_loss += loss.item()

                pred_points = self.head._post_processing(pred)
                calculate_ap_test.add_buffer(pred_points, target)
                for k, v in calculate_ap_test.map_eval.items():
                    precision.append(v.average_precision)

                pbar.set_description("val -- epoch:{} | loss:{} | AP:{}".format(epoch, epoch_loss / (i + 1),
                                                                         sum(precision) / len(precision) * 100))

            ap = sum(precision) / len(precision) * 100
            torch.save(self.backbone.state_dict(), "{}/epoch{}-ap{:.3f}-main.pth".format('models', epoch, ap))
            torch.save(self.head.state_dict(), "{}/epoch{}-ap{:.3f}.pth".format('models', epoch, ap))

    def run(self):
        epoch = 0
        while epoch < 22:
            self.train(epoch)
            self.test(epoch)
            epoch += 1

if __name__ == "__main__":
    folder_path = 'Data_folder'
    Process = MainProcess(folder_path)
    Process.run()