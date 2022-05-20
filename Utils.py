import os
import random
import numpy as np
from collections import deque

class MapEvaluator:
    def __init__(self, class_name=None, length=1000):
        # self._class_name = class_name
        self._buffer = deque([], length)

        ## counts
        self._total_num_detections = 0
        self._total_num_points = 0

    def add_detection(self, score, real):
        self._total_num_detections += 1
        self._total_num_points += real
        self._buffer.append([score, real])

    def add_real_but_not_detected(self, num_not_predicted):
        self._total_num_points += num_not_predicted
        for _ in range(num_not_predicted):
            self._buffer.append([0, 1])

    def shuffle(self):
        random.shuffle(self._buffer)

    def clear(self):
        self._buffer.clear()

    def _cum_prec_recs(self):
        if len(self._buffer) < 4: return [], [], []
        score_realls = np.array(self._buffer)
        idx = score_realls[:, 0].argsort()[::-1]
        score_realls = score_realls[idx]

        ## calculate recall and precision
        scores = score_realls[:, 0]
        recalls = score_realls[:, 1].cumsum() / (score_realls[:, 1].sum() + 1e-8)
        precisions = score_realls[:, 1].cumsum() / np.arange(1, 1 + len(score_realls))
        precisions[score_realls[:, 0] == 0] = 0
        fmax = precisions[-1]
        for i in range(len(precisions) - 1, -1, -1): fmax = precisions[i] = max(precisions[i], fmax)
        return scores, recalls, precisions

    @property
    def average_precision(self):
        scores, recalls, precisions = self._cum_prec_recs()
        ## calculate ap
        try:
            areas_piecewise = np.diff(recalls, prepend=0) * precisions * (scores > 0)
            return areas_piecewise.sum()
        except:
            return 0


    @property
    def best_score_by_f(self):
        scores, recalls, precisions = self._cum_prec_recs()
        f = 2 * recalls * precisions / (precisions + recalls + 1e-8)
        i = f.argmax()
        return scores[i], recalls[i], precisions[i]

    # def plot_map(self, ax):
    #     scores, recalls, precisions = self._cum_prec_recs()
    #     ax.plot(recalls, precisions, label=self._class_name)
    #
    # def plot_f_score(self, ax):
    #     scores, recalls, precisions = self._cum_prec_recs()
    #     f = 2 * recalls * precisions / (precisions + recalls + 1e-8)
    #     ax.plot(scores, f, label=self._class_name)

class DistEvaluator:
    def __init__(self, class_name=None, length=100):
        # self._class_name = class_name
        self._buffer = deque([], length)

    def add(self, score, distance):
        self._buffer.append(distance)

    def clear(self):
        self._buffer.clear()

    def average_distance(self):
        return np.mean(self._buffer)

def create_point_dist_mat(points, labels):
    m, n = len(labels), len(points)
    labels_yx = np.array([label for label in labels]).astype(np.float32)
    points_yx = np.array([point[:2] for point in points]).astype(np.float32)

    dist_mat = np.zeros([m, n])
    for i in range(m):
        dist_mat[i] = np.linalg.norm(points_yx - labels_yx[i], axis=1)
    return dist_mat

def col_first_closest_match(dist_mat):
    n = dist_mat.shape[1]
    for j in range(n):
        dists = dist_mat[:, j]
        i = dists.argmin()
        d = dists[i]
        if d == np.inf: continue
        dist_mat[i] = np.inf
        dist_mat[:, j] = np.inf
        dist_mat[i, j] = d
    return dist_mat

class Calculate_AP:
    def __init__(self, topic, radius, fn, length):
        self._topic = topic
        self.label_back = fn
        self._radius = radius
        self.map_eval = {}
        self.dist_eval = {}
        self.map_eval[0] = MapEvaluator("%s" % (topic), length)
        self.dist_eval[0] = DistEvaluator("%s" % (topic), length)

    def clear(self):
        for v in self.map_eval.values(): v.clear()
        for v in self.dist_eval.values(): v.clear()

    def add_buffer(self, Preds, Labels, draw=False, dir=None):
        Labels = [self.label_back(label) for label in Labels]
        # if draw:
        #     img = np.zeros([240, 320, 3], np.uint8)
        #     for i in range(len(Labels)):
        #         img_truth = deepcopy(img)
        #         for each in Labels[i]:
        #             points_yx = each['yx']
        #             img_truth = cv2.circle(img_truth, (int(points_yx[1]), int(points_yx[0])), 5, (0, 0, 255), 5)
        #         for each in Preds[i]:
        #             points_yx = each.yx
        #             img_truth = cv2.circle(img_truth, (int(points_yx[1]), int(points_yx[0])), 5, (255, 0, 0), 5)
        #         img_truth = img_truth.astype(np.uint8)
        #         cv2.imwrite(filename=dir + str(i) + '_img_truth.jpg', img=img_truth)
        for points, labels in zip(Preds, Labels):
            m, n = len(labels), len(points)

            ## no real labels
            if m == 0:
                for p in points:
                    self.map_eval[0].add_detection(p[-1], 0)
                continue

            ## no predictions
            if n == 0:
                for l in labels:
                    self.map_eval[0].add_real_but_not_detected(1)
                continue

            ## create distance matrix
            dist_mat = create_point_dist_mat(points, labels)
            dist_mat[dist_mat > self._radius] = np.inf
            dist_mat = col_first_closest_match(dist_mat)

            ## append precision
            for j, p in enumerate(points):
                d = dist_mat[:, j].min()
                if d != np.inf:
                    self.dist_eval[0].add(p[-1], d)
                    self.map_eval[0].add_detection(p[-1], 1)
                else:
                    self.map_eval[0].add_detection(p[-1], 0)

            ## append left labels
            for i, l in enumerate(labels):
                if dist_mat[i].min() != np.inf: continue
                self.map_eval[0].add_real_but_not_detected(1)