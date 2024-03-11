import numpy as np
from copy import deepcopy

class ComputeDiffPoints():
    """
    Class to compute start, end and predict steps
    for base normal and base prediction
    and for different normal and base prediction from prediction
    """
    def __init__(self, treshold = 0):
        # all time_points - start .. between .. end
        self.treshold = treshold
        self.time_labels = np.array([0])

    def __init_data(self):
        # normal way
        self.s_point = self.time_labels[0]
        self.e_point = self.time_labels[-1]
        #remove start point
        self.points = self.time_labels[1:]
        self.next_points = self.points + 1

        # back way
        self.back_points = (self.e_point - self.time_labels)[::-1]
        self.back_s_point = self.back_points[0]
        self.back_e_point = self.back_points[-1]
        #remove start point
        self.back_points = self.back_points[1:]
        self.back_next_points = self.back_points + 1

    def getpoints_diftrain(self):
        self.__init_data()

        configs = self.__way_config(self.points, self.next_points)
        back_configs = self.__way_config(self.back_points, self.back_next_points)
        return configs, back_configs

    def getpoints_train(self):
        self.__init_data()

        configs = self.__way_config(self.points, self.points)
        back_configs = self.__way_config(self.back_points, self.back_points)
        return configs, back_configs

    def __way_config(self, points_1, points_2):
        idxs_1 = []
        idxs_2 = []
        for i, _ in enumerate(points_1):
          for j, _ in enumerate(points_2):
            if points_1[i] == points_2[j]:
              idxs_1.append(i)
              idxs_2.append(j)

        config = dict()
        config[f'id_uclip_emb'] = []
        config[f'id_img_emb_s'] = []
        config[f'id_img_delta'] = []
        config[f'delta'] = []

        if len(idxs_1) > 1:
          for id_1 in idxs_1:
            for id_2 in idxs_2:
              delta = (points_1[id_1] - points_2[id_2])
              if delta > self.treshold:
                config[f'id_uclip_emb'].append(id_2)
                # place next_points_2[id_2] in points_1
                id_img_emb_s = list(points_1).index(points_2[id_2])
                config[f'id_img_emb_s'].append(id_img_emb_s)
                config[f'id_img_delta'].append(id_1)
                config[f'delta'].append(delta)
        return config

# Create class to shuffle array
class Shuff_Reshuff(object):
    """
    Class to compute shuffle and unshufle indexess
    """
    # Constructor
    def __init__(self, d: int):

        # Initializes the temp_array
        self.idx_base = np.arange(d)

    # method to shuffle array
    def shuffle(self):
        self.idx_shuffle = deepcopy(self.idx_base)
        # Shuffle array
        np.random.shuffle(self.idx_shuffle)
        return self.idx_shuffle

    # method to unshuffle array
    def unshuffle(self):
      idx_reshuffle = np.ones(self.idx_base.shape[0])
      for i, b in enumerate(self.idx_base):
          idx = np.argwhere((self.idx_shuffle == b) == [True])
          if idx.any():
              idx_reshuffle[i] = idx[0][0]
      return idx_reshuffle.astype(int)