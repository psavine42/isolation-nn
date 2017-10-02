import numpy as np
import torch
from torch.utils import data
import glob
import os


class CoupledLoader(data.Dataset):
    def __init__(self, root, ext=".npz", pos_in_file=64, base_root='', vl=False, train=True):
        self.root = root
        self.ext = ext
        self.files = {}
        self.base_root = base_root
        self.base_files = {}
        self.is_value_training = vl
        files_root = glob.glob(self.root + '*' + self.ext)
        percent_train = 0.98
        #theyre in order, so this is how i role
        total = int(len(files_root) * percent_train)
        if train:
            self.files = files_root[:total]
        else:
            self.files = files_root[total:]

        #adhoc load base games for wins cuz i forgot
        if self.base_root:
            #total_base = int(len(files_root_base) * percent_train)
            files_root_base = glob.glob(self.base_root + '*' + self.ext)
            self.base_files = [os.path.splitext(os.path.basename(fl))[0] for fl in files_root_base]


        self.player = 0
        self.out_of_this_file = 0
        self.current_file = None

        if not self.files:
            raise Exception("No files in" +  self.root)

        print("positions: %d " % (len(self.files)))

    def __len__(self):
        """__len__"""
        return len(self.files)

    def load_npz(self, fname, kys):
        npz = np.load(fname)
        ret = dict((name, npz[name].copy()) for name in kys)
        npz.close()
        return ret

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        ret = self.load_npz(self.files[index], ['position', 'move'])
        position = torch.from_numpy(ret['position']).float()

        if not self.is_value_training:
            #so i did not save the frickin win index in my position files
            #cuz i hadnt read that far in alphago paper
            position = torch.from_numpy(ret['position']).float()
            moves = torch.from_numpy(ret['move']).long()
            return position, moves
        else:
            # to retrieve it, I have to find the origional game..
            name = os.path.splitext(os.path.basename(self.files[index]))[0] 
            nm = name.split('m')[0]
            files_root = glob.glob(self.base_root + nm + '*.npz' )[0]
            if not files_root:
                return self.__getitem__(index + 1)
            else:
                game_hist = self.load_npz(files_root, ['hist'])
                hist = game_hist['hist']
                move = np.where(hist == ret['move'])
                

            ##########KD: nakilo;rhHHHHHHHHHhkjhneaihjgn aregoirjg 
            return position, []

    def set_new_file(self):
        self.player = 0
        self.out_of_this_file = 0
        self.current_file = None