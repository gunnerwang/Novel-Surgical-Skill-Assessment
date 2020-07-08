import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from opts import *

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

class VideoDataset(Dataset):
    def __init__(self, mode, val_type, val_id):
        super(VideoDataset, self).__init__()

        self.mode = mode # train or test
        self.annotations = dict()

        if self.mode == 'train':
            a = open(meta_file, 'r')
            for item in a.readlines():
                item = item.strip('\n').split('\t')
                self.annotations[item[0].split('_')[0].lower() + '-' + item[0].split('_')[-1]] = {}
                self.annotations[item[0].split('_')[0].lower() + '-' + item[0].split('_')[-1]]['final_score'] = int(item[3])
                self.annotations[item[0].split('_')[0].lower() + '-' + item[0].split('_')[-1]]['sp_skill'] = sp_skill_map[item[2]]
            a1 = open(os.path.join(dataset_task_dir, val_type, 'split' + val_id + '.train'), 'r')
            self.keys = [item.strip('\n') for item in a1.readlines()]
        elif self.mode == 'test':
            a = open(meta_file, 'r')
            for item in a.readlines():
                item = item.strip('\n').split('\t')
                self.annotations[item[0].split('_')[0].lower() + '-' + item[0].split('_')[-1]] = {}
                self.annotations[item[0].split('_')[0].lower() + '-' + item[0].split('_')[-1]]['final_score'] = int(item[3])
                self.annotations[item[0].split('_')[0].lower() + '-' + item[0].split('_')[-1]]['sp_skill'] = sp_skill_map[item[2]]
            a1 = open(os.path.join(dataset_task_dir, val_type, 'split' + val_id + '.test'), 'r')
            self.keys = [item.strip('\n') for item in a1.readlines()]
        else:
            print('Either train or test is supported.')

        gt_files = os.listdir(gt_path)
        for gt_file in gt_files:
            if features_full_path:
                tmp_features = np.load(features_full_path + gt_file.split('.')[0] + '.npy')
            file_ptr = open(gt_path + gt_file, 'r')
            content = file_ptr.read().split('\n')[:-1]
            if features_full_path:
                classes = np.zeros(min(np.shape(tmp_features)[1], len(content)))
            else:
                classes = np.zeros(len(content))
            for i in range(len(classes)):
                classes[i] = actions_dict[content[i]]
            if with_gesture_recognition:
                self.annotations[gt_file.split('.')[0]]['segments'] = classes[::1]
            file_ptr.close()

        impre_files = os.listdir(impre_path)
        for impre_file in impre_files:
            impre_list = []
            ges_list = []
            ges_imp_encode = [1.0 for i in range(num_gesture_types)]
            start_end_frame = [0,0]
            a = open(impre_path + impre_file, 'r')
            a1 = a.readlines()
            for _, line in enumerate(a1):
                line = line.strip('\n').strip(' ').split(' ')
                impre_list.append(float(int(line[3])))
                ges_list.append(line[2])
                if _ == 0: start_end_frame[0] = int(line[0])
                if _ == len(a1) - 1: start_end_frame[1] = int(line[1])
            for i in range(len(impre_list)):
                if impre_list[i] == 0.0:
                    ges_imp_encode[actions_dict[ges_list[i]]-1] = impre_list[i]
            self.annotations[str(impre_file).split('_')[0].lower() + '-' + str(impre_file).split('.')[0].split('_')[-1]]['start_end_frame'] = start_end_frame
            if with_gesture_recognition:
                self.annotations[str(impre_file).split('_')[0].lower() + '-' + str(impre_file).split('.')[0].split('_')[-1]]['impression'] = impre_list
            if with_skill_classification:
                self.annotations[str(impre_file).split('_')[0].lower() + '-' + str(impre_file).split('.')[0].split('_')[-1]]['gesture_impression'] = ges_imp_encode
            a.close()

    def __getitem__(self, ix):
        label_final_score = self.annotations.get(self.keys[ix]).get('final_score')
        label_sp_skill = self.annotations.get(self.keys[ix]).get('sp_skill')
        label_seg = self.annotations.get(self.keys[ix]).get('segments')
        label_impre = self.annotations.get(self.keys[ix]).get('impression')
        label_gesimp = self.annotations.get(self.keys[ix]).get('gesture_impression')

        data = {}
        data['name'] = self.keys[ix]
        data['label_final_score'] = float(label_final_score) / final_score_std
        data['label_sp_skill'] = label_sp_skill

        if with_gesture_recognition:
            batch_target_tensor = torch.ones(len(label_seg), dtype=torch.long) * (-100)
            mask = torch.zeros(num_classes, len(label_seg), dtype=torch.float)
            batch_target_tensor[:np.shape(label_seg)[0]] = torch.from_numpy(label_seg)
            mask[:, :np.shape(label_seg)[0]] = torch.ones(num_classes, np.shape(label_seg)[0])

            data['label_seg'] = batch_target_tensor
            data['label_seg_mask'] = mask

            data['label_impre'] = label_impre

        if with_skill_classification:
            data['label_gesimp'] = label_gesimp

        return data

    def __len__(self):
        return len(self.keys)
