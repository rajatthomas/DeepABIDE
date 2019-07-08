from torch.utils.data import Dataset
import os.path as osp
import torch
import h5py as h5

import numpy as np


class abide_data(Dataset):

    def __init__(self, opt, split_indicies, measure, subset=subset, transform=None):
        """

        :param opt: Command line option(/defaults)
        :param split_indicies: train | val | test indices
        :param measure: One of the following: 'alff', 'T1', 'degree_centrality_binarize', 'degree_centrality_weighted',
                   'eigenvector_centrality_binarize', 'eigenvector_centrality_weighted', 'lfcd_binarize',
                   'lfcd_weighted', 'entropy', 'reho', 'vmhc', 'autocorr', 'falff',
        :param subset: boolean indices with which subjects to select (for example ABIDEI vs ABIDEII)
        """
        data_file = osp.join(opt.root_path, opt.data_file)
        hfile = h5.File(data_file)

        data = np.squeeze(hfile[f'summaries/{measure}'][subset, :, :, :, :])

        #import pdb; pdb.set_trace()

        if opt.standardize:
            mdata = data.mean(axis=0)
            sdata = data.std(axis=0)

            data = (data - mdata)/(sdata+1)  # 1 avoids the divide-by-zero error

        data = data[split_indicies, :, :, :]
        labels = hfile['summaries'].attrs['DX_GROUP'][split_indicies]


        # if opt.standardize:
        #
        #     mask = all_data['mask_3d']
        #
        #     n_subj = data.shape[0]
        #     for i_subj in range(n_subj):
        #         data_subj = data[i_subj]
        #         mean_subj = data_subj[mask].mean(axis=0)
        #         std_subj = data_subj[mask].std(axis=0)
        #         pdb.set_trace()
        #         if np.any(std_subj == 0) or np.any(np.isnan(mean_subj)) or np.any(np.isnan(std_subj)):
        #             import pdb;pdb.set_trace()
        #         data[i_subj] = mask * (data_subj - mean_subj) / std_subj

        self.data = torch.from_numpy(np.expand_dims(data, axis=1)).type(torch.FloatTensor)
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):

        return self.data[item], self.labels[item]


def get_data_set(opt, split_indicies, measure, subset=subset, transform=None):
    data_set = abide_data(opt, split_indicies, measure, subset=subset, transform=transform)
    return data_set

