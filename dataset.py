from torch.utils.data import Dataset
import os.path as osp
import torch
import h5py as h5


class abide_data(Dataset):

    def __init__(self, opt, split_indicies, measure, transform=None):
        """

        :param opt: Command line option(/defaults)
        :param split_indicies: train | val | test indices
        :param measure: One of the following: 'alff', 'T1', 'degree_centrality_binarize', 'degree_centrality_weighted',
                   'eigenvector_centrality_binarize', 'eigenvector_centrality_weighted', 'lfcd_binarize',
                   'lfcd_weighted', 'entropy', 'reho', 'vmhc', 'autocorr', 'falff',
        """
        data_file = osp.join(opt.root_path, opt.data_file)
        hfile = h5.File(data_file)

        data = hfile[f'summaries/{measure}'].value[split_indicies, :, :, :]
        labels = hfile['summaries'].attrs['DX_GROUP'][split_indicies]

        #import pdb; pdb.set_trace()

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

        self.data = torch.from_numpy(data).type(torch.FloatTensor)
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


def get_data_set(opt, split_indicies, transform=None):
    data_set = abide_data(opt, split_indicies, transform=transform)
    return data_set

