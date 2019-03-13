import os
from glob import glob
from time import time
from sys import stdout
import pandas as pd
import nibabel as nib
import numpy as np
import h5py
import os.path as osp

# from scipy.special import xlogy


def subject_dict(data_dir, abide_struct_path, abide_fmri_path, abide_mask_path, pheno_file,
                 abide_metrics=['alff', 'degree_weighted', 'eigenvector_weighted','falff', 'lfcd']):
    """

    :param data_dir: path to main directory
    :param abide_fmri_path: path to abide fMRI specific directory
    :param abide_mask_path: path to abide functional mask specific directory
    :param abide_metrics: all metrics given by abide preprocessed
    :param pheno_file: file with the phenotypical variable

    :return: dictionary with subject_ids as keys and [file_path_fmri, file_path_mask, diagnosis] as value
    """
    all_fmri_files = sorted(glob(os.path.join(data_dir, abide_fmri_path, '*.nii.gz')))
    all_mask_files = sorted(glob(os.path.join(data_dir, abide_mask_path, '*.nii.gz')))
    all_struct_files = sorted(glob(os.path.join(data_dir, abide_struct_path, '*.nii')))

    all_metrics_file = []
    metric_path = 'Outputs/cpac/filt_noglobal'
    for metric in abide_metrics:
        all_metrics_file.append(sorted(glob(os.path.join(data_dir, metric_path, metric, '*.nii.gz'))))

    pheno_data = pd.read_csv(os.path.join(data_dir, pheno_file))

    # Autism = 1, Control = 2
    autistic_ids = pheno_data[pheno_data['DX_GROUP'] == 1]['SUB_ID']
    control_ids = pheno_data[pheno_data['DX_GROUP'] == 2]['SUB_ID']

    func_subject_vars = dict()
    struct_subject_vars = dict()

    for s in all_struct_files:
        sub_id_str = s.split('/')[-1].split('_')[0]
        sub_id = int(sub_id_str)
        s = os.path.join(data_dir, abide_struct_path, '{}_T1_shft_res.nii'.format(sub_id_str))

        if autistic_ids.isin([sub_id]).any():
            struct_subject_vars[sub_id] = [s, 1]

        if control_ids.isin([sub_id]).any():
            struct_subject_vars[sub_id] = [s, 0]

    for f, m, a, dw, ew, fa, l in zip(all_fmri_files, all_mask_files, *all_metrics_file):
        sub_id_str = f.split('_')[-3]
        sub_id = int(sub_id_str)

        if autistic_ids.isin([sub_id]).any():
            func_subject_vars[sub_id] = {'rsfmri': f, 'mask': m, 'alff': a, 'degree_weighted': dw,
                                         'eigenvector_weighted': ew, 'falff': fa, 'lfcd': l, 'DX': 1}
        if control_ids.isin([sub_id]).any():
            func_subject_vars[sub_id] = {'rsfmri': f, 'mask': m, 'alff': a, 'degree_weighted': dw,
                                         'eigenvector_weighted': ew, 'falff': fa, 'lfcd': l, 'DX': 0}

    return func_subject_vars, struct_subject_vars


# def get_entropy(series, nbins=15):
#     """
#     :param series: a 1-D array for which we need to find the entropy
#     :param nbins: number of bins for histogram
#     :return: entropy
#     """
#     # https://www.mathworks.com/matlabcentral/answers/27235-finding-entropy-from-a-probability-distribution
#
#     counts, bin_edges = np.histogram(series, bins=nbins)
#     p = counts / np.sum(counts, dtype=float)
#     bin_width = np.diff(bin_edges)
#     entropy = -np.sum(xlogy(p, p / bin_width))
#
#     return entropy
#
#
# def get_autocorr(series, lag_cc=0.5):
#     """
#
#     :param series: time course
#     :param lag_cc: width to be determined at lag=lag_cc
#     :return:
#     """
#     cc = np.abs(np.correlate(series, series, mode='full'))
#     cc = cc/cc.max()
#     cc = np.abs(cc-lag_cc)
#
#     return np.argsort(cc)[0] - len(cc) # because max cc is at len(cc)


def read_summary(pheno_file, metric='entropy'):
    """

    :param data_nifti: path to the 4-D resting-state or other abide summary scans.
    :return: 4D nifti-file with metric values in each voxels for all subjects dim -> [nsubjects, <vol_dims>].
    """

    pheno = pd.read_csv(pheno_file)

    nsubjects = len(pheno)

    metric_all_subjects = np.zeros((nsubjects, 45, 54, 45))
    return voxelwise_measure


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_hdf5_file(hdf5_file, pheno_file, metrics = ['structural', 'entropy','autocorr']):
    """

    :param hdf5_file: filename of the hdf5_file
    :param pheno_file: csv file with file paths and DX groups
    :param metric: [entropy, ALFF, pALFF, mean or any others
    :return: hdf5 file with the metric of interest[entropy, ALFF, pALFF, mean]
    """


    pheno = pd.read_csv(pheno_file)

    nsubjs = len(func_subject_vars)
    struct_nsubjs = len(struct_subject_vars)

    data_shape, struct_data_shape = get_data_shape(func_subject_vars, struct_subject_vars)

    sub_labels_func = []
    abide_ids_func = []
    sub_labels_struct = []
    abide_ids_struct = []

    with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'w') as hfile:

        entry = hfile.create_group(u'summaries')

        all_metrics = ['alff', 'cleaned_mni_rsfmri', 'T1', 'degree_centrality_binarize',
                       'degree_centrality_weighted', 'eigenvector_centrality_binarize',
                       'eigenvector_centrality_weighted',
                       'lfcd_binarize', 'lfcd_weighted', 'entropy', 'reho', 'vmhc', 'autocorr', 'falff', ]
        # Remember to refactor to include as many metrics with a loop

        for metric in metrics:

            dataset = entry.create_dataset(metric, shape=(nsubjs, ) + data_shape + (1,), dtype=np.float32)
            dataset[:, :, :, :] = read_summary(pheno_file, metric=metric)

        # if 'entropy' in metrics:
        #     fmri_entropy = entry.create_dataset(u'data_entropy', shape=(nsubjs, ) + data_shape + (1,), dtype=np.float32)
        #
        # if 'autocorr' in metrics:
        #     fmri_autocorr = entry.create_dataset(u'data_autocorr', shape=(nsubjs,) + data_shape + (1,),
        #                                          dtype=np.float32)
        #
        # if 'alff' in metrics:
        #     fmri_alff = entry.create_dataset(u'data_alff', shape=(nsubjs,) + data_shape + (1,),
        #                                          dtype=np.float32)
        #
        # if 'degree_weighted' in metrics:
        #     fmri_degree_weighted = entry.create_dataset(u'data_degree_weighted', shape=(nsubjs,) + data_shape + (1,),
        #                                                 dtype=np.float32)
        #
        # if 'eigenvector_weighted' in metrics:
        #     fmri_eigenvector_weighted = entry.create_dataset(u'data_eigenvector_weighted', shape=(nsubjs,) + data_shape + (1,),
        #                                                      dtype=np.float32)
        #
        # if 'falff' in metrics:
        #     fmri_falff = entry.create_dataset(u'data_falff', shape=(nsubjs,) + data_shape + (1,),
        #                                          dtype=np.float32)
        #
        # if 'lfcd' in metrics:
        #     fmri_lfcd = entry.create_dataset(u'data_lfcd', shape=(nsubjs,) + data_shape + (1,),
        #                                          dtype=np.float32)

        # if 'func_mask' in metrics:
        #     fmri_mask = entry.create_dataset(u'data_func_mask', shape=(nsubjs,) + data_shape + (1,),
        #                                          dtype=np.float32)

        for sub_id, (k, v) in enumerate(func_subject_vars.items()):
            t1 = time()

            if 'autocorr' in metrics:
                summary_img = convert2summary(v['rsfmri'], v['mask'], metric='autocorr')
                summary_img = np.array(summary_img, dtype=np.float32)[:, :, :, np.newaxis]
                fmri_autocorr[sub_id, :, :, :] = summary_img

            if 'entropy' in metrics:
                summary_img = convert2summary(v['rsfmri'], v['mask'], metric='entropy')
                summary_img = np.array(summary_img, dtype=np.float32)[:, :, :, np.newaxis]
                fmri_entropy[sub_id, :, :, :] = summary_img

            if 'alff' in metrics:
                summary_img = convert2summary(v['alff'], metric='alff')
                summary_img = np.array(summary_img, dtype=np.float32)[:, :, :, np.newaxis]
                fmri_alff[sub_id, :, :, :] = summary_img

            if 'degree_weighted' in metrics:
                summary_img = convert2summary(v['degree_weighted'], v['mask'], metric='degree_weighted')
                summary_img = np.array(summary_img, dtype=np.float32)[:, :, :, np.newaxis]
                fmri_degree_weighted[sub_id, :, :, :] = summary_img

            if 'eigenvector_weighted' in metrics:
                summary_img = convert2summary(v['eigenvector_weighted'], metric='eigenvector_weighted')
                summary_img = np.array(summary_img, dtype=np.float32)[:, :, :, np.newaxis]
                fmri_eigenvector_weighted[sub_id, :, :, :] = summary_img

            if 'falff' in metrics:
                summary_img = convert2summary(v['falff'], metric='falff')
                summary_img = np.array(summary_img, dtype=np.float32)[:, :, :, np.newaxis]
                fmri_falff[sub_id, :, :, :] = summary_img

            if 'lfcd' in metrics:
                summary_img = convert2summary(v['lfcd'], metric='lfcd')
                summary_img = np.array(summary_img, dtype=np.float32)[:, :, :, np.newaxis]
                fmri_lfcd[sub_id, :, :, :] = summary_img

            # import pdb; pdb.set_trace()
            # if 'func_mask' in metrics:
            #     summary_img = convert2summary(v['mask'], metric='func_mask')
            #     summary_img = np.array(summary_img, dtype=np.int8)[:, :, :, np.newaxis]
            #     fmri_mask[sub_id, :, :, :] = summary_img

            sub_labels_func.append(v['DX'])  # Diagnosis is the last element
            abide_ids_func.append(k)
            t2 = time()
            stdout.write('\rfunctional {}/{}: {:0.2}sec'.format(sub_id + 1, nsubjs, t2 - t1))
            stdout.flush()

        entry.attrs['func_labels'] = sub_labels_func
        entry.attrs['func_sub_ids'] = abide_ids_func


def get_data_shape(func_subject_vars, struct_subject_vars):
    """

    :param func_subject_vars: Dictionary with keys=sub_id and values=[func, mask, diagnosis]
    :param struct_subject_vars: Dictionary with keys=sub_id and values=[func, mask, diagnosis]
    :return: shape of the 3D output
    """
    # Get any value
    v = next(iter(func_subject_vars.values()))
    m = v['mask']

    s, _ = next(iter(struct_subject_vars.values()))

    return nib.load(m).shape, nib.load(s).shape


def run():

    data_dir   = '/data_local/deeplearning/ABIDE_SummaryMeasures'
    abide_struct_dir  = '/data_local/deeplearning/ABIDE_SummaryMeasures/T1/'
    pheno_file = osp.join(data_dir, 'subject_info.csv')

    output_hdf5_dir  = osp.join(data_dir, 'hdf5_data')
    output_hdf5_file = 'fmri_summary_abideI_II.hdf5'

    # all_metrics = ['alff', 'cleaned_mni_rsfmri', 'T1','degree_centrality_binarize',
    #                'degree_centrality_weighted', 'eigenvector_centrality_binarize', 'eigenvector_centrality_weighted',
    #                'lfcd_binarize','lfcd_weighted', 'entropy', 'reho', 'vmhc', 'autocorr', 'falff', ]
    all_metrics = ['alff', 'T1', 'degree_centrality_binarize', 'degree_centrality_weighted',
                   'eigenvector_centrality_binarize', 'eigenvector_centrality_weighted', 'lfcd_binarize',
                   'lfcd_weighted', 'entropy', 'reho', 'vmhc', 'autocorr', 'falff', ]

    func_subject_vars, struct_subject_vars = subject_dict(data_dir, abide_struct_dir, abide_rsfmri_dir, abide_funcmask_dir, pheno_file)
    create_hdf5_file(func_subject_vars, struct_subject_vars, output_hdf5_dir, output_hdf5_file, metrics=all_metrics)


if __name__ == '__main__':
    run()



