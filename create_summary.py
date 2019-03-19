import os
from time import time
from sys import stdout
import pandas as pd
import nibabel as nib
import numpy as np
import h5py
import os.path as osp


def read_summary(df_pheno, metric='entropy'):
    """

    :param df_pheno: pandas df containing path to the 4-D resting-state or other abide summary measures.
    :return: 4D nifti-file with metric values in each voxels for all subjects dim -> [nsubjects, <vol_dims>].
    """
    n_subjects = len(df_pheno)
    metric_all_subjects = np.zeros((n_subjects, 45, 54, 45))    # MNI 4mm images

    for i, row in df_pheno.iterrows():
        metric_all_subjects[i, :, :, :] = np.squeeze(nib.load(row[metric]).get_data())

    return np.array(metric_all_subjects, dtype=np.float32)[:, :, :, :, np.newaxis]


def create_hdf5_file(hdf5_file, pheno_file, metrics):
    """

    :param hdf5_file: filename of the hdf5_file
    :param pheno_file: csv file with file paths and DX groups
    :param metric: [entropy, ALFF, pALFF, mean or any others
    :return: hdf5 file with the metric of interest[entropy, ALFF, pALFF, mean]
    """

    df_pheno = pd.read_csv(pheno_file)

    n_subjects = len(df_pheno)
    data_shape = (45, 54, 45)

    with h5py.File(os.path.join(hdf5_file), 'w') as hfile:

        entry = hfile.create_group(u'summaries')

        for metric in metrics:

            t1 = time()
            dataset = entry.create_dataset(metric, shape=(n_subjects, ) + data_shape + (1,), dtype=np.float32)
            dataset[:, :, :, :] = read_summary(df_pheno, metric=metric)
            t2 = time()
            stdout.write('\rMetric {}: {:0.2}sec'.format(metric, t2 - t1))
            stdout.flush()

        # Map these columns from df_pheno for later

        entry.attrs['SUB_ID'] = df_pheno['SUB_ID'].values
        entry.attrs['SITE_ID'] = np.string_(list(df_pheno['SITE_ID']))
        entry.attrs['AGE_AT_SCAN'] = df_pheno['AGE_AT_SCAN'].values
        entry.attrs['DX_GROUP'] = df_pheno['DX_GROUP'].values
        entry.attrs['SEX'] = df_pheno['SEX'].values
        entry.attrs['ABIDE_I_or_II'] = df_pheno['ABIDE_I_or_II'].values


def run():

    data_dir = '/data_local/deeplearning/ABIDE_SummaryMeasures'
    pheno_file = osp.join(data_dir, 'subject_info.csv')

    output_hdf5_file = osp.join(data_dir, 'fmri_summary_abideI_II.hdf5')

    # all_metrics = ['alff', 'cleaned_mni_rsfmri', 'T1','degree_centrality_binarize',
    #                'degree_centrality_weighted', 'eigenvector_centrality_binarize', 'eigenvector_centrality_weighted',
    #                'lfcd_binarize','lfcd_weighted', 'entropy', 'reho', 'vmhc', 'autocorr', 'falff', ]

    all_metrics = ['alff', 'T1', 'degree_centrality_binarize', 'degree_centrality_weighted',
                   'eigenvector_centrality_binarize', 'eigenvector_centrality_weighted', 'lfcd_binarize',
                   'lfcd_weighted', 'entropy', 'reho', 'vmhc', 'autocorr', 'falff', ]

    create_hdf5_file(output_hdf5_file, pheno_file, metrics=all_metrics)


if __name__ == '__main__':
    run()



