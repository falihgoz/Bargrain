import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from shutil import copyfile
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import scipy.io
import tarfile
from nilearn import image
from nilearn import datasets as ni_datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


datasets = ['abide', 'cobre', 'acpi']
output_folder = 'preprocessed_data'
data_folder = 'data'


def rest_to_tc(resting_state):
    resting_state = image.load_img(resting_state)
    print(resting_state.shape)
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_filename = atlas.maps
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
    tc = masker.fit_transform(resting_state)
    tc = tc.transpose()

    return tc

def fix_tc_padding(subject_tc, tc_length):
    if (subject_tc.shape[1] < tc_length):
        padding_needed = tc_length - subject_tc.shape[1]
        pad_width = ((0, 0), (0, padding_needed))
        subject_tc = np.pad(subject_tc, pad_width, mode='wrap')

    return subject_tc

def concat_tc_abide(tc_list, tc_raw_file, corr_measure):
    tc_length = 192
    subject_corr_graph = None

    for file_name in os.listdir(tc_raw_file):
        if file_name.endswith('.1D'):
            file_path = os.path.join(tc_raw_file, file_name)
            subject_tc = np.loadtxt(file_path)
            subject_corr_graph = np.array(
                corr_measure.fit_transform([subject_tc])[0])
            subject_tc = np.transpose(subject_tc)

            print(
                f"Processing file: {file_path}, TC shape: {subject_tc.shape}")

            subject_tc = fix_tc_padding(subject_tc, tc_length)
            tc_list = np.append(tc_list, [subject_tc[:, :tc_length]], axis=0)

    return tc_list, subject_corr_graph

def load_data(dataset):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)

    if dataset == 'abide':
        output_dataset_path = f'./{output_folder}/{dataset}'
        raw_folder = os.path.join(
            f'./{data_folder}/{dataset}/ABIDE_pcp/cpac/filt_noglobal')

        subject_ids = np.loadtxt(os.path.join(raw_folder, 'subject_ids.txt'))

        df_phenotypic = pd.read_csv(os.path.join(
            data_folder, dataset, 'Phenotypic_V1_0b_preprocessed1.csv'))
        
        # Correlation type
        corr_measure = ConnectivityMeasure(kind='correlation')

        tc = np.empty((0, 111, 192), dtype=float)
        corr_graph = np.empty((0, 111, 111), dtype=float)
        labels = np.empty(0)

        for subject_id in subject_ids:
            subject_info = df_phenotypic.loc[df_phenotypic['SUB_ID']
                                             == subject_id]
            if subject_info.empty:
                continue

            tc_raw_file = os.path.join(raw_folder, str(int(subject_id)))
            tc, subject_corr_graph = concat_tc_abide(tc, tc_raw_file, corr_measure)
            corr_graph = np.append(corr_graph, [subject_corr_graph], axis=0)
            labels = np.append(labels, subject_info.iloc[0]['DX_GROUP']-1)

        preprocessed_data = {
            'tc': tc,
            'corr_graph': corr_graph,
            'labels': labels
        }

        if not os.path.exists(output_dataset_path):
            os.makedirs(output_dataset_path)
        np.save(f'{output_dataset_path}/{dataset}.npy', preprocessed_data)

    elif dataset == 'cobre':
        raw_folder = os.path.join(f'./{data_folder}/{dataset}')
        with tarfile.open(f'{raw_folder}/COBRE_scan_data.tgz', "r:gz") as tar:
            tar.extractall(path=raw_folder)

        output_dataset_path = f'./{output_folder}/{dataset}'
        if not os.path.exists(output_dataset_path):
            os.makedirs(output_dataset_path)

        df_phenotypic = pd.read_csv(os.path.join(
            raw_folder, 'COBRE_phenotypic_data.csv'))
        subject_ids = df_phenotypic.iloc[:-2, 0].astype(str).to_numpy()

        # Atlas for parcellation
        atlas = ni_datasets.fetch_atlas_harvard_oxford(
            'cortl-maxprob-thr25-2mm')
        atlas_filename = atlas.maps
        masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

        # Correlation type
        corr_measure = ConnectivityMeasure(kind='correlation')

        tc = np.empty((0, 96, 150), dtype=float)
        corr_graph = np.empty((0, 96, 96), dtype=float)
        labels = np.empty(0)

        for subject_id in subject_ids:
            subject_info = df_phenotypic[df_phenotypic.iloc[:, 0].astype(
                int) == int(subject_id)]

            if subject_info.empty:
                continue

            rest_file = os.path.join(
                raw_folder, f'COBRE/00{subject_id}/session_1/rest_1/rest.nii.gz')
            rest_state = image.load_img(rest_file)
            subject_tc = masker.fit_transform(rest_state)
            
            if subject_tc.shape[1] != tc.shape[1] or subject_tc.shape[0] < tc.shape[2]:
                continue

            subject_corr_graph = np.array(
                corr_measure.fit_transform([subject_tc])[0])
            corr_graph = np.append(corr_graph, [subject_corr_graph], axis=0)

            subject_tc = np.transpose(subject_tc)
            tc = np.append(tc, [subject_tc], axis=0)

            label = 1 if subject_info.iloc[0]['Subject Type'] == "Patient" else 0
            labels = np.append(labels, label)

            print(
                f"Processing file: {rest_file}, TC shape: {subject_tc.shape}")

        preprocessed_data = {
            'tc': tc,
            'corr_graph': corr_graph,
            'labels': labels
        }

        if not os.path.exists(output_dataset_path):
            os.makedirs(output_dataset_path)
        np.save(f'{output_dataset_path}/{dataset}.npy', preprocessed_data)

    elif dataset == 'acpi':
        raw_folder = os.path.join(f'./{data_folder}/{dataset}')
        with tarfile.open(f'{raw_folder}/mta_1_ts_cc200_rois.tar.gz', "r:gz") as tar:
            tar.extractall(path=raw_folder)

        output_dataset_path = f'./{output_folder}/{dataset}'
        if not os.path.exists(output_dataset_path):
            os.makedirs(output_dataset_path)

        df_phenotypic = pd.read_csv(os.path.join(
            raw_folder, 'mta_1_phenotypic_data.csv'))
        subject_ids = df_phenotypic.iloc[:-2, 0].astype(str).to_numpy()

        # Correlation type
        corr_measure = ConnectivityMeasure(kind='correlation')

        tc = np.empty((0, 200, 700), dtype=float)
        tc_length = tc.shape[2]
        corr_graph = np.empty((0, 200, 200), dtype=float)
        labels = np.empty(0)


        for subject_id in subject_ids:
            subject_info = df_phenotypic[df_phenotypic.iloc[:, 0].astype(
                int) == int(subject_id)]
            if subject_info.empty:
                continue

            tc_path = os.path.join(
                raw_folder, f'00{subject_id}-session_1/ts_cc200_rois.csv')
            try:
                subject_tc = pd.read_csv(tc_path)
            except FileNotFoundError:
                continue

            subject_tc = subject_tc.to_numpy()
            subject_tc = subject_tc[:, 1:]
            subject_tc = fix_tc_padding(subject_tc, tc_length)
            tc = np.append(tc, [subject_tc[:, :tc_length]], axis=0)

            subject_corr_graph = np.array(corr_measure.fit_transform(
                [np.transpose(subject_tc)])[0])
            corr_graph = np.append(corr_graph, [subject_corr_graph], axis=0)

            label = subject_info.iloc[0]['MJUser']
            labels = np.append(labels, label)

            print(f"Processing file: {tc_path}, TC shape: {subject_tc.shape}")

        preprocessed_data = {
            'tc': tc,
            'corr_graph': corr_graph,
            'labels': labels,
        }

        if not os.path.exists(output_dataset_path):
            os.makedirs(output_dataset_path)
        np.save(f'{output_dataset_path}/{dataset}.npy', preprocessed_data)

    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')


if __name__ == '__main__':
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            load_data(d)
    else:
        print("Usage: python preprocess_data.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")