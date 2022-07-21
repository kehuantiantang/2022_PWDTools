# coding=utf-8
import os
import os.path as osp
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from tqdm import tqdm

from pascal_voc_utils import Reader


def split_disease_noDisease(path, target):
    os.makedirs(osp.join(target, 'disease'), exist_ok= True)
    os.makedirs(osp.join(target, 'no_disease'), exist_ok= True)
    nb_disease_img, nb_no_disease_img = 0, 0
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('jpg'):
                name =  filename.split('.')[0]
                nb_disease = int(name.split('_')[-1])
                if nb_disease > 0:
                    shutil.copy(osp.join(root, filename), osp.join(target, 'disease'))
                    shutil.copy(osp.join(root, filename.replace('jpg', 'xml')), osp.join(target, 'disease'))
                    nb_disease_img += 1
                else:
                    shutil.copy(osp.join(root, filename), osp.join(target, 'no_disease'))
                    nb_no_disease_img += 1
    print('Disease/No disease image: %d/%d'%(nb_disease_img, nb_no_disease_img))


def make_dataset(source, target_path, target_cls, resolution_info = None):
    label_dict = {
        'back_ground': 0,
        'disease': 1,
        'neg':2,
        'oak':2,
        'hn':3,
        'wb':4,
        'ub':4,
        'wg':5,
        'disease_sim':6,
        'disease_am':7,
        'yellow':8,
        'bd':9,
        'maple':10,
        'tiny':11,
        'hn2':11,
    }
    source_info = ['other' for _ in range(100000000)]
    if resolution_info:
        ri = pd.read_csv(resolution_info)
        for _, row in ri.iterrows():
            for i in range(row['start'], row['end']+1):
                source_info[i] = row['name']


    data = defaultdict(list)
    for root, _, filenames in os.walk(source):
        for filename in tqdm(filenames, desc = root):
            if filename.endswith('jpg'):
                # xml = osp.join(root, filename).replace('disease', 'xml').replace('img', 'xml').replace('jpg',
                #                                                                                        'xml')#.replace('all', 'xml')
                xml = osp.join(root, filename).replace('jpg', 'xml')
                if osp.exists(xml):
                    info = Reader(xml).get_objects()
                    for bbox, name in zip(info['bboxes'], info['name']):
                        if name in target_cls:
                            id = osp.basename(filename).split('.')[0]
                            data['image_id'].append(id)
                            data['width'].append(768)
                            data['height'].append(768)
                            data['bbox'].append(bbox)
                            data['nb_bbox'].append(len(info['bboxes']))
                            data['label'].append(label_dict[name])
                            data['name'].append(name)
                            try:
                                data['source'].append(source_info[int(id[4:])])
                            except:
                                data['source'].append('')
                else:
                    data['image_id'].append(osp.basename(filename).split('.')[0])
                    data['width'].append(768)
                    data['height'].append(768)
                    data['bbox'].append(None)
                    data['nb_bbox'].append(0)
                    # no bbox image
                    data['label'].append(0)
                    data['source'].append('normal_img')


    os.makedirs(target_path, exist_ok=True)
    df = pd.DataFrame.from_dict(data)
    df.sort_values(by=['image_id'], inplace=True)
    print(df.info(verbose = True))
    print(df.head())
    df.to_csv(osp.join(target_path, 'data.csv'), index = False)


def prepare_fold(df_path, target_path, test_image_paths = None, num_fold = 5):
    df = pd.read_csv(df_path)
    # skf = StratifiedKFold(n_splits=num_fold, shuffle = True, random_state=1)

    skf = ShuffleSplit(n_splits=2, shuffle = True, random_state=1)

    df_folds = df[['image_id', 'source', 'name']].copy()

    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = df[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x}').values.astype(str)

    )

    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    df_folds['image_id'] = df_folds.index

    if test_image_paths is not None:
        fold = -1
        test_image_paths = test_image_paths.split('|')
        test_dict = defaultdict(list)
        for test_image_path in test_image_paths:
            for _, _, filenames in os.walk(test_image_path):
                for filename in filenames:
                    if filename.endswith('jpg'):
                        test_dict['image_id'].append(filename.split('.')[0])
                        test_dict['fold'].append(fold)

                        test_dict['stratify_group'].append('normal%d'%fold)

            fold -= 1

        df_folds = pd.concat([df_folds, pd.DataFrame.from_dict(test_dict)])

    # df_folds.sort_values(by = ['image_id'], inplace=True)
    df_folds.to_csv(osp.join(target_path, 'fold.csv'), index = False)

import cv2
if __name__ == '__main__':
    # split_disease_noDisease('/Users/sober/Downloads/Project/2022_pwd/train',
    #                         '/Users/sober/Downloads/Project/2022_pwd/disease_split')

    # make_dataset('/Users/sober/Downloads/Project/2022_pwd/disease_split/disease',
    #              '/Users/sober/Downloads/Project/2022_pwd/disease_split' ,  target_cls = ['disease', 'background'])

    # prepare_fold('/home/khtt/code/detector2FPN/ssd/clear_data/data.csv',  '/home/khtt/code/detector2FPN/ssd/clear_data',  '/home/khtt/code/detector2FPN/ssd/img')

    path = '/Users/sober/Downloads/Project/2022_pwd/20220720_r/mask'
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('png'):
                img = cv2.imread(osp.join(root, filename))

                value = np.unique(img)
                print(filename, value)