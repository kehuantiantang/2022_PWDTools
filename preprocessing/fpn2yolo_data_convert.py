# coding=utf-8
import shutil

import pandas as pd
import os
import os.path as osp

from misc.pascal_voc_utils import Writer


def generate_trainVal_folders(fold_path, target_path, img_dir, train_fold = ['0', '1', '2', '3'], val_fold = ['4'],
                              normal_image_dir = None):
    fold_df = pd.read_csv(fold_path)
    fold_df = fold_df.astype(str)
    train_ids = fold_df[fold_df.fold.isin(train_fold)]['image_id'].values
    val_ids = fold_df[fold_df.fold.isin(val_fold)]['image_id'].values

    # os.makedirs(osp.join(target_path, 'xml'), exist_ok=True)
    os.makedirs(osp.join(target_path, 'train'), exist_ok=True)
    os.makedirs(osp.join(target_path, 'val'), exist_ok=True)
    os.makedirs(target_path, exist_ok=True)


    train_txt, val_txt, train_count, val_count, val_normal_count = [], [], 0, 0, 0
    for id in train_ids:
        img_path = osp.join(img_dir, '%s.jpg'%id)
        if osp.exists(img_path):
            # train_txt.append(img_path)
            shutil.copy(img_path, osp.join(target_path, 'train'))
            shutil.copy(img_path.replace('jpg', 'xml'), osp.join(target_path, 'train'))
            train_txt.append(osp.join(target_path, 'train', '%s.jpg'%id))
            train_count += 1
        else:
            print('File not exist: %s'%img_path)

    for id in val_ids:
        img_path = osp.join(img_dir, '%s.jpg'%id)
        if osp.exists(img_path):
            # val_txt.append(img_path)
            shutil.copy(img_path, osp.join(target_path, 'val'))
            shutil.copy(img_path.replace('jpg', 'xml'), osp.join(target_path, 'val'))
            val_txt.append(osp.join(target_path, 'val', '%s.jpg'%id))
            val_count += 1
        else:
            print('File not exist: %s'%img_path)

    # with open(osp.join(target_path, 'train.txt'), 'w') as f:
    #     f.write('\n'.join(train_txt))
    # with open(osp.join(target_path, 'val.txt'), 'w') as f:
    #     f.write('\n'.join(val_txt))

    if normal_image_dir is not None:
        val_with_normal = []
        os.makedirs(osp.join(target_path, 'val_with_normal'), exist_ok=True)
        for root, _, filenames in os.walk(normal_image_dir):
            for filename in filenames:
                if filename.endswith('jpg'):
                    shutil.copy(osp.join(root, filename), osp.join(target_path, 'val_with_normal'))
                    val_with_normal.append(osp.join(target_path, 'val_with_normal', filename))

                    # the size of image
                    writer = Writer(osp.join(target_path, 'val_with_normal', filename), 768, 768)
                    writer.save(osp.join(target_path, 'val_with_normal', filename.replace('jpg', 'xml')))

                    val_normal_count += 1

        for p in val_txt:
            shutil.copy(p, osp.join(target_path, 'val_with_normal'))
            shutil.copy(p.replace('jpg', 'xml'), osp.join(target_path, 'val_with_normal'))

            filename = osp.split(p)[-1]
            val_with_normal.append(osp.join(target_path, 'val_with_normal', filename))
            val_normal_count += 1


        # with open(osp.join(target_path, 'val_with_normal.txt'), 'w') as f:
        #     f.write('\n'.join(val_with_normal))

    print('The number of train/val files: %d/%d'%(train_count, val_count))

def replace_root_fold(path, source_root, target_root):
    with open(path, 'r') as f:
        context = f.read()
        context = context.replace(source_root, target_root)

    with open(path, 'w') as f:
        f.write(context)

if __name__ == '__main__':
    generate_trainVal_folders('/Users/sober/Downloads/Project/2022_pwd/disease_split/fold.csv',
                          '/Users/sober/Downloads/Project/2022_pwd/disease_split/yolo',
                          '/Users/sober/Downloads/Project/2022_pwd/disease_split/disease',
                              normal_image_dir='/Users/sober/Downloads/Project/2022_pwd/disease_split/no_disease')

    # generate_trainVal_txt('/Users/sober/Downloads/Project/2022_pwd/disease_split/fold.csv',
    #                       '/Users/sober/Downloads/Project/2022_pwd/disease_split/yolo',
    #                       '/Users/sober/Downloads/Project/2022_pwd/disease_split/disease',
    #                       normal_image_dir=None)

    # replace_root_fold('/Users/sober/Downloads/Project/2022_pwd/disease_split/yolo/train.txt',
    #                   '/Users/sober/Downloads/Project/2022_pwd/disease_split/yolo/',
    #                   './')
    #
    # replace_root_fold('/Users/sober/Downloads/Project/2022_pwd/disease_split/yolo/val.txt',
    #                   '/Users/sober/Downloads/Project/2022_pwd/disease_split/yolo/',
    #                   './')
    #
    #
    # replace_root_fold('/Users/sober/Downloads/Project/2022_pwd/disease_split/yolo/val_with_normal.txt',
    #                   '/Users/sober/Downloads/Project/2022_pwd/disease_split/yolo/',
    #                   './')




