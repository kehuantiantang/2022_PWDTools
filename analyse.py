# coding=utf-8
import os
import os.path as osp
from collections import defaultdict
from copy import deepcopy

from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

from misc.pascal_voc_utils import Reader, PaddingModify, Writer
from misc.utils import load_pkl, save_pkl


def box_scales(path, target_path):
    '''
    获得所有bbox的大小尺度
    :param path:
    :return:
    '''
    scale_counters = defaultdict(set)
    min_height, min_width, max_height, max_width = sys.maxsize, sys.maxsize, 0, 0
    minH_bbox, minW_bbox, maxH_bbox, maxW_bbox = None, None, None, None
    for root, _, filenames in os.walk(path):
        for filename in tqdm(filenames, desc=root):
            if filename.endswith('xml'):
                op = osp.join(root, filename)
                obj = Reader(op).get_objects()

                for c, box in zip(obj['category_name'], obj['bboxes']):
                    x_min, y_min, x_max, y_max = box
                    height, width = y_max - y_min, x_max - x_min
                    scale_counters[c].add((height, width))

                    if height < min_height:
                        min_height = height
                        minH_bbox = (height, width)

                    if width < min_width:
                        min_width = width
                        minW_bbox = (height, width)

                    if height > max_height:
                        max_height = height
                        maxH_bbox = (height, width)

                    if width > max_width:
                        max_width = width
                        maxW_bbox = (height, width)

    print('minH_bbox, minW_bbox, maxH_bbox, maxW_bbox')
    print(minH_bbox, minW_bbox, maxH_bbox, maxW_bbox)
    save_pkl(target_path, scale_counters)


def scatter_image(pkl_path):
    '''
    read pkl file and show the box scales
    :param pkl_path:
    :return:
    '''
    data = load_pkl(pkl_path)
    fig, axs = plt.subplots(len(data.keys())+1)
    for index, c in enumerate(data.keys()):
        for (h, w) in tqdm(data[c], desc=c):
            axs[index].scatter(h, w, c = 'r', alpha = 0.8, s = 2)
        axs[index].set_xlim(0, 250)
        axs[index].set_ylim(0, 250)
    plt.title(pkl_path)
    plt.show()

def bbox_img_counter(path, count_img = False):
    counter, img_counter = defaultdict(int), 0
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('xml'):
                objs = Reader(osp.join(root, filename)).get_objects()
                for label in objs['category_name']:
                    counter[label] += 1
                if count_img:
                    if osp.exists(osp.join(root, filename).replace('xml', 'jpg')):
                        img_counter += 1

    print(counter, img_counter)


def enlarge_xml(path, target_path, padding_pixel):
    '''
    >>> enlarge_xml('/Users/sober/Downloads/800/test/large_xml',
    >>>             '/Users/sober/Downloads/make_data/enlarge_xml',
    >>>             padding_pixel=15)
    :param path:
    :param target_path:
    :param padding_pixel:
    :return:
    '''
    os.makedirs(target_path, exist_ok=True)
    for root, _, filenames in os.walk(path):
        # sub_path = root.replace(path, '').strip('/')
        for filename in tqdm(sorted(filenames)):
            if filename.endswith(".xml"):
                op = os.path.join(root, filename)
                # tp_root = os.path.join(target_path, sub_path)
                # os.makedirs(tp_root, exist_ok=True)

                # tp = os.path.join(tp_root, filename.replace('vis', 'rgb'))

                PaddingModify(op, op.replace(root, target_path), padding_pixel).run()

def drop_small(path, target_path, min_size = 0.01):
    os.makedirs(target_path, exist_ok=True)
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('xml'):
                objs = Reader(osp.join(root, filename)).get_objects()
                width, height = objs['width'], objs['height']
                obj_dicts = deepcopy(objs)
                obj_dicts['name'], obj_dicts['bboxes'], obj_dicts['category_name'], obj_dicts[
                    'difficult'] = [], [], [], []
                for name, category_name, bbox, difficult in zip(objs['name'], objs['category_name'], objs['bboxes'],
                                                                objs['difficult']):
                    xmin, ymin, xmax, ymax = bbox
                    if xmax - xmin <= width * min_size or ymax - ymin <= height *min_size:
                        print('Drop: ', filename, (xmin, ymin, xmax, ymax))
                    else:
                        obj_dicts['name'].append(name)
                        obj_dicts['category_name'].append(name)
                        obj_dicts['bboxes'].append(bbox)
                        obj_dicts['difficult'].append(difficult)

                if len(obj_dicts['bboxes']) > 0:
                    writer = Writer(obj_dicts['path'], obj_dicts['width'], obj_dicts['height'], database =
                    osp.split(obj_dicts['path'])[0])
                    writer.addBboxes(obj_dicts['bboxes'], obj_dicts['category_name'])
                    writer.save(osp.join(root, filename).replace(path, target_path))


def check_bbox(path, min_size = 0.01):
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('xml'):
                objs = Reader(osp.join(root, filename)).get_objects()
                width, height = objs['width'], objs['height']
                for (xmin, ymin, xmax, ymax) in objs['bboxes']:
                     if xmax - xmin <= width * min_size or ymax - ymin <= height *min_size:
                         print(filename, (xmin, ymin, xmax, ymax))

def a(path):
    counter = 0
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('jpg'):
                if int(filename.split('.')[0].split('_')[-1]) == 0:
                    counter += 1
    print(counter)

if __name__ == '__main__':
    # drop_small('/Users/sober/Downloads/Project/2022_pwd/disease_split/disease',
    #            '/Users/sober/Downloads/Project/2022_pwd/disease_split/disease_drop_small')

    # enlarge_xml('/Users/sober/Downloads/Project/2022_pwd/disease_split/disease_drop_small',
    #             '/Users/sober/Downloads/Project/2022_pwd/disease_split/disease_enlarge8', padding_pixel= 8)


    # check_bbox('/Users/sober/Downloads/Project/2022_pwd/disease_split/disease')

    # box_scales(r'E:\make_pine_data\rgb\auged_data\val\xml', r'H:\val_aug_scales.pkl')
    # scatter_image(r"H:\val_aug_scales.pkl")

    # bbox_img_counter('/Users/sober/Downloads/Project/2022_pwd/disease_split/yolo/val', True)
    # a('/Users/sober/Downloads/Project/2022_pwd/disease_split/yolo/val_with_normal')

    import cv2
    import numpy as np
    img = cv2.imread('/Users/sober/Downloads/Project/2022_pwd/temp/kangsan/test/img/AP_IMAGE_512/FR_GS_AP_35712028_443.tif', cv2.IMREAD_UNCHANGED)
    print(img.shape, np.unique(img))
    # p = '/Users/sober/Downloads/Project/2022_pwd/temp/zezhu/test/label/PI_512/METADATA/FR_JJ_PI_6049_FGT_META.json'
    # # encoding = 'cp1250', errors = 'ignore'
    # from bs4 import UnicodeDammit
    # with open(p, encoding='utf-8-sig' ) as f:
    #     context = f.read()
    #     print(context)
    #     print(context[1:])

        # suggestion = UnicodeDammit(context)
        # print(suggestion.original_encoding)