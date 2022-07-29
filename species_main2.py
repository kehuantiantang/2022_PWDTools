# coding=utf-8
import pprint
from collections import OrderedDict

from tqdm import tqdm

from json_polygon import JsonLoader
import numpy as np
import cv2
import os
import os.path as osp

from pascal_voc_utils import Writer
from util import save_json

index2cls = {0:'판독불가', 110:'소나무', 120:'낙엽송', 130:'기타침엽수', 140:'활엽수',  150:'침엽수', 160:'기타침엽수', 170:'삼나무', 180:'삼나무',
             190:'비산림'}

def get_color_dict(name):
    # label_colours = {'bg':(0, 0, 0)  # 0=background
    #     , '판독불가':(255, 0, 0), '비산림':(0, 255, 0), '상록활엽수':(0, 0, 255), 'a':(255, 255, 0), 'a':(255, 0, 255)
    #     , 'a':(0, 128, 128), 'a':(128, 128, 128), 'a':(64, 0, 0), 'a':(192, 128, 0), 'a':(64, 0, 128), 'a':(192, 0, 0), 'a':(64, 128, 0),  'a':(192, 0, 128), 'a':(64, 128, 128),
    #                  'a':(192, 128, 128), 'a':(0, 64, 0), 'a':(128, 64, 0), 'a':(0, 192, 0), 'a':(128, 192, 0), 'a':(0, 64, 128)}
    label_colours = {'판독불가':(0, 0, 0)  # 0=background
        , '소나무':(1, 255, 0), '삼나무':(2, 0, 255), '낙엽송':(3, 128, 0), '침엽수':(4, 0, 128), '기타침엽수':(5, 128, 128)
        , '활엽수':(6, 60, 60), '상록활엽수':(7, 60, 0), '비산림':(8, 0, 60), 'a':(9, 30, 30), 'a':(64, 0, 128), 'a':(192, 0,
                                                                                                              0), 'a':(64, 128, 0),  'a':(192, 0, 128), 'a':(64, 128, 128),
                     'a':(192, 128, 128), 'a':(0, 64, 0), 'a':(128, 64, 0), 'a':(0, 192, 0), 'a':(128, 192, 0), 'a':(0, 64, 128)}
    return label_colours[name]

if __name__ == '__main__':

    folder = 'kangsan'
    train_test, directory = 'test', 'PI_IMAGE_512'
    root_img_path = '/Users/sober/Downloads/Project/2022_pwd/temp/%s/%s/img/%s'%(folder, train_test, directory)
    root_label_path = '/Users/sober/Downloads/Project/2022_pwd/temp/%s/%s/label/%s'%(folder, train_test,
                                                                                     directory.replace(
        '_IMAGE', ''))
    target_path = '/Volumes/2022 AI 산림해충 방제 시스템/수종데이트/maked_data/%s/%s/%s'%(folder, train_test,
directory.replace('_IMAGE', ''))
    vis_path, mask_path, img_path = osp.join(target_path, 'vis'), \
                                                         osp.join(target_path,'mask'), osp.join(target_path, 'img')
    [os.makedirs(i, exist_ok= True) for i in [vis_path, mask_path, img_path]]

    for root, _, filenames in os.walk(root_img_path):
        for filename in tqdm(sorted(filenames), desc=root):
            if filename.lower().endswith('tif'):
                name = filename.split('.')[0].replace('_1024', '')
                # 1024, 512
                res = osp.split(root)[-1].split('_')[-1]
                root_img_path = osp.join(root, filename)

                vis_img_path = osp.join(root_label_path, 'FGT_TIF', '%s_FGT_%s.tif'%(name, res))
                if not osp.exists(vis_img_path):
                    vis_img_path = osp.join(root_label_path, 'FGT_TIF', '%s_FGT.tif'%(name))


                # img
                img = cv2.imread(root_img_path)
                cv2.imwrite(osp.join(img_path, '%s.jpg'%name), img)

                vis = cv2.imread(vis_img_path, cv2.IMREAD_UNCHANGED)

                # mask
                mask = np.zeros_like(img)
                try:
                    for p in np.unique(vis):
                        mask[vis == p] = get_color_dict(index2cls[p])
                except Exception as e:
                    print(vis_img_path, e)
                    print(root_img_path)
                    print(np.unique(vis), img.shape)
                cv2.imwrite(osp.join(mask_path, '%s.png'%name), mask)


                vis = cv2.imread(vis_img_path)
                vis = cv2.hconcat([img, mask, vis])
                cv2.imwrite(osp.join(vis_path, '%s.jpg'%name), vis)





