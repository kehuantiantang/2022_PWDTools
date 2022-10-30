# coding=utf-8
from collections import OrderedDict

from tqdm import tqdm

from preprocessing.json_polygon import JsonLoader
import numpy as np
import cv2
import os
import os.path as osp

from misc.pascal_voc_utils import Writer
from misc.util import save_json


def get_objects(contexts):
    meta_context, context = contexts
    meta_context = meta_context[0]
    geo_x, geo_y = meta_context.coordinates.split(',')
    geo_x, geo_y = float(geo_x.strip()), float(geo_y.strip())

    height, width = int(meta_context.img_height), int(meta_context.img_width)
    resolution, coordinate = float(meta_context.img_resolution), meta_context.img_coordinate
    path = meta_context.img_id

    # height, width, path, resolution, coordinate = 0, 0, 0, 0, 0


    obj_dicts = OrderedDict({'name': [], 'bboxes': [], 'category_name': [], 'name_pattern': '', 'height': height,
                     'width': width,
                 'path': path, 'polygons': [], 'filename': path, 'resolution': resolution, 'coordinate': coordinate})

    # try:
    for object in context.features:

        name = object.properties.ANN_NM.strip()
        geo_type = object.geometry.type
        if geo_type.lower() == 'polygon':
            geo_points = object.geometry.coordinates[0]
        elif geo_type.lower() == 'multipolygon':
            geo_points = object.geometry.coordinates[0][0]
        else:
            assert ValueError('geo_type not belong to any type')
        img_points = []

        hs, ws = [], []
        for point in geo_points:
            h, w = point
            h, w = h - geo_x, geo_y - w
            h, w = h / resolution, w / resolution
            # print(h - geo_x, geo_y - w)
            h, w = np.clip(h, 0, height), np.clip(w, 0, width)
            h, w = int(round(h)), int(round(w))
            hs.append(h)
            ws.append(w)

            img_points.append([h, w])

        xmin, ymin, xmax, ymax = max(min(hs), 0), max(min(ws), 0), min(max(hs), width), min(max(ws), height)
        if xmin >= xmax or ymin >= ymax:
            continue

        obj_dicts['name'].append(name)
        obj_dicts['bboxes'].append([xmin, ymin, xmax, ymax])
        obj_dicts['category_name'].append(name)
        obj_dicts['polygons'].append(img_points)
    # except Exception as e:
    #     print(path, e)

    return obj_dicts


def get_zezhu_objects(contexts):
    meta_context, context = contexts
    meta_context = meta_context[0]
    # geo_x, geo_y = meta_context.coordinates.split(',')
    geo_y, geo_x = meta_context.coordinates.split(',')
    geo_x, geo_y = float(geo_x.strip()), float(geo_y.strip())

    height, width = int(meta_context.img_height), int(meta_context.img_width)
    resolution, coordinate = float(meta_context.img_resolution), meta_context.img_coordinate
    path = meta_context.img_id

    # height, width, path, resolution, coordinate = 0, 0, 0, 0, 0


    obj_dicts = OrderedDict({'name': [], 'bboxes': [], 'category_name': [], 'name_pattern': '', 'height': height,
                             'width': width,
                             'path': path, 'polygons': [], 'filename': path, 'resolution': resolution, 'coordinate': coordinate})

    # try:
    for object in context.features:

        name = object.properties.ANN_NM.strip()
        geo_type = object.geometry.type
        if geo_type.lower() == 'polygon':
            geo_points = object.geometry.coordinates[0]
        elif geo_type.lower() == 'multipolygon':
            geo_points = object.geometry.coordinates[0][0]
        else:
            assert ValueError('geo_type not belong to any type')
        img_points = []

        hs, ws = [], []
        for point in geo_points:
            h, w = point
            h, w = h - geo_x, geo_y - w
            h, w = h / resolution, w / resolution
            # print(h - geo_x, geo_y - w)
            h, w = np.clip(h, 0, height), np.clip(w, 0, width)
            h, w = int(round(h)), int(round(w))
            hs.append(h)
            ws.append(w)

            img_points.append([h, w])

        xmin, ymin, xmax, ymax = max(min(hs), 0), max(min(ws), 0), min(max(hs), width), min(max(ws), height)
        if xmin >= xmax or ymin >= ymax:
            continue

        obj_dicts['name'].append(name)
        obj_dicts['bboxes'].append([xmin, ymin, xmax, ymax])
        obj_dicts['category_name'].append(name)
        obj_dicts['polygons'].append(img_points)
    # except Exception as e:
    #     print(path, e)

    return obj_dicts

def get_color_dict():
    # label_colours = {'bg':(0, 0, 0)  # 0=background
    #     , '판독불가':(255, 0, 0), '비산림':(0, 255, 0), '상록활엽수':(0, 0, 255), 'a':(255, 255, 0), 'a':(255, 0, 255)
    #     , 'a':(0, 128, 128), 'a':(128, 128, 128), 'a':(64, 0, 0), 'a':(192, 128, 0), 'a':(64, 0, 128), 'a':(192, 0, 0), 'a':(64, 128, 0),  'a':(192, 0, 128), 'a':(64, 128, 128),
    #                  'a':(192, 128, 128), 'a':(0, 64, 0), 'a':(128, 64, 0), 'a':(0, 192, 0), 'a':(128, 192, 0), 'a':(0, 64, 128)}
    label_colours = {'bg':(0, 0, 0)  # 0=background
        , '소나무':(1, 255, 0), '삼나무':(2, 0, 255), '낙엽송':(3, 128, 0), '침엽수':(4, 0, 128), '기타침엽수':(5, 128, 128)
        , '활엽수':(6, 60, 60), '상록활엽수':(7, 60, 0), '비산림':(8, 0, 60), '판독불가':(9, 30, 30), 'a':(64, 0, 128), 'a':(192, 0,
                                                                                                              0), 'a':(64, 128, 0),  'a':(192, 0, 128), 'a':(64, 128, 128),
                     'a':(192, 128, 128), 'a':(0, 64, 0), 'a':(128, 64, 0), 'a':(0, 192, 0), 'a':(128, 192, 0), 'a':(0, 64, 128)}
    return label_colours

if __name__ == '__main__':
    # jl = JsonLoader(get_objects)
    # name = 'FR_JJ_AP_33607060_010_FGT'
    # meta_path, json_path = '/Users/sober/Downloads/Project/%s_META.json'%name, \
    #                     '/Users/sober/Downloads/Project/%s.json'%name
    # meta_context = jl.load_json(meta_path, encoding = 'EUC-KR')
    # context = jl.load_json(json_path, encoding = None)
    #
    # attributes = jl.get_objects([meta_context, context])
    # # print(context.features[0].geometry.coordinates)
    # print(attributes)
    #
    # img_path = '/Users/sober/Downloads/Project/%s.tif'%(name.replace('_CGT', '').replace('_FGT', ''))
    # jpg_img = cv2.imread(img_path)
    # jpg_mask = jl.draw_mask(jpg_img, attributes, color_dict = get_color_dict())
    # print(jpg_img.shape, jpg_mask.shape)
    #
    # vis_im = cv2.addWeighted(jpg_img, 0.8, jpg_mask, 0.2, 0)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(vis_im)
    # plt.show()

    jl = JsonLoader(get_zezhu_objects)
    train_test, directory = 'test', 'PI_IMAGE_512'
    root_img_path = '/Users/sober/Downloads/Project/2022_pwd/temp/zezhu/%s/img/%s'%(train_test, directory)
    root_label_path = '/Users/sober/Downloads/Project/2022_pwd/temp/zezhu/%s/label/%s'%(train_test, directory.replace(
        '_IMAGE', ''))
    target_path = '/Volumes/2022 AI 산림해충 방제 시스템/수종데이트/maked_data/zezhu/%s/%s'%(train_test,
    directory.replace('_IMAGE', ''))
    vis_path, mask_path, img_path, json_path, xml_path = osp.join(target_path, 'vis'), \
                                                         osp.join(target_path,'mask'), \
                                                         osp.join(target_path, 'img'), \
                                                         osp.join(target_path, 'json'), \
                                                         osp.join(target_path, 'xml')
    [os.makedirs(i, exist_ok= True) for i in [vis_path, mask_path, img_path, json_path, xml_path]]

    for root, _, filenames in os.walk(root_img_path):
        for filename in tqdm(sorted(filenames), desc=root):
            if filename.lower().endswith('tif'):
                name = filename.split('.')[0].replace('_1024', '')
                # 1024, 512
                res = osp.split(root)[-1].split('_')[-1]
                root_img_path = osp.join(root, filename)

                vis_img_path = osp.join(root_label_path, 'FGT_TIF', '%s_FGT_%s.tif'%(name, res))
                if osp.exists(vis_img_path):
                    meta_path, context_path = osp.join(root_label_path, 'METADATA', '%s_FGT_META_%s.json'%(name,
                                                                                                           res)), \
                                       osp.join(root_label_path, 'FGT_JSON', '%s_FGT_%s.json'%(name, res))
                else:
                    vis_img_path = osp.join(root_label_path, 'FGT_TIF', '%s_FGT.tif'%(name))
                    meta_path, context_path = osp.join(root_label_path, 'METADATA', '%s_FGT_META.json'%(name)), \
                                           osp.join(root_label_path, 'FGT_JSON', '%s_FGT.json'%(name))

                try:
                    meta_context = jl.load_json(meta_path, encoding = 'EUC-KR', replace_pair=[('"region"', ',"region"'),
                                                                                              ('"korft_description"',
                                                                                               ',"korft_description"')])
                except:
                    try:
                        meta_context = jl.load_json(meta_path, encoding = None)
                    except:
                        meta_context = jl.load_json(meta_path, encoding = 'utf-16le', replace_pair=[('"region"', ',"region"'),
                                                                                                  ('"korft_description"',
                                                                                                   ','
                                                                                                   '"korft_description"')], skip= 1)


                context = jl.load_json(context_path, encoding = None, replace_pair=[("null", '"활엽수"')])

                attributes = jl.get_objects([meta_context, context])
                # print(context.features[0].geometry.coordinates)
                # print(attributes)

                # img
                img = cv2.imread(root_img_path)
                cv2.imwrite(osp.join(img_path, '%s.jpg'%name), img)

                # mask
                mask = jl.draw_mask(img, attributes, color_dict = get_color_dict())
                cv2.imwrite(osp.join(mask_path, '%s.png'%name), mask)

                vis = cv2.imread(vis_img_path)
                vis = cv2.hconcat([img, mask, vis])
                cv2.imwrite(osp.join(vis_path, '%s.jpg'%name), vis)

                #  xml
                if len(attributes['bboxes']) > 0:
                    writer = Writer(attributes['filename'], attributes['width'], attributes['height'], database = attributes['filename'])
                    writer.addBboxes(attributes['bboxes'], attributes['category_name'])
                    writer.save(osp.join(xml_path, '%s.xml'%name))

                # json
                save_json(osp.join(json_path, '%s.json'%name), attributes)


    # path = '/Users/sober/Downloads/Project/2022_pwd/temp'

    # names = set()
    # for root, _, filenames in os.walk(path):
    #     for filename in tqdm(filenames, desc = root):
    #         if filename.endswith('json'):
    #             context = jl.load_json(osp.join(root, filename), encoding = None)
    #             attributes = jl.get_objects([None, context])
    #             for name in attributes['name']:
    #                 names.add(name)
    #     # break
    # pprint.pprint(names)



