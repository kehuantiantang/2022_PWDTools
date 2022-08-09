# coding=utf-8
from tqdm import tqdm
import cv2.cv2
from json_polygon import JsonLoader
from pascal_voc_utils import Writer
from util import load_json, namespace2dict, save_json
import os.path as osp
import os
from util import imread, imwrite

def read_dir(path):
    '''
    read the file from directory
    :param path:
    :return:
    '''
    files = {}
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('json'):
                name = filename.split('.')[0]
                files[name] = osp.join(root, name)

    return files

def json_to_xml(path, xml_root):
    '''
    load json file and save to xml bounding box
    :param path:
    :param xml_root:
    :return:
    '''
    os.makedirs(xml_root, exist_ok=True)

    jl = JsonLoader()

    for root, _, filenames in os.walk(path):
        for filename in tqdm(filenames):
            if filename.endswith('json'):
                p = osp.join(root, filename)
                xml_p = osp.join(xml_root, filename.replace('json', 'xml'))

                context = jl.load_json(p)
                objects = jl.get_objects(context)

                if len(objects['bboxes']) > 0:
                    writer = Writer(objects['filename'], objects['width'], objects['height'], database = objects['filename'])
                    writer.addBboxes(objects['bboxes'], objects['category_name'])

                    writer.save(xml_p)

if __name__ == '__main__':
    # data path, has *.json, *.jpg file
    root = r'H:\tp\언양(완료)\output\18\225098'
    # target path to save
    target_path = r'H:\tp\언양(완료)\output\18\225098_r'

    # json include polygon, point
    json_target = osp.join(target_path, 'json')
    # image gis image
    img_target = osp.join(target_path, 'img')
    # segmentation mask
    mask_target = osp.join(target_path, 'mask')
    # check whether the polygon and bbox is correctly annotate
    vis_target = osp.join(target_path, 'vis')
    # bbox xml
    xml_target = osp.join(target_path, 'xml')

    os.makedirs(json_target, exist_ok=True)
    os.makedirs(img_target, exist_ok=True)
    os.makedirs(vis_target, exist_ok=True)
    os.makedirs(mask_target, exist_ok=True)
    os.makedirs(xml_target, exist_ok=True)


    jl = JsonLoader()

    file_path = read_dir(root)
    disease_counter, dis_img_counter, no_dis_img_counter, total_img = 0, 0, 0, 0
    color_dict = {'car': (0, 255, 255), 'tree': (0, 255, 0), 'road':(255, 0, 0), 'building': (0, 0, 255), 'field': (255, 255, 0)}
    for name, path in tqdm(file_path.items()):
        # jpg_path = path + '.jpg'
        jpg_path = path + '.png'
        gt_path = path + '_gt.jpg'
        json_path = path + '.json'



        jpg_img = imread(jpg_path)
        gt_img = imread(gt_path)


        context = jl.load_json(json_path)
        attributes = jl.get_objects(context)

        nb_disease = len(attributes['polygons'])

        disease_counter += nb_disease
        if nb_disease > 0:
            dis_img_counter += 1
        else:
            no_dis_img_counter += 1
        total_img += 1

        objs = namespace2dict(context)

        # draw bbox image
        jpg_img_boxes = jl.draw_bboxes(jpg_img, attributes)
        # draw polygon image
        jpg_img_polygons = jl.draw_polygons(jpg_img, attributes, color_dict = color_dict)
        # draw image
        # jpg_mask = jl.draw_mask(jpg_img, attributes, color_dict= color_dict, single_channel= True)
        jpg_mask = jl.draw_mask(jpg_img, attributes, color_dict=color_dict)

        # if nb_disease == 0:
        imwrite(osp.join(img_target, name + '_%02d.jpg'%nb_disease), jpg_img)
        imwrite(osp.join(vis_target, name + '_%02d.jpg'%nb_disease), cv2.hconcat([jpg_img_boxes, jpg_img_polygons]))
        imwrite(osp.join(mask_target, name + '_%02d.png'%nb_disease), jpg_mask)
        save_json(osp.join(json_target, name + '_%02d.json'%nb_disease), objs)


    print('The number of disease: %d, disease image/no disease image/total: %d/%d/%d' % (disease_counter,
                                                                                        dis_img_counter,
                                                                                 no_dis_img_counter, total_img))
    print('Convert json to xml bbox', '.'*50)
    json_to_xml(json_target, xml_target)








