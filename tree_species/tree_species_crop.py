# coding=utf-8
# @Project  ：2022_PWDTools 
# @FileName ：tree_species_crop.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/1/4 16:50

# coding=utf-8
from tqdm import tqdm
import cv2
from misc.utils import save_json, namespace2dict
from preprocessing.json_polygon import JsonLoader
from misc.pascal_voc_utils import Writer
import os.path as osp
import os

def read_dir(path):
    '''
    read the file from directory
    :param path:
    :return:
    '''
    files = {}
    for root, _, filenames in os.walk(path):
        for filename in sorted(filenames):
            if filename.endswith('json'):
                name = filename.split('.')[0]
                files[name] = osp.join(root, name)

    return files

def json_to_xml(source_root, xml_root):
    '''
    load json file and save to xml bounding box
    :param source_root:
    :param xml_root:
    :return:
    '''
    os.makedirs(xml_root, exist_ok=True)

    jl = JsonLoader()

    for root, _, filenames in os.walk(source_root):
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


def crop_img(expand_pixel = 10):
    # data path, has *.json, *.jpg file
    # root = '/dataset/khtt/dataset/pine2022/ECOM/7.evaluations/fasterrcnn_deploy_split_test_a_20221211_222419/resnet101/Fold_0'
    root = '/dataset/khtt/dataset/pine2022/elcom/2.labled/CONTOUR_V3_20221122_145813_R12345_25000_Tag_orderedAll'
    # target path to save
    # target_path = '/Users/sober/Downloads/Project/2022_pwd/20220720_r'
    target_path = '/dataset/khtt/dataset/pine2022/pine_tree_species'

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
    for name, path in tqdm(file_path.items()):
        jpg_path = path + '.tif'
        # gt_path = path + '_gt.jpg'
        json_path = path + '.json'


        jpg_img = cv2.imread(jpg_path)
        try:
            height, width, _ = jpg_img.shape
            context = jl.load_json(json_path)
            attributes = jl.get_objects(context, json_shape_type='rectangle')
        except:
            print('error: ', jpg_path)
            continue
        # gt_img = cv2.imread(gt_path)
        nb_disease = len(attributes['polygons'])

        # disease_counter += nb_disease
        total_img += 1

        if nb_disease > 0:
            dis_img_counter += 1
        else:
            no_dis_img_counter += 1
            # if no disease exit, continue
            continue

        bboxes, category_names = attributes['bboxes'], attributes['category_name']
        for bbox, category_name in zip(bboxes, category_names):

            # such type of category name is not necessary to save
            if category_name in ['disease', 'unknown', 'forest', 'noForest', 'disease_unknown-init', 'disease_unknown-middle', 'disease_unknown-late']:
                continue

            xmin, ymin, xmax, ymax = bbox

            if xmax - xmin < 16 or  ymax - ymin < 16:
                continue


            xmin, ymin, xmax, ymax = max(0, xmin - expand_pixel), max(0, ymin - expand_pixel), min(width, xmax + expand_pixel), min(height, ymax + expand_pixel)
            patch = jpg_img[ymin:ymax, xmin:xmax, :]


            patch_name = '%05d-%s-%s.jpg'%(disease_counter, name, category_name.replace('disease_', ''))


            try:
                path_shape = patch.shape
                patch = cv2.resize(patch, (224, 224))
                cv2.imwrite(osp.join(img_target, patch_name), patch)
                # if path_shape[0] == 0 or path_shape[1] == 0:
                #     print('error: ', xmin, ymin, xmax, ymax, category_name, img_shape)
                #     continue
            except:
                print('error: ', xmin, xmax, ymin, ymax, category_name, (height, width))
                continue

            disease_counter += 1

    print('The number of disease: %d, disease image/no disease image/total: %d/%d/%d' % (disease_counter,
                                                                                         dis_img_counter,
                                                                                         no_dis_img_counter, total_img))
    # print('Convert json to xml bbox', '.'*50)
    # json_to_xml(json_target, xml_target)


def train_val_test_split(path, target_path, train_ratio=0.8, val_ratio=0.1):
    files = []
    for root, _, filenames in os.walk(path):
        for filename in tqdm(sorted(filenames)):
            if filename.endswith('jpg'):
                files.append(filename)

    # the file is strongly overlap, so we cannot randomly shuffle them
    size = len(files)
    train_size = int(size * train_ratio)
    val_size = int(size * val_ratio)

    train_set = files[:train_size]
    val_set = files[train_size:train_size + val_size]
    test_set = files[train_size + val_size:]

    print('train size: %d, val size: %d, test size: %d' % (len(train_set), len(val_set), len(test_set)))


    # write to file
    os.makedirs(target_path, exist_ok=True)
    with open(osp.join(target_path, 'train.txt'), 'w') as f:
        for file in train_set:
            f.write(file + '\n')

    with open(osp.join(target_path, 'val.txt'), 'w') as f:
        for file in val_set:
            f.write(file + '\n')

    with open(osp.join(target_path, 'test.txt'), 'w') as f:
        for file in test_set:
            f.write(file + '\n')


if __name__ == '__main__':
    # crop_img()
    # train_val_test_split('/dataset/khtt/dataset/pine2022/pine_tree_species/img', '/dataset/khtt/dataset/pine2022/pine_tree_species')

    target_path = '/dataset/khtt/dataset/pine2022/pine_tree_species'
    f1 = open(osp.join(target_path, 'val.txt'))
    f2 = open(osp.join(target_path, 'test.txt'))

    f1_lines = f1.readlines()
    f2_lines = f2.readlines()

    with open(osp.join(target_path, 'val+test.txt'), 'w') as f:
        for line in f1_lines:
            f.write(line)

        for line in f2_lines:
            f.write(line)







