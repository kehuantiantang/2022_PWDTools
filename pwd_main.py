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
        for filename in filenames:
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


if __name__ == '__main__':
    # data path, has *.json, *.jpg file
    # root = '/dataset/khtt/dataset/pine2022/ECOM/7.evaluations/fasterrcnn_deploy_split_test_a_20221211_222419/resnet101/Fold_0'
    root = '/dataset/khtt/dataset/pine2022/ECOM/7.evaluations/yolov5_split_test_a_20221216_054424'
    # target path to save
    # target_path = '/Users/sober/Downloads/Project/2022_pwd/20220720_r'
    target_path = '/dataset/khtt/dataset/pine2022/ECOM/tp3'

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
        jpg_path = path + '.jpg'
        # gt_path = path + '_gt.jpg'
        json_path = path + '.json'


        jpg_img = cv2.imread(jpg_path)
        # gt_img = cv2.imread(gt_path)


        context = jl.load_json(json_path)
        attributes = jl.get_objects(context, json_shape_type='rectangle')

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
        # jpg_img_polygons = jl.draw_polygons(jpg_img, attributes)
        # draw image
        # jpg_mask = jl.draw_mask(jpg_img, attributes, color_dict= (1, 1, 1), single_channel= True)

        # if nb_disease == 0:
        cv2.imwrite(osp.join(img_target, name + '_%02d.jpg'%nb_disease), jpg_img)
        # cv2.imwrite(osp.join(vis_target, name + '_%02d.jpg'%nb_disease), cv2.hconcat([jpg_img_boxes, jpg_img_polygons]))
        cv2.imwrite(osp.join(vis_target, name + '_%02d.jpg'%nb_disease), jpg_img_boxes)
        # cv2.imwrite(osp.join(mask_target, name + '_%02d.png'%nb_disease), jpg_mask)
        save_json(osp.join(json_target, name + '_%02d.json'%nb_disease), objs)


    print('The number of disease: %d, disease image/no disease image/total: %d/%d/%d' % (disease_counter,
                                                                                        dis_img_counter,
                                                                                 no_dis_img_counter, total_img))
    print('Convert json to xml bbox', '.'*50)
    json_to_xml(json_target, xml_target)


# if __name__ == '__main__':
    # self.index2label['01000000'] = 'forest'
    # self.index2label['02000000'] = 'noForest'
    # self.index2label['01110100'] = 'disease_pine-init'
    # self.index2label['01110200'] = 'disease_pine-middle'
    # self.index2label['01110300'] = 'disease_pine-late'
    # self.index2label['01120100'] = 'disease_beauty-init'
    # self.index2label['01120200'] = 'disease_beauty-middle'
    # self.index2label['01120300'] = 'disease_beauty-late'
    # self.index2label['01130100'] = 'disease_japan-init'
    # self.index2label['01130200'] = 'disease_japan-middle'
    # self.index2label['01130300'] = 'disease_japan-late'


    # path = "/Volumes/2022 AI 산림해충 방제 시스템/20220705 탐지학습데이터/labled"
    # path = '/Volumes/SoberSSD/CONTOUR_V3_20221122_145813_R12345_25000_Tag'
    # jl = JsonLoader()
    # counter = 0
    # name_list = []
    # for root, _, filenames in os.walk(path):
    #     for filename in tqdm(sorted(filenames)):
    #         if filename.endswith('.json'):
    #             context = jl.load_json(osp.join(root, filename))
    #             attributes = jl.get_objects(context)
    #
    #             name_list.extend(attributes['name'])
    #             # if 'disease_black-late' in attributes['name']:
    #             #     name = filename.split('.')[0]
    #             #     if counter <= 50:
    #             #         print(name)
    #             #         plt.figure()
    #             #         plt.imshow(cv2.imread(osp.join(root, '%s_vis.jpg')%name)[:, :, ::-1])
    #             #         plt.show()
    #             #         counter += 1
    #             #     else:
    #             #         break
    #             counter += 1
    #
    #             # if counter == 20:
    #             #     break
    #
    #
    # print(counter)
    # print(jl.index2label)
    #
    # pprint.pprint(Counter(name_list), indent=5)





