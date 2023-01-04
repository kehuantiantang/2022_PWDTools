# coding=utf-8
# @Project  ：2022_PWDTools 
# @FileName ：main.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2022/11/6 2:42 下午
import os
import pprint
import urllib
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp

from TSNE.utils import pkl2xml, obtain_features
from misc.pascal_voc_utils import Reader
from misc.utils import load_pkl, save_pkl


def extract_features_byXML(feature_dir, xml_dir, target_path, name2index = [('tp|wb',0), ('tp|wg', 1), ('tp|yellow' ,2),
                                                                            ('tp|maple',3), ('fp|wb', 4),
                                                                            ('fp|wg',5), ('fp|yellow',6), ('fp|maple',7)]):
    x, y = [], []
    name2index = {k:v for k, v in name2index}
    for root, _, filenames in os.walk(xml_dir):
        for filename in tqdm(filenames):
            if filename.endswith('xml'):
                name = filename.split('.')[0]
                objs = Reader(osp.join(root, filename)).get_objects()
                img_path = osp.join(feature_dir, '%s.pkl' % name)

                if osp.exists(img_path):
                    img = load_pkl(img_path)

                    for cls_name, bbox in zip(objs['name'], objs['bboxes']):
                        if cls_name in name2index.keys():
                            xmin, ymin, xmax, ymax = bbox
                            if xmin < xmax and ymin < ymax and xmax - xmin > 32 and ymax - ymin > 32:
                                #['0', '1', '2', '3']
                                feature_dict = obtain_features(img, xmin, ymin, xmax, ymax)
                                feature = np.concatenate([feature_dict[i] for i in ['0', '1', '2', '3']])

                                x.append(feature)
                                y.append(name2index[cls_name.strip()])
                else:
                    pass
                    # print(img_path)

    save_pkl(target_path, {'x':np.array(x), 'y':np.array(y)})


def extract_features_byPKL(feature_root, pkl_path, target_path, tp_fp_threshold = [0.5, 0.3], draw_category =
['disease'], filter_area = 64 * 64, filter_ratio = 2):
    '''
    :param feature_root:
    :param pkl_path:
    :param target_path:
    :param tp_fp_threshold:
    :param draw_category:
    :param filter_area: bounding box pixel size small than filter_area will ignore
    :param filter_ratio: [1: inf] bounding box large than filter_ratio will ignore
    :return:
    '''
    def keep_bbox(xmin, ymin, xmax, ymax):
        height, width = ymax - ymin, xmax - xmin
        if height * width < filter_area:
            return False

        if max(height, width) * 1.0 / min(height, width) > filter_ratio :
            return False

        return True

    context = load_pkl(pkl_path)

    print(target_path, '='*20)

    img_dict = {}
    for root, _, filenames in os.walk(feature_root):
        for filename in filenames:
            if filename.endswith('pkl'):
                img_dict[filename.split('.')[0]] = osp.join(root, filename)

    os.makedirs(target_path, exist_ok=True)

    counter = defaultdict(int)

    # context, name --> class --> fp, fpr, tp, gt
    extract_features = defaultdict(list)
    for name, clss in tqdm(context.items()):
        if name in img_dict.keys():
            p = img_dict[name]
            if osp.exists(p):
                img = load_pkl(p)

                for cls, lines in clss.items():
                    fp = lines['fp']
                    tp = lines['tp']
                    gt = lines['gt']


                    for f in fp:
                        # only the detected threshold large than threshold
                        score, xmin, ymin, xmax, ymax =  f
                        if score >= tp_fp_threshold[1] and keep_bbox(xmin, ymin, xmax, ymax):
                            if cls in draw_category:
                                c = 'red'
                                extract_features['fp'].append((name, obtain_features(img, xmin, ymin, xmax, ymax)))
                            else:
                                c = 'sliver'
                            counter[cls + '_fpDraw'] += 1
                        counter[cls + '_fpAll'] += 1


                    for t in tp:
                        score, xmin, ymin, xmax, ymax =  t
                        if score >= tp_fp_threshold[0] and keep_bbox(xmin, ymin, xmax, ymax):
                            if cls in draw_category:
                                c = 'green'
                                extract_features['tp'].append((name, obtain_features(img, xmin, ymin, xmax, ymax)))
                            else:
                                c = 'yellow'
                            counter[cls[0] + '_tpDraw'] += 1
                        counter[cls[0] + '_tpAll'] += 1

                    for g in gt:
                        if cls in draw_category:
                            pass
                        counter[cls[0] + '_gtDraw'] += 1

    pprint.pprint(counter)
    save_pkl(osp.join(target_path, 'extract_features_name.pkl'), extract_features)

    def to_sklearn_array(extract_features):
        tps, fps = extract_features['tp'], extract_features['fp']
        x, y = [], []
        for tp in tps:
            x.append(np.concatenate([tp[1][i] for i in ['0', '1', '2', '3']]))
            y.append(1)
        for fp in fps:
            x.append(np.concatenate([fp[1][i] for i in ['0', '1', '2', '3']]))
            y.append(0)
        # fp is zero, tp is one
        save_pkl(osp.join(target_path, 'extract_features_sklearn.pkl'), {'x':np.array(x), 'y':np.array(y)})

    to_sklearn_array(extract_features)
    return counter


def draw_tsne(pkl_path = '/home/khtt/code/detector2FPN/output/company/20220929/resnet101_extract_features_tp'
                         '/extract_features_sklearn.pkl', index2name = [(0, 'FP'), (1, 'TP')],
target_path='/home/khtt/code/detector2FPN/output/company/20220929/resnet101_extract_features_tp/tsne.png'):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    data = load_pkl(pkl_path)
    x_tsne = tsne.fit_transform(data['x'])
    x_tsne_data = np.vstack((x_tsne.T, data['y'])).T
    df_tsne = pd.DataFrame(x_tsne_data, columns=['dim1', 'dim2', 'class'])
    for index, name in index2name:
        df_tsne['class'][df_tsne['class'] == index] = name

    df_tsne.head()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    import seaborn as sns
    sns.scatterplot(data=df_tsne, hue='class', x='dim1', y='dim2',s=18)
    plt.show()
    plt.savefig(target_path,
                dpi = 300)


def pkl2voc():
    path = '/Users/sober/Workspace/Python/2022_PWDTools/TSNE/fp_tp_trainData.pkl'
    # target_root = '/Volumes/Share/PWD2022_TSNE/pwd2022_test_train_xml'
    target_root = '/Users/sober/Downloads/tp'
    pkl_content = load_pkl(path)
    pkl2xml(pkl_content, target_root, 0.2)
    # print(pkl_content)



if __name__ == '__main__':
    # xml_root = '/Users/sober/Workspace/Python/2022_PWDTools/TSNE/wrong_code'
    # for root, _, filenames in os.walk(xml_root):
    #     for filename in filenames:
    #         if filename.endswith('.xml'):
    #             # utf_8_name = filename.encode('utf-8')
    #         #     new_name = urllib.quo
    #         #     (utf_8_name )
    #         #     print(new_name)
    #             print(filename.encode('Windows-1252'))

    folder = 'feature_pkl_dir'
    os.makedirs(folder, exist_ok=True)

    name2index_all = [('tp|wb',0), ('tp|wg', 1), ('tp|yellow' ,2),
                      ('tp|maple',3), ('fp|wb', 4),
    ('fp|wg',5), ('fp|yellow',6), ('fp|maple',7)]

    name2index_tp = [('tp|wb',0), ('tp|wg',1), ('tp|yellow',2),
                                 ('tp|maple',3)]
    name2index_fp = [('fp|wb',0),
    ('fp|wg',1), ('fp|yellow',2), ('fp|maple',3)]

    name2index_wb = [('tp|wb',0), ('fp|wb',1)]
    name2index_wg = [('tp|wg',0), ('fp|wg',1)]
    name2index_yellow = [('tp|yellow',0), ('fp|yellow',1)]
    name2index_maple = [('tp|maple',0), ('fp|maple',1)]

    name2index_4cls = [('tp|wb',0), ('tp|wg', 1), ('tp|yellow' ,2),
                       ('tp|maple',3), ('fp|wb', 0),
                       ('fp|wg',1), ('fp|yellow',2), ('fp|maple',3)]

    name2index_2cls = [('tp|wb',0), ('tp|wg', 0), ('tp|yellow' ,0),
                       ('tp|maple',0), ('fp|wb', 1),
                       ('fp|wg',1), ('fp|yellow',1), ('fp|maple',1)]

    name2index_yellowAndMaple = [('tp|yellow',0), ('fp|yellow',1), ('tp|maple',2), ('fp|maple',3)]


    # cls_features = {'all_6cls':name2index_all, 'tp_3cls':name2index_tp, 'fp_3cls':name2index_fp, 'wb_2cls':name2index_wb, 'wg_2cls':name2index_wg,
    #                 'yellow_2cls':name2index_yellow, 'maple_2cls':name2index_maple}

    # cls_features = {'tp_3cls':name2index_tp, 'fp_3cls':name2index_fp, 'wb_2cls':name2index_wb, 'wg_2cls':name2index_wg,
    #                 'yellow_2cls':name2index_yellow, 'maple_2cls':name2index_maple}


    # cls_features = {'yellow_2cls':name2index_yellow}
    # cls_features = {'maple_2cls':name2index_maple}
    # cls_features = {'all_6cls':name2index_all}
    # cls_features = {'fp_3cls':name2index_fp}
    # cls_features = {'tp_3cls':name2index_tp}

    # cls_features = {'maple_2cls':name2index_maple}

    cls_features = {'mergeTPFP_4cls':name2index_4cls}
    # cls_features = {'mergeTPFP_2cls':name2index_2cls}
    # cls_features = {'yellowAndMaple':name2index_yellowAndMaple}


    for name, name2index in cls_features.items():
        print(name, '*'*50)
        feature_dir, xml_dir = '/dataset/khtt/dataset/pwd_analyse/resnet101_extract_features_all/features', './annotation'
        extract_features_byXML(feature_dir, xml_dir, osp.join(folder, '%s.pkl'%name), name2index = name2index)

        # draw_tsne(pkl_path = osp.join(folder, '%s.pkl'%name), index2name =[(v, k)for k, v in name2index], \
        #           target_path=osp.join(folder, '%s.png'%name))

    # for name, name2index in cls_features.items():
        draw_tsne(pkl_path = osp.join(folder, '%s.pkl'%name), index2name =[(0, 'wb'), (1, 'wg'), (2, 'yellow'), (3, 'maple')],
                  target_path=osp.join(folder, '%s.png'%name))






