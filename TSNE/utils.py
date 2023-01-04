# coding=utf-8
# @Project  ：2022_PWDTools 
# @FileName ：misc.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2022/11/7 4:22 下午
import os
import os.path as osp
from tqdm import tqdm
from misc.pascal_voc_utils import Writer, Reader


def pkl2xml(pkl_content, target_root, confi_threshold = 0.0):
    def add_bbox(writer, lines, name):
        for score, xmin, xmax, ymin, ymax in lines:
            if score > confi_threshold:
                writer.addObjectByBox((xmin, xmax, ymin, ymax), '%s|'%name)

    os.makedirs(target_root, exist_ok=True)
    for name, value in tqdm(pkl_content.items()):
        tps, fps = value['disease']['tp'], value['disease']['fp']
        writer = Writer(name, width=800, height=800, depth=3, database='Unknown', segmented=0)
        add_bbox(writer, tps, 'tp')
        add_bbox(writer, fps, 'fp')
        writer.save(osp.join(target_root, '%s.xml'%name))





def rescale_bbox(xmin, ymin, xmax, ymax, level):
    # [200, 100, 50, 25] downsampling ratio
    reduce = {'0': 4.0, '1': 8.0, '2': 16.0, '3': 32.0}
    if not level in ['0', '1', '2', '3']:
        raise ValueError('Level must in %s'%str(['0', '1', '2', '3']))

    c_xmin = int(round(xmin / reduce[level]))
    c_ymin = int(round(ymin / reduce[level]))
    c_xmax = int(round(xmax / reduce[level]))
    c_ymax = int(round(ymax / reduce[level]))

    return c_xmin, c_ymin, c_xmax, c_ymax


def obtain_features(feature, xmin, ymin, xmax, ymax):
    extract_feature = {}
    for level in ['0', '1', '2', '3']:
        c_xmin, c_ymin, c_xmax, c_ymax = rescale_bbox(xmin, ymin, xmax, ymax, level)
        crop_feature = feature[level][:, c_ymin:c_ymax, c_xmin:c_xmax]
        extract_feature[level] = crop_feature.mean(axis=(1,2))
    return extract_feature


def check_xml_name(xml_root):
    from collections import defaultdict
    name = defaultdict(int)
    for root, _, filenames in os.walk(xml_root):
        for filename in filenames:
            if filename.endswith('.xml'):
                objs = Reader(os.path.join(root, filename)).get_objects()
                for cls_name in objs['name']:
                    if cls_name in ['tp|mapel', 'fp|ywllow', 'tp|yeloow', 'wg', 'tg|wg']:
                        print(filename)
                    name[cls_name]+=1
    print(name)


if __name__ == '__main__':
    check_xml_name('/Users/sober/Workspace/Python/2022_PWDTools/TSNE/annotation')