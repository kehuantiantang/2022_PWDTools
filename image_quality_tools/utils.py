# coding=utf-8
# @Project  ：2022_PWDTools 
# @FileName ：utils.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2022/11/9 9:54 上午
import json
import os
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import stats
import os.path as osp
import pandas as pd

def correlation_coefficient(y_true, y_pred):
    sq = np.reshape(np.asarray(y_true), (-1,))
    # sq_std = np.reshape(np.asarray(self._y_std), (-1,))
    q = np.reshape(np.asarray(y_pred), (-1,))

    srocc = stats.spearmanr(sq, q)[0]
    krocc = stats.kendalltau(sq, q)[0]
    plcc = stats.pearsonr(sq, q)[0]
    rmse = np.sqrt(((sq - q) ** 2).mean())
    mae = np.abs((sq - q)).mean()
    # outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

    # return srocc, krocc, plcc, rmse, mae, outlier_ratio
    # return srocc, krocc, plcc , rmse, mae
    return srocc, krocc, plcc

def analyse_scoring(score_dir = '/Users/sober/Workspace/Python/2022_PWDTools/image_quality_tools/annotation',
    nb_people = 5):
    json_contents = defaultdict(list)
    label_name = []
    for root, _, filenames in os.walk(score_dir):
        for id_person, filename in enumerate(sorted(filenames)):
            counter = []
            if filename.endswith('.bk'):
                label_name.append(filename.split('.')[-2])
                print(id_person, filename, '*'*10)
                with open(osp.join(root, filename), encoding='utf-8') as f:
                    content = json.load(f)
                    for k, v in content['assessment'].items():
                        json_contents[k].append(int(v))
                        counter.append(int(v))
                print(id_person, Counter(counter))
    # print(json_contents)

    mean_json_contents = {k:np.mean(v) for k, v in json_contents.items()}
    # print(mean_json_contents)
    vote_json_contents = {k:Counter(v).most_common(1)[0][0] for k, v in json_contents.items()}
    media_json_contents = {k:np.median(v) for k, v in json_contents.items()}
    mean_norm_json_contents = {k: (np.array(v)/4.0).mean() for k, v in json_contents.items()}
    # mean_norm_json_contents = [(k, (np.array(v)/4.0).mean()) for k, v in json_contents.items()]



    merge_json_contents = {'mean':mean_json_contents, 'vote':vote_json_contents, 'media':media_json_contents,
                           'mean_norm': mean_norm_json_contents}

    for name, content in merge_json_contents.items():
        print(name, '-'*50)
        y_pred, y_true = [[] for _ in range(nb_people)], [[] for _ in range(nb_people)]
        for k, v in content.items():
            for i in range(nb_people):
                y_pred[i].append(json_contents[k][i])
                y_true[i].append(v)
        result_summary= []
        for person, i in zip(label_name, range(nb_people)):
            print('Person %s'%i, person, '*'*10)
            results = correlation_coefficient(y_true[i], y_pred[i])
            print('Result: ', results)
            result_summary.append(results)


        print(np.array(result_summary).mean(axis=0))
        print(np.array(result_summary).mean(axis=0).mean())


    return merge_json_contents

if __name__ == '__main__':
    # test = {"base_zl_20_tx_899729_ty_409703__from_2S_244_258(\uc7ac\ucd2c\uc601)": 2,
    # "base_zl_20_tx_899729_ty_409704__from_2S_244_258(\uc7ac\ucd2c\uc601)": 2,
    # "base_zl_20_tx_899729_ty_409708__from_2S_244_258(\uc7ac\ucd2c\uc601)": 2,
    # "base_zl_20_tx_899729_ty_409714__from_2S_244_258(\uc7ac\ucd2c\uc601)": 3,
    # "base_zl_20_tx_899729_ty_409723__from_2S_244_258(\uc7ac\ucd2c\uc601)": 2,
    # "base_zl_20_tx_899730_ty_409704__from_2S_244_258(\uc7ac\ucd2c\uc601)": 2,}
    #
    # for k, v in test.items():
    #     # result = re.sub('\\\\x[a-f|0-9]+','',k)
    #     # print(result)
    #     print(k)
    #
    # for root, _, filenames in os.walk('/Users/sober/Workspace/Python/2022_PWDTools/TSNE/wrong_code'):
    #     for filename in filenames:
    #         a = filename.replace('#U', '\\u')
    #         print(a)
    #         print(filename.encode('utf-8'))



    merge_json_contents = analyse_scoring()
    # import pandas as pd
    # df = pd.DataFrame(np.array([('%s.tif'%k, v) for k, v in merge_json_contents['mean_norm'].items()]), columns = [
    #     'image',
    #                                                                                                     'mose'])
    # df.to_csv('pwd_image_labeled_by_score.csv', index = False)



