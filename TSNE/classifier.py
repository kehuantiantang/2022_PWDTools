# coding=utf-8
# @Project  ：2022_PWDTools 
# @FileName ：classifier.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2022/11/9 3:39 下午
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from misc.utils import load_pkl
import os.path as osp
import numpy as np
if __name__ == '__main__':
    pkl_dir = '/home/khtt/code/2022_PWDTools/TSNE/feature_pkl_dir'
    pkl_content = load_pkl(osp.join(pkl_dir, 'mergeTPFP_4cls.pkl'))
    X_train, X_test, y_train, y_test = train_test_split(pkl_content['x'], pkl_content['y'], stratify=pkl_content[
        'y'],                       random_state=0, test_size = 0.2)

    clf = MLPClassifier(random_state=2000, max_iter=2000, hidden_layer_sizes = (200, 100, 50), batch_size = 400,
                        verbose = True, n_iter_no_change = 80, tol=1e-6).fit(X_train,
                                                                                                         y_train)
    print(clf.score(X_test, y_test))
    y_pred = clf.predict_proba(X_test)

    print(classification_report(y_test, np.argmax(y_pred, -1), target_names=['wb', 'wg', 'yellow', 'maple']))


    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(y_pred)
    x_tsne_data = np.vstack((x_tsne.T, y_test)).T
    import pandas as pd
    df_tsne = pd.DataFrame(x_tsne_data, columns=['dim1', 'dim2', 'class'])
    for index, name in [(0, 'wb'), (1, 'wg'), (2, 'yellow'), (3, 'maple')]:
        df_tsne['class'][df_tsne['class'] == index] = name

    df_tsne.head()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    import seaborn as sns
    sns.scatterplot(data=df_tsne, hue='class', x='dim1', y='dim2',s=18)
    plt.show()



