#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing  
def feature_selection_and_sort_by_chromosome(data, annotation_path, preprocessed_file_path='data'):
    big_table = pd.read_csv(data)
    feature_name = list(big_table)
    feature_name = feature_name[1:]
    labels = np.array(big_table.iloc[:, 0])
    annotation = pd.read_csv(annotation_path, dtype=str)
    gene_id_annotation = list(annotation.loc[:, "gene"])
    feature_name =[feature.split('|')[0] for feature in feature_name]
    idx = []
    print('features have been sorted based on chromosome')
    for gene_id in gene_id_annotation:
        if gene_id in feature_name:
            idx.append(feature_name.index(gene_id))
    feature_name=np.array(feature_name)[idx]
    features_raw = np.array(big_table.iloc[:, 1:])
    features = np.log2(1.0 + features_raw)
    features[np.where(features <= 1)] = 0
    # numpy is different from lis
    features = features[:, idx]
    print('remove the features that  Variance is low than threshold') 
    selector = VarianceThreshold(threshold=1)
    selector.fit(features)
    idx2 = selector.get_support(indices=True)
    features = features[:, idx2]
    feature_name=np.array(feature_name)[idx2]

    print(features.shape,len(feature_name))
    print('normalise the data in [0,1])')
    max_abs_scaler = preprocessing.MaxAbsScaler()
    features = max_abs_scaler.fit_transform(features)
    feature_name_path = os.path.join(preprocessed_file_path, 'feature_name.csv')
    features = np.concatenate((labels.reshape(-1,1),features),axis=1)
    feature_name = np.concatenate((np.array(['label']),feature_name))
    print(features.shape,feature_name.shape)
    pd.DataFrame(features,columns=feature_name).to_csv(feature_name_path,index=0)
    print('features are selected, the selected preprocessing data are saved at', feature_name_path)
    return features, labels


if __name__ == '__main__':
    feature_selection_and_sort_by_chromosome('data/big_gene_expression_data.csv','data/Annotation.csv')










