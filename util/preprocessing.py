#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
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
    x_train, x_test, y_train,y_test = train_test_split( features,labels, test_size=.1, random_state=0)
    scaler = normalise_and_save(x_train,feature_name,y_train ,file_path='data/train.csv')
    normalise_and_save(x_test,feature_name,y_test,scaler, 'data/test.csv')
    

    
def normalise_and_save(features,feature_name,labels,scaler=None,file_path='train.csv'):
    print(features.shape,len(feature_name))
    print('normalise the data in [0,1])')
    if scaler is None:
        scaler = preprocessing.MinMaxScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    feature_name_path = os.path.join(file_path)
    features = np.concatenate((labels.reshape(-1,1),features),axis=1)
    feature_name = np.concatenate((np.array(['label']),feature_name))
    print(features.shape,feature_name.shape)
    pd.DataFrame(features,columns=feature_name).to_csv(feature_name_path,index=0)
    print('features are selected, the selected preprocessing data are saved at', feature_name_path)
    return scaler


if __name__ == '__main__':
    feature_selection_and_sort_by_chromosome('data/big_gene_expression_data.csv','data/Annotation.csv')










