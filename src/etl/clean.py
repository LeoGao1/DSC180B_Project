import pandas as pd

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import multiprocessing.popen_spawn_win32
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import preprocessing

def data_cleaning (data_path, output_path,feature_num = 100):
    #start a local dask cluster
    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)

    #read data
    sys_info = dd.read_csv(data_path,
                           delimiter ="\1",
                           assume_missing=True)
    print('read data successfully')

    #find used columns
    used_cols =['chassistype',
                'chassistype_2in1_category',
                'countryname_normalized',
                'modelvendor_normalized',
                'model_normalized',
                'ram',
                'os',
                '#ofcores',
                'age_category',
                'graphicsmanuf',
                'graphicscardclass',
                'processornumber',
                'cpuvendor',
                'cpu_family',
                'cpu_suffix',
                'screensize_category',
                'persona',
                'processor_line',
                'vpro_enabled',
                'discretegraphics']
    df = sys_info[used_cols]

    #cleaning
    df = df.dropna()
    df = df[df.persona!= 'Unknown'].reset_index(drop=True)
    df = df[df.processornumber!= 'Unknown'].reset_index(drop=True)
    df = df.compute()

    df['processornumber'] = df['processornumber'].apply(lambda x: x[:2] ).astype('int32',errors='raise')
    df['ram'] =df['ram'].astype('int32')
    df['#ofcores'] =df['#ofcores'].astype('int32',errors='raise')

    #define the columns with different type
    used_cols.remove('persona')
    int_cols = ['ram','#ofcores','processornumber']
    cat_cols = [i for i in used_cols if i not in int_cols]
    print('clean data successfully')

    #one hot encoding on cat_cols
    df = pd.get_dummies(df, columns =cat_cols).reset_index(drop=True)
    #get the x and y
    y = df['persona'].values
    temp = list(df.columns.values)
    temp.remove('persona')
    x = df[temp].values

    #apply label encoder on persona
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    #apply PCA on y
    pca = PCA(n_components=feature_num)
    x = pca.fit_transform(x,y)

    print('get features successfully')

    # x_df = pd.DataFrame(x)
    # y_df = pd.DataFrame(y)
    #
    # x_df.to_csv(output_path + 'x_df.csv', index = False)
    # y_df.to_csv(output_path + 'y_df.csv', index = False)


    return x,y
