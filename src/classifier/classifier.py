import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import multiprocessing.popen_spawn_win32
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification


def build_classifier(data_path):
    #start a local dask cluster
    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)

    knn_param = list(range(2,10))
    dt_param = list(range(5,10))
    rf_param = list(range(5,10))

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
    pca = PCA(n_components=100)
    x = pca.fit_transform(x,y)

    #train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    print('Start to build the classifier and tune the parameter')

    result = pd.DataFrame(columns =['classifier','parameter','train_score','test_score'])

    #KNN
    print("Start KNN")
    #tune the parameter
    for i in knn_param:

        clf1 = KNeighborsClassifier(n_neighbors=i)
        clf1.fit(x_train, y_train)

        #get train score
        y_train_pred = clf1.predict(x_train)
        train_sc = accuracy_score(y_train, y_train_pred)

        #get test score
        y_test_pred = clf1.predict(x_test)
        test_sc = accuracy_score(y_test, y_test_pred)

        temp = {'classifier':'KNN',
                'parameter': i,
                'train_score':train_sc,
                'test_score':test_sc}
        print(temp)
        result = result.append(temp,ignore_index=True)


    #DecisionTree
    print("Start Decision Tree")
    for i in dt_param:
        clf2 = DecisionTreeClassifier(max_depth= i)
        clf2.fit(x_train, y_train)

        #get train score
        y_train_pred = clf2.predict(x_train)
        train_sc = accuracy_score(y_train, y_train_pred)

        #get test score
        y_test_pred = clf2.predict(x_test)
        test_sc = accuracy_score(y_test, y_test_pred)

        temp = {'classifier':'Decision Tree',
                'parameter': i,
                'train_score':train_sc,
                'test_score':test_sc}
        print(temp)
        result = result.append(temp,ignore_index=True)

    #Random Forest
    print("Start Random Forest")
    for i in rf_param:
        clf3 = RandomForestClassifier(max_depth =i)
        clf3.fit(x_train, y_train)

        #get train score
        y_train_pred = clf3.predict(x_train)
        train_sc = accuracy_score(y_train, y_train_pred)

        #get test score
        y_test_pred = clf3.predict(x_test)
        test_sc = accuracy_score(y_test, y_test_pred)

        temp = {'classifier':'Random Forest',
                'parameter': i,
                'train_score':train_sc,
                'test_score':test_sc}
        print(temp)
        result = result.append(temp,ignore_index=True)

    print("All Done!")
    return result
