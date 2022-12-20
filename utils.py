
import pandas as pd
import numpy as np
import concurrent.futures as cf
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV



def change_type(df,n=0):
    col=df.columns[n]
    
    if df[col].dtype=="O":
            try:
               df=df.astype({col:"float64"})
               
            except:
                df=df.astype({col:"str"})
                
                 
            
    if n==len(df.columns)-1:
        return df
    else:
        n+=1
        return change_type(df,n)

def clean_col(df):
    
    df["S_2"]=df["S_2"].astype('datetime64[ns]')
    df["Year"]=df["S_2"].dt.year
    df["Month"]=df["S_2"].dt.month

    return df

def reage_list(long,n=0,new_lis1=[],new_lis2=[]):
    
    new_lis1=list(map(lambda x: [long[n],x],long))

    new_lis2=new_lis2+new_lis1

    if n==len(long)-1:

        new_lis2=list(filter(lambda x:x  if x[0]!=x[1] else None,new_lis2))

        return new_lis2

    else:

        n+=1

        return reage_list(long,n,new_lis1,new_lis2)

def organise_df(data,inter,n=0):
    now=inter[n]
    try: 
    
        data[now[0][1]]=pd.concat([data[now[0][1]],data[now[0][0]][data[now[0][0]]["customer_ID"]==now[1][0]]])
    
        data[now[0][0]]=data[now[0][0]].drop(data[now[0][0]][data[now[0][0]]["customer_ID"]==now[1][0]].index)
        
    except:
        pass

    if n==len(inter)-1:
        return data
    else:
        n+=1
        return organise_df(data,inter,n)
            
def auto_sys(csv):

    chunks=pd.read_csv(csv,chunksize=500000)

    with cf.ThreadPoolExecutor() as executor:
        data=list(executor.map(lambda x:x , chunks))

    arr=np.arange(len(data)).tolist()
    
    arr_lis=reage_list(arr)

    with cf.ThreadPoolExecutor() as executor:
        inter=list(executor.map(lambda x:[x, np.intersect1d(data[x[0]]["customer_ID"].unique(), (data[x[1]]["customer_ID"].unique())).tolist()] ,arr_lis))
    

    data=organise_df(data,inter)



    with cf.ThreadPoolExecutor() as executor:
        data=list(executor.map(clean_col,data))
    

    
    return data

def save_csv(lista_df,n=0,lista_link=[]):
    
    link="train_num"+str(n)+".csv"

    df=lista_df[n]
    
    df.to_csv(link)

    lista_link.append(link)
    
    if n==len(lista_df)-1:
        
        return lista_link

    else:

        n+=1

        return save_csv(lista_df,n,lista_link)
    
def map_data(data,gr_data,num_data,n=0,dic={}):

    

    array_ID=gr_data[gr_data["S_2"]==num_data[n]]["customer_ID"].unique()

    new_frame=data[data["customer_ID"].isin(array_ID)]

    dic.update({num_data[n]:new_frame})

    if n==len(num_data)-1:
        return dic
    else:
        n+=1
        return map_data(data,gr_data,num_data,n,dic)

def dic_maker(list_link):

    data=pd.read_csv(list_link)
    
    gr_data=data.groupby("customer_ID")["S_2"].count().to_frame().reset_index()

    num_data=list(gr_data["S_2"].unique())

    max_val=max(num_data)
    
    num_data.pop(num_data.index(max_val))
    
    n=0
    
    dic={}
    
    dic=map_data(data,gr_data,num_data,n,dic)
    
    array_ID=gr_data[gr_data["S_2"]==max_val]["customer_ID"].unique()
    
    data=data[data["customer_ID"].isin(array_ID)]
    
    dic.update({max_val:data})

    del data

    return dic

def del_dic(data,num_key,num):

    del data[num][num_key]

    if num==len(data)-1:
        return data
    else:
        num+=1
        return del_dic(data,num_key,num)


def concat_dic(data,arr_key):

    num_key=arr_key[0]

    data[0][num_key]=pd.concat(list(map(lambda x : x[num_key] , data)))

    num=1
    
    data=del_dic(data,num_key,num)
    
    arr_key.pop(0)

    if len(arr_key)==0:
        return data

    else:
        return concat_dic(data,arr_key)

def read_dic(data,data_keys,n=0):

    df=data[data_keys[0]]

    if len(data_keys)==1:
        n+=1
        link="train_month"+str(data_keys[0])+"_"+str(n)+".csv"

    else:
        link="train_month"+str(data_keys[0])+".csv"
    
    df.to_csv(link)

    data_keys.pop(0)


    if len(data_keys)==0:
        
        del data

        return 

    else:

        return read_dic(data,data_keys,n)
    

def save_me(data,n=0):
    df=data[n][13]

    link="train_month13_"+str(n)+".csv"

    df.to_csv(link)

    if n==len(data)-1:
        return

    else:
        n+=1
        return save_me(data,n)
    
def change_col(link):
    
    data_1=pd.read_csv(link)
    
    data_1=change_type(data_1)
    
    lista_col=list(filter(lambda x: x if (data_1[x].isnull().sum()/len(data_1[x]))*100>=75 else None,data_1.columns))

    data_1.drop(labels=lista_col,axis=1,inplace=True)

    data_1.drop(labels=['Unnamed: 0.1', 'Unnamed: 0'],axis=1,inplace=True)

    data_1.to_csv(link)
    
    return


def prep_data(data_X):

    categorical=list(data_X.select_dtypes("object").columns)
    numerical=list(data_X.select_dtypes("number").columns)

    cat_pipe=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
    ("encoder",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)),
    ("scaler",StandardScaler())
    ])

    num_pipe=Pipeline([
    ("imputer",KNNImputer(n_neighbors=5,weights="uniform",missing_values=np.nan)),
    ("scaler",StandardScaler())
    ])

    preprocess=ColumnTransformer([("cat",cat_pipe,categorical), 
    ("num",num_pipe,numerical)])
    preprocess.fit(data_X)
    data_X=preprocess.transform(data_X)

    return data_X,preprocess


def specificity(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity
    
def equ_train(X_data,Y_data,equ,dis):
    
    data=pd.merge(X_data,Y_data,how="inner",on="customer_ID")
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    y=y.to_frame()
    customer_ID=data.iloc[:,0]

  
    y_1=y[y["target"]==dis]
    y_0=y[y["target"]==equ]

    y_1_half=y_1.iloc[:int(len(y_0)/2)]
    y_0_half=y_0.iloc[:int(len(y_0)/2)]
    y_train=pd.concat([y_0_half,y_1_half])
    y_1_half=y_1.iloc[int(len(y_0)/2):]
    y_0_half=y_0.iloc[int(len(y_0)/2):]
    y_test=pd.concat([y_0_half,y_1_half])
    X_train=X[X.index.isin(y_train.index)]
    X_test=X[X.index.isin(y_test.index)]

    return X_train,X_test,y_train,y_test


def get_last(X,ID):
    return X[X["customer_ID"]==ID].sort_values("S_2").iloc[-1,:]

def cat_col(data):
    categorical=list(data.select_dtypes("object").columns)
    numerical=list(data.select_dtypes("number").columns)
    return categorical,numerical

def test_model_reg_log(eval,data,y):
    
    model_reglog={
    
    "penalty":["l2","l1","elasticnet"],
    'C': np.arange(0.01, 0.1, 0.01),
    "random_state":[42],
    "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "multi_class":["auto","ovr","multinomial"],
    "max_iter":[1000],
    "n_jobs":[-1]
}



    reg_log = GridSearchCV(LogisticRegression(),
                    model_reglog,
                    cv = 5,
                    n_jobs=-1, scoring=eval)

    reg_log.fit(data, y.values.reshape(1,-1)[0])

    return [reg_log.best_estimator_,reg_log.best_score_]

def test_model_rfc(eval,data,y):
    
        model_rfc={
        
        "n_estimators":[100,300,500,700,1000],
        'criterion': ["gini","entropy","log_loss"],
        "random_state":[42],
        "max_features":["auto","sqrt","log2"],
        "class_weight":[None,"balance"]
    }



        rfc = GridSearchCV(RandomForestClassifier(),
                    model_rfc,
                    cv = 5,
                    n_jobs=-1, scoring="recall")

        rfc.fit(data, y.values.reshape(1,-1)[0])

        return [rfc.best_estimator_,rfc.best_score_]

def test_model_svc(eval,data,y):
    
    model_dtc={
    
    "C":[0.1,0.5,1,1.5,2,5,10],
    "kernel":["linear","rbf","sigmoid","poly"],
    "gamma":["scale","auto"],
    "shrinking":[True,False],
    "probability":[True,False],
    "random_state":[42],
    "class_weight":[None,"balance"],
    "max_iter":[-1]
}



    svc = GridSearchCV(SVC(),
                   model_dtc,
                   cv = 5,
                   n_jobs=-1, scoring=eval)

    svc.fit(data, y.values.reshape(1,-1)[0])

    return [svc.best_estimator_,svc.best_score_]


def test_model_knn(eval,data,y):
    
    model_dtc={
    
    "n_neighbors":[1,3,5,10,20],
    "weights":["uniform","distance"],
    "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
    "leaf_size":[10,20,30,40,50,100],
    "p":[1,2],
    "n_jobs":[-1]
}



    knn = GridSearchCV( KNeighborsClassifier(),
                   model_dtc,
                   cv = 5,
                   n_jobs=-1, scoring=eval)

    knn.fit(data, y.values.reshape(1,-1)[0])

    return [knn.best_estimator_,knn.best_score_]

def full_test(data,y,eval,eval_name,dic={}):

    knn=test_model_knn(eval[0],data,y)
    reg_log=test_model_reg_log(eval[0],data,y)
    rfc=test_model_rfc(eval[0],data,y)
    
    dic.update({eval_name[0]:[knn,reg_log,rfc]})
    eval.pop(0)
    eval_name.pop(0)
    if len(eval)==0:
        return dic
    else:
        return full_test(data,y,eval,eval_name,dic)