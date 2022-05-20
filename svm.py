#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

# 

p_C = float(sys.argv[1])
p_kernel = sys.argv[2]
p_degree = int(sys.argv[3])
p_gamma = sys.argv[4]
p_probability = True if sys.argv[5]=="True" else False
p_tol = float(sys.argv[6])
p_max_iter = int(sys.argv[7])
p_random_state = None if sys.argv[8]=="None" else int(sys.argv[8])

pathfile=r'data_anonymous'

# reflist: list of epc in each box
reflist=pd.DataFrame()
# 
files=os.listdir(pathfile)
for file in files:
   
    if file.startswith('reflist_'):
        temp=pd.read_csv(os.path.join(pathfile,file),sep=',').reset_index(drop=True)[['Epc']]
        temp['refListId']=file.split('.')[0]
        reflist=reflist.append(temp)
reflist=reflist.rename(columns={'refListId':'refListId_actual'})
reflist['refListId_actual']=reflist['refListId_actual'].apply(lambda x:int(x[8:]))
Q_refListId_actual=reflist.groupby('refListId_actual')['Epc'].nunique().rename('Q refListId_actual').reset_index(drop=False)
reflist=pd.merge(reflist,Q_refListId_actual,on='refListId_actual',how='left')
reflist.head()


# 

# In[4]:


# pathfile=r'data_anonymous'
# 
# df : rfid readings
df=pd.DataFrame()
# 
files=os.listdir(pathfile)
for file in files:
    
    if file.startswith('ano_APTags'):
        temp=pd.read_csv(os.path.join(pathfile,file),sep=',')
        df=df.append(temp)
df['LogTime']=pd.to_datetime (df['LogTime'] ,format='%Y-%m-%d-%H:%M:%S') 
df['TimeStamp']=df['TimeStamp'].astype(float)
df['Rssi']=df['Rssi'].astype(float)
df=df.drop(['Reader','EmitPower','Frequency'],axis=1).reset_index(drop=True)
df=df[['LogTime', 'Epc', 'Rssi', 'Ant']]
# antennas 1 and 2 are facing the box when photocell in/out 
Ant_loc=pd.DataFrame({'Ant':[1,2,3,4],'loc':['in','in','out','out']})
df=pd.merge(df,Ant_loc,on=['Ant'])
df=df.sort_values('LogTime').reset_index(drop=True)
df.head()


# In[5]:


len(df)


# In[ ]:





# In[6]:


# timing: photocells a time window for each box: start/stop (ciuchStart, ciuchStop)
file=r'ano_supply-process.2019-11-07-CUT.csv'
timing=pd.read_csv(os.path.join(pathfile,file),sep=',')
timing['file']=file
timing['date']=pd.to_datetime(timing['date'],format='%d/%m/%Y %H:%M:%S,%f')
timing['ciuchStart']=pd.to_datetime(timing['ciuchStart'],format='%d/%m/%Y %H:%M:%S,%f')
timing['ciuchStop']=pd.to_datetime(timing['ciuchStop'],format='%d/%m/%Y %H:%M:%S,%f')
timing['timestampStart']=timing['timestampStart'].astype(float)
timing['timestampStop']=timing['timestampStop'].astype(float)
timing=timing.sort_values('date')
timing.loc[:,'refListId']=timing.loc[:,'refListId'].apply(lambda x:int(x[8:]))
timing=timing[['refListId', 'ciuchStart', 'ciuchStop']]
timing[:1]


# In[7]:


len(timing)


# In[8]:


timing[:12]


# In[9]:


# ciuchStart_up starts upstream ciuchStart, half way in between the previous stop and the actual start
timing[['ciuchStop_last']]=timing[['ciuchStop']].shift(1)
timing[['refListId_last']]=timing[['refListId']].shift(1)
timing['ciuchStartup']=timing['ciuchStart'] - (timing['ciuchStart'] - timing['ciuchStop_last'])/2
# timing start: 10sec before timing
timing.loc[0,'refListId_last']=timing.loc[0,'refListId']
timing.loc[0,'ciuchStartup']=timing.loc[0,'ciuchStart']-datetime.timedelta(seconds=10)
timing.loc[0,'ciuchStop_last']=timing.loc[0,'ciuchStartup']-datetime.timedelta(seconds=10)
timing['refListId_last']=timing['refListId_last'].astype(int)
# 
timing['ciuchStopdown']= timing['ciuchStartup'].shift(-1)
timing.loc[len(timing)-1,'ciuchStopdown']=timing.loc[len(timing)-1,'ciuchStop']+datetime.timedelta(seconds=10)
timing=timing[['refListId', 'refListId_last','ciuchStartup', 'ciuchStart','ciuchStop','ciuchStopdown']]
timing.head()


# In[10]:


# box 0 always starts
timing[timing['refListId']==0].head()


# In[11]:


# t0_run = a new run starts when box 0 shows up
t0_run=timing[timing['refListId']==0] [['ciuchStartup']]
t0_run=t0_run.rename(columns={'ciuchStartup':'t0_run'})
t0_run=t0_run.groupby('t0_run').size().cumsum().rename('run').reset_index(drop=False)
t0_run=t0_run.sort_values('t0_run')
# 
# each row in timing is merged with a last row in t0_run where t0_run (ciuchstart) <= timing (ciuchstart)
timing=pd.merge_asof(timing,t0_run,left_on='ciuchStartup',right_on='t0_run', direction='backward')
timing=timing.sort_values('ciuchStop')
timing=timing[['run', 'refListId', 'refListId_last', 'ciuchStartup','ciuchStart','ciuchStop','ciuchStopdown','t0_run']]
timing.head()


# In[12]:


timing[:12]


# In[13]:


#plt.figure(figsize=(12,6))
up=(timing['ciuchStart']-timing['ciuchStartup']).apply(lambda x:x.total_seconds())
mid=(timing['ciuchStop']-timing['ciuchStart']).apply(lambda x:x.total_seconds())
down=(timing['ciuchStopdown']-timing['ciuchStop']).apply(lambda x:x.total_seconds())
#plt.boxplot([up,mid,down],labels=['ciuchStartup > ciuchStart','ciuchStart > ciuchStop','ciuchStop > ciuchStopdown'])
#plt.grid()
#plt.title('durations: Startup>Start, Start>Stop, Stop>Stopdown',size=16)
#plt.show()


# In[14]:


#  full window (ciuchStartup > ciuchStopdown) is sliced in smaller slices
# ciuchStartup > ciuchStart: 11 slices named up_0, up_1, ..., up_10
# ciuchStart > ciuchStop: 11 slices named mid_0, mid_1, ... mid_10
# ciuchStop > ciuchStopdown: 11 slices names down_0, down_1, ... down_10
slices=pd.DataFrame()
for i, row in timing .iterrows():
    ciuchStartup=row['ciuchStartup']
    ciuchStart=row['ciuchStart']
    ciuchStop=row['ciuchStop']
    ciuchStopdown=row['ciuchStopdown']
    steps=4
#     
    up=pd.DataFrame(index=pd.date_range(start=ciuchStartup, end=ciuchStart,periods=steps,closed='left'))        .reset_index(drop=False).rename(columns={'index':'slice'})
    up.index=['up_'+str(x) for x in range(steps-1)]
    slices=slices.append(up)
#     
    mid=pd.DataFrame(index=pd.date_range(start=ciuchStart, end=ciuchStop,periods=steps,closed='left'))        .reset_index(drop=False).rename(columns={'index':'slice'})
    mid.index=['mid_'+str(x) for x in range(steps-1)]
    slices=slices.append(mid)
#     
    down=pd.DataFrame(index=pd.date_range(start=ciuchStop, end=ciuchStopdown,periods=steps,closed='left'))        .reset_index(drop=False).rename(columns={'index':'slice'})
    down.index=['down_'+str(x) for x in range(steps-1)]
    slices=slices.append(down)
#     slices=slices.append(up)
slices=slices.reset_index(drop=False).rename(columns={'index':'slice_id'})
# 
timing_slices=pd.merge_asof(slices,timing,left_on='slice',right_on='ciuchStartup',direction='backward')
timing_slices=timing_slices[['run', 'refListId', 'refListId_last','slice_id','slice',                               'ciuchStartup', 'ciuchStart', 'ciuchStop', 'ciuchStopdown','t0_run']]
timing_slices.head()


# In[15]:


len(timing_slices)


# In[16]:


# merge between df and timing
# merge_asof needs sorted df > df_ref
df=df[ (df['LogTime']>=timing['ciuchStartup'].min()) & (df['LogTime']<=timing['ciuchStopdown'].max())  ]
df=df.sort_values('LogTime')
# 
# each row in df_ref is merged with the last row in timing where timing (ciuchstart_up) < df_ref (logtime)
# 
# df_timing=pd.merge_asof(df_ref,timing,left_on=['LogTime'],right_on=['ciuchStartup'],direction='backward')
# df_timing=df_timing.dropna()
# df_timing=df_timing.sort_values('LogTime').reset_index(drop=True)
# df_timing=df_timing[['run', 'Epc','refListId', 'refListId_last', 'ciuchStartup',\
#                      'LogTime', 'ciuchStop', 'ciuchStopdown','Rssi', 'loc', 'refListId_actual']]
# 
# each row in df_ref is merged with the last row in timing_slices where timing (slice) < df_ref (logtime)
# 
df_timing_slices=pd.merge_asof(df,timing_slices,left_on=['LogTime'],right_on=['slice'],direction='backward')
df_timing_slices=df_timing_slices.dropna()
df_timing_slices=df_timing_slices.sort_values('slice').reset_index(drop=True)
df_timing_slices=df_timing_slices[['run', 'Epc','refListId', 'refListId_last', 'ciuchStartup','slice_id','slice','LogTime',                       'ciuchStart','ciuchStop', 'ciuchStopdown', 'Rssi', 'loc','t0_run']]


# In[17]:


# 


# In[18]:


# df_timing_slices=pd.merge(df_timing_slices, reflist, on='Epc',how='left')
# df_timing_slices = df_timing_slices [ ~((df_timing_slices['refListId']==0) & (df_timing_slices['refListId_actual']==9)) ]
# # 
# df_timing_slices = df_timing_slices [ ~((df_timing_slices['refListId']==9) & (df_timing_slices['refListId_actual']==0)) ]
# # # 
# # df_timing_slices = df_timing_slices [ ~((df_timing_slices['refListId']==0) | (df_timing_slices['refListId_actual']==0)) ]

# df_timing_slices=df_timing_slices.drop(['refListId_actual','Q refListId_actual'],axis=1)


# In[19]:


runs_out=df_timing_slices .groupby('run')['refListId'].nunique().rename('Q refListId').reset_index(drop=False)
runs_out[runs_out['Q refListId']!=10]


# In[20]:


current_last_windows=timing_slices.drop_duplicates(['run','refListId','refListId_last'])
current_last_windows=current_last_windows[['run','refListId','refListId_last','ciuchStop']].reset_index(drop=True)
current_last_windows[:1]


# In[21]:


# runs 16 23 32 40 have missing boxes: discarded
# also run 1 is the start, no previous box: discarded
# run 18: box 0 run at the end
# 
timing=timing[~timing['run'].isin([1,18,16,23,32,40])]
timing_slices=timing_slices[~timing_slices['run'].isin([1,18,16,23,32,40])]
df_timing_slices=df_timing_slices[~df_timing_slices['run'].isin([1,18,16,23,32,40])]

df_timing_slices=df_timing_slices.sort_values(['LogTime','Epc'])
# 


# In[22]:


len(timing),len(timing_slices), len(df_timing_slices)


# In[23]:


df_timing_slices[:1]


# In[24]:


# df_timing_slices['dt']=
df_timing_slices['dt']=(df_timing_slices['LogTime']-df_timing_slices['t0_run']).apply(lambda x:x.total_seconds())


# In[25]:


df_timing_slices[:1]


# In[26]:


# 
# df_timing_threshold
# 


# In[27]:


rssi_threshold=-110
df_timing_slices_threshold=df_timing_slices[df_timing_slices['Rssi']>rssi_threshold]


# In[28]:


# readrate
# readrate
round(100*df_timing_slices_threshold.reset_index(drop=False).groupby(['run','loc'])['Epc'].nunique().groupby('loc').mean()    /reflist['Epc'].nunique(),2)

df_timing_slices[df_timing_slices['Epc']=='epc_100']
# # 1 - LA METHODE ANALYTIQUE #

# In[29]:


import seaborn as sb


# In[30]:


df2=df_timing_slices


# **Tout d'abord il nous faut quelques données dont on connait l'emplacement de la boite et des epc afin de réaliser nos opérations de machine learning sur cette nouvelle dataset**

# ## 1) On va modifier la méthode analytique pour construire un dataset avec les classes prédéfinies 

# ### 1.1) Réécriture de la méthode analytique

# In[31]:



df_timing_slices['EPC_run'] = df_timing_slices['Epc'].map(str) + '-' + df_timing_slices['run'].map(str) 
df_timing_slices['EPC_run'].values.tolist()
df_1 =df_timing_slices.copy()
df_1_ml=df_timing_slices.copy()
#print(zert('epc_0-85'))
#df_1['run'].value_counts().sort_values()


# In[32]:


def PossibleBoxes(EPC) :
    df_EPC_mid= df_timing_slices[(df_timing_slices['EPC_run']== EPC) & ((df_timing_slices['slice_id']=='mid_0') | (df_timing_slices['slice_id']=='mid_1') | (df_timing_slices['slice_id']=='mid_2') )]
    df_EPC= df_timing_slices[(df_timing_slices['EPC_run']== EPC)]
    if(len(df_EPC_mid)>0):
        df_percent = df_EPC_mid
    else:
        df_percent = df_EPC
    df_percent = 100*( df_percent.refListId.value_counts() /  df_percent.refListId.count())

    return df_percent


# In[34]:


PossibleBoxes('epc_68-39')


# In[35]:


def BoxbyAnaticalMethod(EPC) : 
    return int(PossibleBoxes(EPC).index[0])
        #if(zert(EPC)!=1) :
            #return None
        #else :


# In[36]:


BoxbyAnaticalMethod('epc_0-39')


# ### 1.2 ) on supprime les lignes où l'epc apparait deux fois dans le même run

# In[37]:


df_1.drop_duplicates( subset ="EPC_run", keep = 'first', inplace=True)


# ### 1.3) On peut apliquer la méthode sur chacune des lignes du nouveau dataset

# In[38]:


df_1


# ### Application de la méthode analytique sur le dataset complet 

# In[39]:


df_1['Box_prediction']=[BoxbyAnaticalMethod(EPC) for EPC in df_1['EPC_run'].values]
#df_1['Box_prediction'].value_counts()
#[BoxbyAnaticalMethod(EPC) for EPC in df_1['EPC_run'].values]


# ### Comparaison des boîtes prédites avec celle du dataframe reflist

# In[40]:


df_1=df_1.merge(reflist, on = ['Epc'])
df_1


# ### Pourcentage de bonnes et mauvaises prédictions

# In[41]:


df_1['GOOD_Prediction']=df_1['Box_prediction']==df_1['refListId_actual']
dfbox_percent = 100*( df_1.GOOD_Prediction.value_counts() /  df_1.GOOD_Prediction.count())
dfbox_percent


# ### Création du dataframe pour récupérer les mauvaises prédictions

# In[42]:


df_false =df_1.copy()


# In[43]:


df_false['BAD_Prediction']=df_false['Box_prediction']!=df_false['refListId_actual']
dfbox2_percent = 100*( df_false.BAD_Prediction.value_counts() /  df_false.BAD_Prediction.count())
dfbox2_percent


# In[44]:


df_false1=df_false[df_false['BAD_Prediction'] == True]
#df_false1


# # 2- MON TRAVAIL, LES METHODES DE MACHINE LEARNING #

# ### 2.1 )Création du dataframe pour récupérer les bonnes prédictions
df_true1=df_false[df_false['BAD_Prediction'] == False]
df_true1
# #### 2.2) On se place sur une seule fenêtre

# In[45]:


df_1_ml=df_timing_slices.copy()


# In[46]:


#df_1_ml=df_1_ml[df_1_ml.slice_id.isin(['mid_0', 'mid_1', 'mid_2'])]
df_1_ml=df_1_ml[df_1_ml.slice_id.isin(['mid_1', 'mid_2'])]


# In[47]:


df_1_ml["slice_id"]="mid"


# In[48]:


df_1_ml['run_window'] = df_1_ml['run'].map(str) + '_' + df_1_ml['slice_id'].map(str) 
#df_1_ml['run_window'].values.tolist()


# In[49]:


df_1_ml=df_1_ml.merge(reflist, on = ['Epc'])


# In[50]:


df_1_ml


#    

# **On peut maintenant créer une colonne classe nous permettant de savoir si le tag réalisé correspond à un epc dont la boite se trouve dans la zone in(classe 1) ou non (classe 0)** 

# In[51]:


df_1_ml['classe']=0
for i in range(len(df_1_ml)-1):
    if (df_1_ml.loc[i,'refListId']==df_1_ml.loc[i,'refListId_actual']):
        df_1_ml.loc[i,'classe']=1
    else :
        df_1_ml.loc[i,'classe']=0  
df_1_ml


#   

# ## 3) Application des modèles ##

# In[52]:


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[53]:


#svm_clf = Pipeline ( [ ( "scaler" , StandardScaler ( ) ) ,( "linear_svc" ,
#            LinearSVC(C=p_C,)) ] )
svm_clf = SVC(C=p_C,kernel=p_kernel,degree=p_degree,gamma=p_gamma,
              probability=p_probability,tol=p_tol,max_iter=p_max_iter,
              random_state=p_random_state)


# In[54]:


df4 =df_1_ml[['loc','Q refListId_actual','refListId_actual','Rssi','run','refListId','refListId_last','classe']]


# In[55]:


df4


# In[56]:


df_rdf=df4.copy()


# In[57]:


df_rdf['loc']=df_rdf['loc'].map({'in':1,'out':0})


# In[58]:


df_rdf


# In[59]:


X_train,X_test,y_train,y_test=train_test_split(df_rdf.drop(['classe'],axis=1),df_rdf['classe'])


# In[60]:


svm_clf.fit(X_train, y_train )


# **Validation croisée à 3 plis, performance moyenne**
# 

# In[61]:


from sklearn.model_selection import cross_val_score
cross_val_score( svm_clf , X_train , y_train , cv=3,scoring="accuracy" )


# **Matrice de confusion**

# In[62]:


y_pred = svm_clf.predict(X_test)


# In[63]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[64]:


from sklearn.metrics import f1_score


# In[65]:


mon_dictionnaire = {"score":f1_score(y_test,y_pred),"TN":confusion_matrix(y_test,y_pred)[0][0],"FP":confusion_matrix(y_test,y_pred)[0][1],"FN":confusion_matrix(y_test,y_pred)[1][0],"TP":confusion_matrix(y_test,y_pred)[1][1]}


# In[66]:


print(mon_dictionnaire)

