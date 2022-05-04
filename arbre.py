
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

# 

# In[2]:



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

# 

# In[3]:


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


# In[4]:



# In[ ]:





# In[5]:


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


# In[6]:


len(timing)


# In[7]:



# In[8]:


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


# In[9]:


# box 0 always starts


# In[10]:


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


# In[11]:


# In[12]:


#plt.figure(figsize=(12,6))
up=(timing['ciuchStart']-timing['ciuchStartup']).apply(lambda x:x.total_seconds())
mid=(timing['ciuchStop']-timing['ciuchStart']).apply(lambda x:x.total_seconds())
down=(timing['ciuchStopdown']-timing['ciuchStop']).apply(lambda x:x.total_seconds())
#plt.boxplot([up,mid,down],labels=['ciuchStartup > ciuchStart','ciuchStart > ciuchStop','ciuchStop > ciuchStopdown'])
#plt.grid()
#plt.title('durations: Startup>Start, Start>Stop, Stop>Stopdown',size=16)


# In[13]:


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
    up=pd.DataFrame(index=pd.date_range(start=ciuchStartup, end=ciuchStart,periods=steps,closed='left')).reset_index(drop=False).rename(columns={'index':'slice'})
    up.index=['up_'+str(x) for x in range(steps-1)]
    slices=slices.append(up)
#     
    mid=pd.DataFrame(index=pd.date_range(start=ciuchStart, end=ciuchStop,periods=steps,closed='left')).reset_index(drop=False).rename(columns={'index':'slice'})
    mid.index=['mid_'+str(x) for x in range(steps-1)]
    slices=slices.append(mid)
#     
    down=pd.DataFrame(index=pd.date_range(start=ciuchStop, end=ciuchStopdown,periods=steps,closed='left')).reset_index(drop=False).rename(columns={'index':'slice'})
    down.index=['down_'+str(x) for x in range(steps-1)]
    slices=slices.append(down)
#     slices=slices.append(up)
slices=slices.reset_index(drop=False).rename(columns={'index':'slice_id'})
# 
timing_slices=pd.merge_asof(slices,timing,left_on='slice',right_on='ciuchStartup',direction='backward')
timing_slices=timing_slices[['run', 'refListId', 'refListId_last','slice_id','slice','ciuchStartup', 'ciuchStart', 'ciuchStop', 'ciuchStopdown','t0_run']]
timing_slices.head()


# In[14]:


# In[15]:


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



# df_timing_slices=pd.merge(df_timing_slices, reflist, on='Epc',how='left')
# df_timing_slices = df_timing_slices [ ~((df_timing_slices['refListId']==0) & (df_timing_slices['refListId_actual']==9)) ]
# # 
# df_timing_slices = df_timing_slices [ ~((df_timing_slices['refListId']==9) & (df_timing_slices['refListId_actual']==0)) ]
# # # 
# # df_timing_slices = df_timing_slices [ ~((df_timing_slices['refListId']==0) | (df_timing_slices['refListId_actual']==0)) ]

# df_timing_slices=df_timing_slices.drop(['refListId_actual','Q refListId_actual'],axis=1)


# In[18]:


runs_out=df_timing_slices.groupby('run')['refListId'].nunique().rename('Q refListId').reset_index(drop=False)

# In[19]:


current_last_windows=timing_slices.drop_duplicates(['run','refListId','refListId_last'])
current_last_windows=current_last_windows[['run','refListId','refListId_last','ciuchStop']].reset_index(drop=True)

# In[20]:


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

# In[23]:


# df_timing_slices['dt']=
df_timing_slices['dt']=(df_timing_slices['LogTime']-df_timing_slices['t0_run']).apply(lambda x:x.total_seconds())


# In[24]:

# In[25]:


# 
# df_timing_threshold
# 


# In[26]:


rssi_threshold=-110
df_timing_slices_threshold=df_timing_slices[df_timing_slices['Rssi']>rssi_threshold]


# In[27]:


# readrate
# readrate
#round(100*df_timing_slices_threshold.reset_index(drop=False).groupby(['run','loc'])['Epc'].nunique().groupby('loc').mean()    /reflist['Epc'].nunique(),2)


# # MON TRAVAIL, LES METHODES DE MACHINE LEARNING #

# In[28]:


#import seaborn as sb


# In[29]:


df2=df_timing_slices


# **Tout d'abord il nous faut quelques données dont on connait l'emplacement de la boite et des epc afin de réaliser nos opérations de machine learning sur cette nouvelle dataset**

# ## 1) On va modifier la méthode analytique pour construire un dataset avec les classes prédéfinies 

# ### 1.1) Réécriture de la méthode analytique

# In[30]:


df_1 =df_timing_slices
df_1['EPC'] = df_1['Epc'].map(str) + '-' + df_1['run'].map(str) 
df_1['EPC'].values.tolist()



# In[31]:


def zert (x) :
    
    df_epc=df_timing_slices[df_timing_slices['EPC']== x ]
    df_epc_in= df_epc[df_epc['loc']== 'in']
    df_epc_out= df_epc[df_epc['loc']== 'out']
    #return df_epc
    if (df_epc_in.shape[0] > df_epc_out.shape[0]) :
        return 0
    elif (df_epc_in.shape[0] < df_epc_out.shape[0]) :
        return 0
    else :
        return 1
    #return df5_in.shape[0] > df5_out.shape[0]
    
    
#print(zert('epc_0-1'))


# In[32]:


zerts = []
epcs = df_1['EPC'].unique()
df_x = pd.DataFrame()
for i in epcs : 
    zerts = zerts + [zert(i)]
zert(df_1.iloc[1,15])


# In[33]:


# ### 1.2 ) on suprime les lignes où l'epc apparait deux fois dans le même run

# df_1.drop_duplicates( subset ="EPC", keep = 'first', inplace=True)

# In[ ]:





# ### 1.3) On peut apliquer la méthode sur chacune des lignes du nouveau dataset

# In[34]:


# df_1.loc[3,'loc_epc']

# In[35]:


compteur= 0
for i in range(36318) :   
    df_1.loc[i,'loc_epc']=zert(df_1.iloc[i,15])


# In[36]:


# In[37]:


df_1.drop_duplicates(keep = 'first', inplace=True)


# In[38]:

# ## 2) Essayons de déterminer des dépendances pour utiliser les méthodes de machine learning les plus pertinantes

# ### 2.1) En réalisant une étude graphique

# In[39]:


#sb.pairplot(data=df_1)


# ### 2.2) En réalisant une étude à l'aide d'une matrice de corrélation(nous avons maintenant la  colonne loc_epc qui représente la position de l'epc)

# In[40]:


#import seaborn as sb

corr = df_1.corr()
#corrMat = plt.matshow(corr, fignum = 2)
#plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
#plt.yticks(range(len(corr.columns)), corr.columns)
#sb.heatmap(corr,annot=True)
#plt.show()


# ## 3) Application des modèles ##

# In[41]:

# In[42]:


# In[43]:


df_tree_class=df_1.drop(columns=["Epc","EPC","loc","ciuchStartup","LogTime","ciuchStart","ciuchStop","ciuchStopdown","slice_id","t0_run","slice"])


# In[44]:


X = df_tree_class.iloc[:,0:5] #les caractéristiques
y = df_tree_class.iloc[:, 5]  #les résulats (classes)


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)


#   

# **Fonction permettant de trouver les paramètres les plus performants**

# In[46]:


from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import GridSearchCV

'''
def dtree_grid_search(X, y, nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
    # decision tree model
    dtree_model=DecisionTreeClassifier()
    #use gridsearch to test all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X, y)
    return dtree_gscv.best_params_
'''

# In[47]:


#dtree_grid_search(X,y,5)


# In[48]:


tree_clf=DecisionTreeClassifier(max_depth=6)
tree_clf.fit(X,y)


# In[49]:

fn=[ 'run','refListId', 'refListId_last', 'Rssi','dt','loc_epc']
cn=['0', '1']
#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=200)
#tree.plot_tree(tree_clf, feature_names = fn, class_names=cn,filled = True);
#fig.savefig('imagename.png')


# **Evaluation du modèle(score)**

# In[50]:


#print(tree_clf.score(X_test, y_test))


# **Matrice de confusion**

# In[51]:


y_pred = tree_clf.predict(X_test)


# In[52]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# **On créé un dictionnaire pour l'affichage sur l'interface**

# In[53]:

mon_dictionnaire = {"score":tree_clf.score(X_test, y_test),
                    "TN":confusion_matrix(y_test,y_pred)[0][0],
                    "FP":confusion_matrix(y_test,y_pred)[0][1],
                    "FN":confusion_matrix(y_test,y_pred)[1][0],
                    "TP":confusion_matrix(y_test,y_pred)[1][1],
                    "testNum": sys.argv[1]}


# In[54]:

print(mon_dictionnaire)


# In[ ]:





# In[ ]:




