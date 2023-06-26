# 导入所需要的库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import roc_curve # 绘制ROC曲线
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,f1_score,recall_score,precision_score
import matplotlib.pyplot as plt
%matplotlib inline
# 数据加载
train = pd.read_csv('./train(1).csv')
train
test = pd.read_csv('./test(1).csv')
test
# 训练集、测试集合并
df =pd.concat([train, test], axis=0)
df
# df

df.info()
# df描述
df.describe()
df.isnull().sum()# 缺失值分析
# 唯一值个数
for col in df.columns:
    print(col, df[col].nunique())

cat_columns = df.select_dtypes(include='O').columns
df[cat_columns]

# 切分数据集
train = df[df['subscribe'].notnull()]
test = df[df['subscribe'].isnull()]

cat_columns1 = train.select_dtypes(include='O').columns
train[cat_columns1]

train[cat_columns1]
train['subscribe'] = train['subscribe'].map({'no': 0, 'yes': 1})
train['subscribe'].value_counts()

train[cat_columns1]


def barplot(x,y, **kwargs):
    sns.barplot(x=x , y=y)
    x = plt.xticks(rotation=45)
f = pd.melt(train,  value_vars=cat_columns1 ,id_vars = 'subscribe')
g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(barplot,"value",'subscribe')

g = sns.FacetGrid(train, col='marital',size=6)
g.map(sns.barplot, 'default', 'subscribe', 'education')
g.add_legend()

cat_columns2 = train.select_dtypes(exclude='O').columns
train[cat_columns2]

f = pd.melt(train,  value_vars=cat_columns2 ,id_vars = 'subscribe')
g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, size=5,hue='subscribe')
g = g.map(sns.distplot,"value",bins=20)
g.add_legend()

# 条形图 统计subscribe 0和1的数量
plt.bar(train["subscribe"].value_counts().index, train["subscribe"].value_counts())
plt.xticks(train["subscribe"].value_counts().index)
plt.title("subscribe 0 and 1")
plt.xlabel("subscribe")
plt.ylabel("count")
plt.savefig("bar.png")
plt.show()
# 由于数据分布不平衡，可以使用上采样解决样本不均衡问题


#数据的分布情况
categorical_features = ['duration','campaign','pdays','previous']
f = pd.melt(train,  value_vars=categorical_features,id_vars=['subscribe'])
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(sns.boxplot, "value")

import matplotlib.pyplot as plt
import seaborn as sns
bins=[0,143,353,1873,5149]
df1=df[df['subscribe']=='yes']
binning=pd.cut(df1['duration'],bins,right=False)
time=pd.value_counts(binning)
# 可视化
time=time.sort_index()
fig=plt.figure(figsize=(6,2),dpi=120)
sns.barplot(time.index,time,color='royalblue')
x=np.arange(len(time))
y=time.values
for x_loc,jobs in zip(x,y):
    plt.text(x_loc, jobs+2, '{:.1f}%'.format(jobs/sum(time)*100), ha='center', va= 'bottom',fontsize=8)
plt.xticks(fontsize=8)
plt.yticks([])
plt.ylabel('')
plt.title('duration_yes',size=8)
sns.despine(left=True)
plt.show()

# 特征间的相关性分析
dt_corr = train.corr(method='pearson')

print('相关性矩阵为：\n',dt_corr)

import warnings
# 绘制热力图
import seaborn as sns
plt.subplots(figsize=(18, 18)) # 设置画面大小
sns.heatmap(dt_corr, annot=True, vmax=1, square=True, cmap='Blues')
plt.show()
plt.close

from sklearn.preprocessing import LabelEncoder
Nu_feature=list(df.select_dtypes(exclude=['object']).columns)
Ca_feature=list(df.select_dtypes(include=['object']).columns)
Ca_feature.remove('subscribe')
lb=LabelEncoder()
cols=Ca_feature
for m in cols:
    train[m]=lb.fit_transform(train[m])
    test[m]=lb.fit_transform(test[m])
train['subscribe']=train['subscribe'].replace(['no','yes'],[0,1])

X=train.drop(columns=['id','subscribe'])
Y=train['subscribe']
test=test.drop(columns='id')
# 划分训练及测试集
x_train,x_test,y_train,y_test = train_test_split( X, Y,test_size=0.3,random_state=1)

#对样本过采样
from imblearn.over_sampling import SMOTE
x_train,y_train= SMOTE().fit_resample(x_train,y_train)
y_train.value_counts()