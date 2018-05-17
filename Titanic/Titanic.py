import pandas as pd
import numpy as np
import warnings
import re
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cross_validation import KFold

warnings.filterwarnings('ignore')

test = pd.read_csv("Data/test.csv")
train = pd.read_csv("Data/train.csv")

full_data = [test, train]

# 计算名字的长度
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

# 判断是否有船舱
train['Has_Cabin'] = train['Cabin'].apply(lambda x:0 if type(x) == float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

# 计算每个人在船上的家人数
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# 判断是否单独一人
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# 用票价的中值填充空的票价,并对票价进行分类
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# qcut根据频率来分组,cut根据值来分组

# 对年龄进行分类
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(('Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'), 'Rare', )

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16 & (dataset['Age'] <= 32)), 'Age'] =1
    dataset.loc[(dataset['Age'] > 32 & (dataset['Age'] <= 48)), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48 & (dataset['Age'] <= 64)), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis =1)
test = test.drop(drop_elements, axis=1)


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
ax = sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)

label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=90, horizontalalignment='right')

g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
                        u'FamilySize', u'Title']], hue='Survived', palette='seismic', size=1.2, diag_kind='kde',
                 diag_kws=dict(shade=True), plot_kws=dict(s=10))
g.set(xticklabels=[])

plt.show()


ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0

NFLODS = 5
kf = KFold(ntrain, n_folds=NFold, random_state=SEED)

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)


    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)


    def predict(self, x):
        return self.clf.predict(x)


    def fit(self, x):
        return self.clf.fit(x)


    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).frature_importances)


    def get_oof(clf, x_train, y_train, x_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFLODS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)

        return oof_train.reshape(-1, 1), oof_test.reshape(-1,1)
