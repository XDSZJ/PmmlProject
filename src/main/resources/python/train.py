import pandas as pd
import sklearn2pmml
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn_pandas import DataFrameMapper

titanic = pd.read_csv(r"F:\myfile\python\machine_learning_practice\泰坦尼克号成员获救情况预测\1.实战代码\train.csv")


# 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    # y即目标年龄
    y = known_age[:, 0]
    # X即特征属性值
    X = known_age[:, 1:]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df


titanic = set_missing_ages(titanic)

dummies_Embarked = pd.get_dummies(titanic['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(titanic['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(titanic['Pclass'], prefix='Pclass')

df = pd.concat([titanic, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# print(df)

# 将数据的Label分离出来
train_label = df['Survived']
train_titanic = df.drop('Survived', 1)

# 对数据进行模型预测
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = model_selection.KFold(n_splits=3, shuffle=False, random_state=None)
scores = model_selection.cross_val_score(alg, train_titanic, train_label, cv=kf)

print(scores.mean())

# 导入测试集的数据，并将数据和测试集上的数据进行一样的处理
titanic_test = pd.read_csv(r"F:\myfile\python\machine_learning_practice\泰坦尼克号成员获救情况预测\1.实战代码\test.csv")
titanic_test = set_missing_ages(titanic_test)
dummies_Embarked = pd.get_dummies(titanic_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(titanic_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(titanic_test['Pclass'], prefix='Pclass')
df_test = pd.concat([titanic_test, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# 先用训练集上的数据训练处一个模型，再在测试集上进行预测，并将结果输出到一个csv文件中
model = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
# model.fit(train_titanic, train_label)
# joblib.dump(model, "test.plk")
# predictions = model.predict(df_test)
# result = pd.DataFrame(
#     {'PassengerId': titanic_test['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
# result.to_csv("random_forest_predictions.csv", index=False)
# print(pd.read_csv("random_forest_predictions.csv"))

mapping = DataFrameMapper([
    ()
])

pipeline = sklearn2pmml.PMMLPipeline([("classifier", model)])
pipeline.fit(train_titanic, train_label)
sklearn2pmml.sklearn2pmml(pipeline, "titanic.pmml", with_repr=True)
