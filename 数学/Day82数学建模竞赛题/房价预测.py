import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = np.log1p(train['SalePrice'])
X_train = train.drop(['Id','SalePrice'], axis=1)
X_test = test.drop(['Id'], axis=1)
ids  = test['Id']

print(f"训练集: {X_train.shape},测试集{X_test.shape}")

all_data = pd.concat([X_train, X_test], axis=0)

num_cols = all_data.select_dtypes(include=[numpy.number]).columns
all_data[num_cols] = all_data[num_cols].fillna(all_data[num_cols].median())

cat_cols = all_data.select_dtypes(exclude=['object']).columns
all_data[cat_cols] = all_data[cat_cols].fillna('None')

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5*all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5*all_data['BsmtHalfBath']
all_data['Age'] = all_data['YearBuilt']
all_data['Remodeled'] = (all_data['YearRemodAdd']!=all_data['YearBuilt']).astype(int)

for col in cat_cols:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col]).astype(str)

X_train_proceessed = all_data[:len(X_train)]
X_test_proceessed = all_data[len(X_train):]

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train_proceessed)
X_test_scaled = scaler.transform(X_test_proceessed)

from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

models = {
    'Ridge': Ridge(alpha=10),
    'Lasso': Lasso(alpha=0.001),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.7),
    'GBR': GradientBoostingRegressor(n_estimators=1000,learning_rate=0.05,max_depth=4,random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth=5,random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=1000,learning_rate=0.05,max_depth=5,random_state=42),
    'CatBoost': CatBoostRegressor(n_estimators=1000,learning_rate=0.05,max_depth=6,verbose=0,random_state=42)
}

def rmsle_cv(model,X,y):
    cv_scores = cross_val_score(model, X, y, cv=5,scoring='neg_mean_squared_error')
    return np.sqrt(-cv_scores)

results = {}
for name,model in models.items():
    scores = rmsle_cv(model,X_train_scaled,y_train)
    results[name] = scores
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

from sklearn.ensemble import StackingRegressor

estimators = [
    ('ridge', Ridge(alpha=10)),
    ('lasso', Lasso(alpha=0.001)),
    ('gbr',GradientBoostingRegressor(n_estimators=1000,learning_rate=0.05,max_depth=4)),
    ('xgb',xgb.XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth=5)),
]

stacking_reg = StackingRegressor(estimators=estimators,
                                 final_estimator=Ridge(alpha=10),
                                 cv=5,
                                 n_jobs=-1)

stacking_reg.fit(X_train_scaled,y_train)

y_pred_log = stacking_reg.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)

submission = pd.DataFrame({'Id':ids,'SalePrice':y_pred})
submission.to_csv('submission.csv',index=False)
print("提交文件生成: submission.csv")