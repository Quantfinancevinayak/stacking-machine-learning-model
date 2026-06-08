import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import BayesianGaussianMixture
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,roc_auc_score,confusion_matrix,precision_recall_curve,classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

vix_data=pd.read_csv('main_df.csv')
vix_data.rename(columns={'Unnamed: 0':'Date'},inplace=True)
vix_data.set_index('Date',drop=True,inplace=True)

for col in vix_data.columns:
    if col != '^INDIAVIX':
        vix_data[col] = vix_data[col].shift(1)

vix_data.dropna(inplace=True)

bgm=BayesianGaussianMixture(n_components=3,random_state=42)
bgm.fit(vix_data[['^INDIAVIX']])
vix_data['regime'] = bgm.predict(vix_data[['^INDIAVIX']])
vix_data['regime_prob'] = bgm.predict_proba(vix_data[['^INDIAVIX']]).max(axis=1)

vix_data['regime'].value_counts()

split=int(len(vix_data)*0.80)
traindata=vix_data.iloc[:split]
testdata=vix_data.iloc[split:]

X_train=traindata.drop(columns=['^INDIAVIX','regime'])
Y_train=traindata['regime']
X_test=testdata.drop(columns=['^INDIAVIX','regime'])
Y_test=testdata['regime']

tsv=TimeSeriesSplit(n_splits=5)

xg=XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42
)
param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

randomized=RandomizedSearchCV(
    xg,param_dist,
    n_iter=30,
    cv=tsv,
    scoring='f1_macro',
    random_state=42

)

Pip=Pipeline([
    ('scale',StandardScaler()),
    ('randomized',randomized)
])

Pip.fit(X_train,Y_train)

best_model=randomized.best_estimator_
Y_pred=Pip.predict(X_test)
print(classification_report(Y_test,Y_pred))