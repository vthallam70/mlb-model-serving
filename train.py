"""Train, evaluate and track a binary classification model."""
import argparse,csv,json,os
import joblib,mlflow,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
p=argparse.ArgumentParser(); p.add_argument('csv'); p.add_argument('model'); a=p.parse_args()
with open(a.csv,newline='') as f: rows=list(csv.DictReader(f))
if not rows: raise ValueError('training file is empty')
names=[k for k in rows[0] if k!='label']; X=np.array([[float(row[k]) for k in names] for row in rows]); y=np.array([int(row['label']) for row in rows])
split=int(len(X)*.8)
if split == 0 or len(set(y[:split])) < 2 or len(set(y[split:])) < 2: raise ValueError('chronological train and test splits must both contain both classes')
X_train,X_test,y_train,y_test=X[:split],X[split:],y[:split],y[split:]
model=LogisticRegression(max_iter=1000).fit(X_train,y_train); prob=model.predict_proba(X_test)[:,1]
metrics={'accuracy':accuracy_score(y_test,prob>=.5),'roc_auc':roc_auc_score(y_test,prob)}
artifact={'model':model,'features':names,'reference_mean':X_train.mean(axis=0).tolist(),'reference_std':X_train.std(axis=0).tolist()}; joblib.dump(artifact,a.model)
mlflow.set_experiment('binary-model-serving')
with mlflow.start_run():
 mlflow.log_params({'model':'logistic_regression','features':','.join(names),'train_rows':len(X_train)}); mlflow.log_metrics(metrics); mlflow.log_artifact(a.model)
print(json.dumps(metrics))
