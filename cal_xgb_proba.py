import numpy as np
import random
from xgboost import XGBClassifier

X=np.random.normal(1,1,[10,6])
y=np.random.randint(2,size=10)

model=XGBClassifier(learning_rate=0.1,n_estimators=2)
model.fit(X,y)

Xtest=np.random.normal(1,1,[2,6])
ytest=np.random.randint(2,size=2)

model.predict_proba(Xtest)
model.get_booster().dump_model('output.txt')
with open('output.txt','r') as f:
    lmodel_leaves=f.read()
print(model_leaves)
