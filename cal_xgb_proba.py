import numpy as np
from xgboost import XGBClassifier

X=np.random.normal(1,1,[10,6])
y=np.random.randint(2,size=10)

model=XGBClassifier(learning_rate=0.1,n_estimators=2)
model.fit(X,y)

Xtest=np.random.normal(1,1,[2,6])
ytest=np.random.randint(2,size=2)
#get prediction results
model.predict_proba(Xtest)
#get tree results
model.get_booster().dump_model('output.txt')
with open('output.txt','r') as f:
    lmodel_leaves=f.read()
print(model_leaves)

#check proba results
#for each row, find the leaf value on each tree, there are two trees in this example
#proba = 1/(1+exp(-(tree0_leaf+tree1_leaf))
