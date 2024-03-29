#calculate the xgboost probability
import numpy as np
from xgboost import XGBClassifier

#simulate inputs for training the model
#simulation with a normail distribution N{mean=1,std=1}, generating a matrix of 10*6, each element is iid from the normal distribution
X=np.random.normal(1,1,[10,6])
#randomly generate 10 number, either 1 or 0
y=np.random.randint(2,size=10)

#use xgboost to train
model=XGBClassifier(learning_rate=0.1,n_estimators=2)
model.fit(X,y)

#simulate test data
Xtest=np.random.normal(1,1,[2,6])
ytest=np.random.randint(2,size=2)
#get prediction results
model.predict_proba(Xtest)
#get tree results
model.get_booster().dump_model('output.txt')
with open('output.txt','r') as f:
    lmodel_leaves=f.read()
print(model_leaves)

#replicate proba results with tree leaf results
#for each row, find the leaf value on each tree, there are two trees in this example
#proba = 1/(1+exp(-(tree0_leaf+tree1_leaf))
