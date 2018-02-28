import pandas as pd
import numpy as np
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("data.csv")
df.replace(to_replace='?',value=-99999,inplace=True)
df.fillna(value=-99999,inplace=True)
df=df.convert_objects(convert_numeric=True)
df['cigarette']=df['Smokes (packs/year)']*df['Smokes (years)']
df.drop(['STDs:HPV','Smokes (packs/year)','Smokes (years)','Smokes','Hormonal Contraceptives','IUD','STDs: Number of diagnosis','STDs','STDs: Time since first diagnosis','STDs: Time since last diagnosis','Dx','Hinselmann','Schiller','Citology'],axis=1,inplace=True)
X=np.array(df.drop(['Biopsy'],1));
Y=np.array(df['Biopsy']);
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
clf=svm.SVC(kernel='rbf',gamma=1000)
clf.fit(X_train,Y_train)
with open('Classifier.pkl','wb') as f:
	pickle.dump(clf,f)
