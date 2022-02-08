# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:05:51 2022

@author: btindol
"""
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os 

os.chdir('C:/Users/btindol/OneDrive - Stryker/Python Scripts/Deploy_ML_App_to_Aure')

import warnings
warnings.filterwarnings('ignore')


# reading data
data = pd.read_csv("diabetes.csv")
data.head()

# variables data shape
print("Shape of dataset is" + str(data.shape))

data.info() # No nan values all normal

data.isnull().sum() # find the null values and sum

# Replace  value with na
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
data.isnull().sum()


# skewness for each column 
from scipy.stats import skew
for col in data.drop('Outcome',axis=1).columns:
    print("Skewness for the column {} is {}".format(col,data[col].skew()))
    
    
# insulin column skew is 2 so it is highly skewed
# Imputation with median of column
data['Insulin'] = data['Insulin'].fillna(data['Insulin'].median())

# do mean for glucose blood preasure and bmi and thinkness
for col in ['Glucose','BloodPressure','SkinThickness','BMI']:
    data[col] = data[col].fillna(data[col].mean())
    
data.isnull().sum() # no more null values 


################################################################################################################
################################################################################################################
# Exploratory Data Analysis
corr_data = data.corr()
sns.clustermap(corr_data,annot=True,cmap = 'YlGnBu',fmt = '.2f')
plt.title("Correlation Between Features")
plt.show()

# Histogram plots (outcome not equal make sample equal class imbalance)
fig = data.hist(figsize = (20,20), color = '#09FBD3',alpha = 0.7, rwidth = 0.85)


sns.set(style="whitegrid")
labels = ['Non-Diabetic','Diabetic']
sizes = data['Outcome'].value_counts(sort = True)

#(outcome not equal make sample equal class imbalance) 65 percent non diabetic and 34 diabetic
colors = ["#09FBD3","#FDF200"]
explode = (0.05,0)

plt.figure(figsize=(7,7))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=90,)
plt.title('Non-Diabetic vs Diabetic')
plt.show()


# Box and wisker plot outlier detection
plt.style.use('ggplot')
f, ax = plt.subplots(figsize=(11,15))
ax.set_facecolor('#09FBD3')
ax.set(xlim=(-.05,200))
plt.ylabel('Variables')
plt.title("Overview")
ax= sns.boxplot(data=data,orient='h',palette='Set2',)


def distplot(col_name):
    plt.figure()
    sns.set(style="whitegrid")
    ax = sns.distplot(data[col_name][data.Outcome ==1],color="gold",rug=True)
    sns.distplot(data[col_name][data.Outcome==0],color = '#09FBD3',rug=True)
    plt.legend(['Diabetic','Healthy'])
    
#Scatter plot glucose and age
sns.set(style='whitegrid')
g = sns.scatterplot(x='Glucose',y='Age',data=data,hue='Outcome',palette='pastel',legend=True)
plt.legend(title='Result',loc='upper left',labels=['Healthy','Diabetic'])
plt.show(g)

# box plot glucose and outcome outliers
pallete = {0:"#09FBD3",1:"gold"}
sns.boxplot(x="Outcome",y='Glucose',boxprops = dict(alpha=.5),data=data,palette=pallete)
plt.title('Glucose vs Outcome')
plt.show()

# Distplot function that we created with filled linegraph with distribution 
# Shows diabetic people tend to have much higher glucose level 
distplot('Glucose')

# Ladies with higher pregnancies tend to have higher pregnanceis
distplot('Pregnancies')

# Insulin
distplot('Insulin')

distplot('BloodPressure')


# BMI
pallete = {0:"#09FBD3",1:"gold"}
sns.boxplot(x="Outcome",y='BMI',boxprops = dict(alpha=.5),data=data,palette=pallete)
plt.title('Glucose vs Outcome')
plt.show()


##########################################################################################
##########################################################################################
# Modeling Classification models

df = data
zero = df[df['Outcome']==0] # zeros in outcome column 
one = df[df['Outcome']==1] # one values in outcome column 

from sklearn.utils import resample

# minority class taht 1 we need to upsample/inncrease that class so that there is no bias
# n_samples= 500 means we want to sample of class 1, since tehre are 500 samples of class 0
df_minority_unsampled = resample(one,replace=True,n_samples=500)

# concatenate
df=pd.concat([zero,df_minority_unsampled])

from sklearn.utils import shuffle
df = shuffle(df) # shuffling to avoid any particular sequence ( to avoid biased)

df.shape

# split into train and test split
from sklearn.model_selection import train_test_split
X= df.drop('Outcome',axis=1)
y=df['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)


#import models to try on train test split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pickle

from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
import seaborn as sns

# bring in all models from packages

classifiers = {
    "Naive Bayes": GaussianNB(),
    "LogisticRegression":LogisticRegression(),
    "KNN":KNeighborsClassifier(),
    "Support Vector Classifier":SVC(),
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()}

accuracy = []
cf_matrix = dict.fromkeys(classifiers.keys())

# for each model 
for key,classifier in classifiers.items():
    model = classifier.fit(X_train,y_train.values.ravel()) # take classifier and train it
    y_pred = model.predict(X_test)  # then predicti it
    cf_matrix[key] =confusion_matrix(y_test,y_pred) # make confusion matrix dictionary list 
    accuracy.append("{:.2f}".format(accuracy_score(y_test,y_pred))) # grab the accuracy and put in that list

    # Save the model to local storage
    filename = key +".sav"
    pickle.dump(classifier,open(filename,"wb"))
    
# get ready to plot
fig,axn = plt.subplots(1,6,sharex=True,sharey=True,figsize=(16,2))

for i, ax in enumerate(axn.flat):
    k = list(cf_matrix)[i]
    sns.heatmap(cf_matrix[k], ax=ax, cbar=i==6,annot = True,cmap="Blues")
    ax.set_title('{:s} \nAccuracy: {:s}'.format(k,accuracy[i]),fontsize=12)

####################################
## Testing the saved models
##################################
for key, _ in classifiers.items():
    filename = key+ ".sav"
    # load model from local storage
    loaded_model = pickle.load(open(filename,"rb"))
    y_pred = loaded_model.predict(X_test)
    
    cf_matrix[key] = confusion_matrix(y_test,y_pred)
    accuracy.append("{:.2f}".format(accuracy_score(y_test,y_pred)))
    
fig,axn = plt.subplots(1,6,sharex=True,sharey=True,figsize=(16,2))
for i, ax in enumerate(axn.flat):
    k = list(cf_matrix)[i]
    sns.heatmap(cf_matrix[k], ax=ax, cbar=i==6,annot = True,cmap="Blues")
    ax.set_title('{:s} \nAccuracy: {:s}'.format(k,accuracy[i]),fontsize=12)






