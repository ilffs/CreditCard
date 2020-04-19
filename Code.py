#Importing relevant libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
#Rounding to three decimal places
pd.set_option('float_format', '{:.3f}'.format)
import os
os. getcwd()
#Entering the sheet of data
df = pd.read_csv('datacreditcards.csv', header=None,)
df.head()
#Checking for missing values
df[df.isin(['?']).any(axis=1)]
#Naming columns
df = pd.read_csv('datacreditcards.csv', header=None, names=['Gender','Age','Debt','Married','EducationLevel','BankCustomer','YearsEmployed','PositiveAccountHistory','Employed','CreditScore','DriversLicense','Citizen','Income','Account balance','Approved'], decimal='.', na_values='?', dtype = {'Gender':'category','Married':'category','BankCustomer':'category','EducationLevel':'category','PositiveAccountHistory':'category','Employed':'category','DriversLicense':'category','Citizen':'category','Approved':'category'})
df.head() 
#Size of data    
print('Sheet contains ' + str(df.shape[0]) + ' lines and ' + str(df.shape[1]) + ' columns.')
summary = pd.DataFrame(df.dtypes, columns=['Dtype'])
summary['Nulls'] = pd.DataFrame(df.isnull().any())
summary['Sum_of_nulls'] = pd.DataFrame(df.isnull().sum())
summary['Per_of_nulls'] = round((df.apply(pd.isnull).mean()*100),2)
summary.Dtype = summary.Dtype.astype(str)
print(summary)
df.select_dtypes(include = ['category']).describe()
#Changing the type of variables
df['PositiveAccountHistory'] = df['PositiveAccountHistory'].astype('uint8')
df['Employed'] = df['Employed'].astype('uint8')
df['Gender'] = df['Gender'].astype('uint8')
df['DriversLicense'] = df['DriversLicense'].astype('uint8')
df['Approved'] = df['Approved'].astype('uint8')
df.head()
#Basic statistics 
stats = df.select_dtypes(['float', 'int']).describe()
stats = stats.transpose()
stats = stats[['count','std','min','25%','50%','75%','max','mean']]
stats
df.select_dtypes(['category', 'uint8']).columns
cat = pd.DataFrame(df.Gender.value_counts())
cat.rename(columns={'Gender':'Num_of_obs'},inplace=True)
cat['Per_of_obs'] = cat['Num_of_obs']/df.shape[0]*100
cat
#Crosstabs 
pd.crosstab(df.Gender, df.Approved)
pd.crosstab(df.Married, df.Approved)
pd.crosstab(df.BankCustomer, df.Approved)
pd.crosstab(df.EducationLevel, df.Approved)
pd.crosstab(df.PositiveAccountHistory, df.Approved)
pd.crosstab(df.Employed, df.Approved)
pd.crosstab(df.DriversLicense, df.Approved)
pd.crosstab(df.Citizen, df.Approved)
df = pd.concat([df,pd.get_dummies(df.Married, prefix='Married')], axis = 1)
df = pd.concat([df,pd.get_dummies(df.BankCustomer, prefix='BankCustomer')], axis = 1)
df = pd.concat([df,pd.get_dummies(df.EducationLevel, prefix='EducationLevel')], axis = 1)
df = pd.concat([df,pd.get_dummies(df.Citizen, prefix='Citizen')], axis = 1)
df.drop(['Married', 'BankCustomer', 'EducationLevel', 'Citizen'], axis = 1, inplace = True)
df.head()
df.Approved.value_counts(normalize=True)
y = df.Approved
df.drop('Approved', axis = 1, inplace = True)
x_tr, x_te, y_tr, y_te = train_test_split(df, y, test_size = 0.2, random_state = 7042018, stratify = y)
print(y_tr.value_counts(normalize = True))
print(y_te.value_counts(normalize = True))
#FIRST MODEL
model = DecisionTreeClassifier()
cv = cross_val_score(model, x_tr, y_tr, cv = 10, scoring = 'accuracy')
print('Average Accuracy: ' + str(cv.mean().round(3)))
x_tr.columns.shape
model.fit(x_tr, y_tr)
pred = model.predict(x_te)
print('Benchmark: ' + str(round(accuracy_score(pred, y_te),3)))
parameters = {'criterion':('entropy', 'gini'), 'splitter':('best','random'), 'max_depth':np.arange(1,10), 'min_samples_split':np.arange(2,10), 'min_samples_leaf':np.arange(1,5)}
#SECOND MODEL
classifier = GridSearchCV(DecisionTreeClassifier(), parameters, cv=10)
classifier.fit(x_tr, y_tr)
classifier.best_params_
cv = cross_val_score(DecisionTreeClassifier(**classifier.best_params_), x_tr, y_tr, cv = 10, scoring = 'accuracy')
print('Average Accuracy: ' + str(cv.mean().round(3)))
#THIRD MODEL
selector = RFE(model, 8, 6)
cols = x_tr.iloc[:,selector.fit(x_tr, y_tr).support_].columns
print(cols)
classifier2 = GridSearchCV(DecisionTreeClassifier(), parameters, cv=10)
classifier2.fit(x_tr[cols], y_tr)
print(classifier2.best_params_)
cv = cross_val_score(DecisionTreeClassifier(**classifier2.best_params_), x_tr[cols], y_tr, cv = 10, scoring = 'accuracy')
print('Average Accuracy: ' + str(cv.mean().round(3)))
model = DecisionTreeClassifier(**classifier2.best_params_)
model.fit(x_tr[cols], y_tr)
pred = model.predict(x_te[cols])
print('Final result is: ' + str(round(accuracy_score(pred, y_te),3)))
dot_data = tree.export_graphviz(model, out_file=None,feature_names=cols,class_names=['0','1'], filled = True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
out = dot_data[0:14] + 'ranksep=.75; size = "20,30";' + dot_data[14:]
graph = graphviz.Source(out)
graph.format = 'png'
graph.render('dtree_render',view=True)
samples = x_te[cols].sample(3).sort_index()
print(samples)
print(y_te[y_te.index.isin(samples.index)].sort_index())
print(model.predict(samples)) 