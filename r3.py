from imblearn.ensemble import BalancedRandomForestClassifier
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import seaborn as sns 
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif


train_data = pd.read_csv("Training_Data.csv")
test_data = pd.read_csv("Test_Data.csv") 

train_data['train'] = 1
test_data['train'] = 0

all_data = pd.concat([train_data,test_data]).copy()


x = all_data["Contract_End_Date"].str.split("/", n = 2, expand = True)
all_data["contract_end_d"]= x[0]
all_data["contract_end_m"]= x[1]
all_data["contract_end_y"]= x[2] 

x = all_data["Call_Date"].str.split("/", n = 2, expand = True)
all_data["call_d"]= x[0]
all_data["call_m"]= x[1]
all_data["call_y"]= x[2] 

train_data = all_data[all_data['train'] == 1].copy()
plot_data = all_data[all_data['train'] == 1].copy()

duplicateRowsDF = train_data[train_data.duplicated()] 


#%%
numeric = train_data[['Total_Number_of_Calls', 'AHT', 'Customer_Type']].copy()
categorical = train_data[['Product_Type', 'Product_Status', 'Customer_Type', 'contract_end_d', 'contract_end_m', 'contract_end_y', 'call_d', 'call_m', 'call_y']].copy()

customer_order = ['Type 1','Type 2','Type 3']


#%%
# Numeric variables

for i in numeric.columns:
    plt.hist(numeric[i])
    plt.title(i)
    plt.show()
    
# If AHT = (-) they called after contract expired
# If AHT = (+) they called before contract expiring

sns.histplot(data=numeric, x="AHT", hue="Customer_Type")
sns.histplot(data=numeric, x="Total_Number_of_Calls", hue="Customer_Type", multiple="dodge", shrink=10)

print(numeric.corr())
sns.heatmap(numeric.corr())


#%%
# Categorical variables 

for i in categorical.columns:
    sns.barplot(categorical[i].value_counts().index,categorical[i].value_counts()).set_title(i)
    plt.show()
  

for i, col in enumerate(categorical.columns):
    plt.figure(i)
    sns.countplot(x='Customer_Type', hue=col , data=categorical, order= customer_order)    
  
#%%
# Apply changes to both training and test sets
all_data["Total_Number_of_Calls"].replace({5: 4}, inplace=True)
all_data["contract_end_y"].replace({'2025': '25/26'}, inplace=True)
all_data["contract_end_y"].replace({'2026': '25/26'}, inplace=True)


def conditions(s):
    if (s['call_y'] == '2020'):
        return '2020'
    if (s['call_y'] == '2019') or (s['call_y'] == '2021') or (s['call_y'] == '2022'):
        return 'Not 2020'
    
all_data['call_y'] = all_data.apply(conditions, axis=1)

#%%
train_data = all_data[all_data['train'] == 1].copy()
test_data_csv = all_data[all_data['train'] == 0].copy()

#%%
a = train_data[['Total_Number_of_Calls', 'AHT']].copy()
scaler = MinMaxScaler()
a = scaler.fit_transform(a)
a = pd.DataFrame(data=a, columns=["Total_Number_of_Calls", "AHT"])


b = pd.get_dummies(train_data[['Product_Type', 'contract_end_y', 'call_y']], drop_first=True).copy()

X = pd.concat([a, b], axis=1, sort=False).copy()

y= train_data['Customer_Type'].copy()

#%%
# Balanced Random Forest

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.3, random_state=50)


brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train) 
y_pred = brf.predict(X_test)


cv = cross_val_score(brf,X_train,y_train,cv=5)
#print(cv)
brfcv = cv.mean()

#%%

X_resampled, y_resampled = SMOTE().fit_resample(X, y)
#print(sorted(Counter(y_resampled).items()))


X_train, X_test, y_train, y_test = train_test_split (X_resampled,y_resampled,test_size=0.3, random_state=50)

#%% 
# Logistic regression

lr = LogisticRegression(max_iter = 2000).fit(X_train, y_train) 
cv = cross_val_score(lr,X_train, y_train,cv=5)
#print(cv)
lrcv= cv.mean()

#%%
# KNN 

knn = KNeighborsClassifier().fit(X_train, y_train) 
cv = cross_val_score(knn,X_train,y_train,cv=5)
#print(cv)
knncv = cv.mean()

#%%
# XGB

xgb = XGBClassifier(random_state =1)
cv = cross_val_score(xgb,X_train, y_train,cv=5)
#print(cv)
xgbcv = cv.mean()


#%%
# Random forest
clf_smote = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
cv = cross_val_score(clf_smote,X_train,y_train,cv=5)
#print(cv)
rfcv = cv.mean()

#%%
# CV scores
cvs = [brfcv, lrcv, knncv, xgbcv, rfcv]
print (cvs)

#%%
# KNN to model data 

knn = KNeighborsClassifier().fit(X_train, y_train)
y_pred =knn.predict(X_test)

print(classification_report(y_test, y_pred))
print('Accuracy:',accuracy_score(y_test, y_pred))
#%%
confusion_matrix(y_test, y_pred)

#%%
# KNN to test data csv file PREDICTION

a = test_data_csv[['Total_Number_of_Calls', 'AHT']].copy()
scaler = MinMaxScaler()
a = scaler.fit_transform(a)
a = pd.DataFrame(data=a, columns=["Total_Number_of_Calls", "AHT"])


b = pd.get_dummies(test_data_csv[['Product_Type', 'contract_end_y', 'call_y']], drop_first=True).copy()

X_test_data_csv = pd.concat([a, b], axis=1, sort=False).copy()


knn_predict = KNeighborsClassifier().fit(X_resampled, y_resampled)
y_pred_csv =knn.predict(X_test_data_csv)


#%%
test_data_BG = pd.read_csv("Test_Data.csv") 
y_pred_csv = pd.DataFrame(data=y_pred_csv, columns=["Customer_Prediction"])
test_data_BG = pd.concat([test_data_BG, y_pred_csv], axis=1, sort=False).copy()
test_data_BG.to_csv('test_data_BG.csv', index=False)
#%%
#Feature selection

a = pd.DataFrame(columns = ['Number of Calls', 'AHT', 'Product 2', 'Product 3', 'Product 4', 'Contract End 2021', 'Contract End 2022', 'Contract End 25/26', 'Called in 2021'])
importances = mutual_info_classif(X_resampled, y_resampled)
feat_importances = pd.Series(importances, a.columns[0:len(a.columns)])
feat_importances = feat_importances.sort_values(ascending=False)
feat_importances.plot(kind='barh', color= 'teal')
plt.show


Z = X[['AHT', 'Product_Type_Product 4', 'contract_end_y_25/26']].copy()


X_resampled, y_resampled = SMOTE().fit_resample(Z, y)
#print(sorted(Counter(y_resampled).items()))


X_train, X_test, y_train, y_test = train_test_split (X_resampled,y_resampled,test_size=0.3, random_state=50)


knn = KNeighborsClassifier().fit(X_train, y_train) 
cv = cross_val_score(knn,X_train,y_train,cv=5)
print(cv)


knn = KNeighborsClassifier().fit(X_train, y_train)
y_pred =knn.predict(X_test)

print(classification_report(y_test, y_pred))
print('Accuracy:',accuracy_score(y_test, y_pred))

#%%
#Plots 

numeric = plot_data[['Total_Number_of_Calls', 'AHT', 'Customer_Type']].copy()
categorical = plot_data[['Total_Number_of_Calls', 'Product_Type', 'Customer_Type', 'contract_end_y', 'call_y']].copy()

customer_order = ['Type 1','Type 2','Type 3']
product_order = ['Product 1','Product 2','Product 3', 'Product 4']
year_order = ['2020', '2021', '2022', '2023', '2024', '2025', '2026']
year_order2 = ['2020', '2021', '2022']

#%%
# If AHT = (-) they called after contract expired
# If AHT = (+) they called before contract expiring

sns.set_theme()
sns.set_style("dark")


sns.histplot(data=numeric, x="AHT", hue="Customer_Type", palette=["green", "orange", "blue"]).set(ylabel=None)
plt.legend(title='Customer Type', labels=['1', '2', '3'])

sns.countplot(x='Total_Number_of_Calls', hue='Customer_Type' , data=categorical, hue_order = ['Type 1', 'Type 2', 'Type 3']).set(ylabel=None) 
plt.legend(title="Customer Type", labels=['1', '2', '3'], loc='upper right')
plt.xlabel("Total Number of Calls")
#plt.legend([],[], frameon=False)

sns.countplot(x='Product_Type', hue='Customer_Type' , data=categorical, order= product_order, hue_order = ['Type 1', 'Type 2', 'Type 3']).set(ylabel=None) 
plt.legend(title="Customer Type", labels=['1', '2', '3'], loc='upper right')
plt.xlabel("Product Type")
#plt.legend([],[], frameon=False)

sns.countplot(x='contract_end_y', hue='Customer_Type' , data=categorical, order= year_order, hue_order = ['Type 1', 'Type 2', 'Type 3']).set(ylabel=None) 
plt.legend(title="Customer Type", labels=['1', '2', '3'], loc='upper right')
plt.xlabel("Contract End Year")
#plt.legend([],[], frameon=False)

sns.countplot(x='call_y', hue='Customer_Type' , data=categorical, order= year_order2, hue_order = ['Type 1', 'Type 2', 'Type 3']).set(ylabel=None) 
plt.legend(title="Customer Type", labels=['1', '2', '3'], loc='upper right')
plt.xlabel("Call Year")
#plt.legend([],[], frameon=False)

sns.countplot(x='Customer_Type', data=categorical, order=["Type 1", "Type 2", "Type 3"]).set(ylabel=None) 
plt.xlabel("Customer Type")


sns.despine()













