import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt



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




#%%
numeric = train_data[['Total_Number_of_Calls', 'AHT', 'Customer_Type']].copy()
categorical = train_data[['Total_Number_of_Calls', 'Product_Type', 'Customer_Type', 'contract_end_y', 'call_y']].copy()

customer_order = ['Type 1','Type 2','Type 3']
product_order = ['Product 1','Product 2','Product 3', 'Product 4']
year_order = ['2020', '2021', '2022', '2023', '2024', '2025', '2026']
year_order2 = ['2020', '2021', '2022']

#%%
# If AHT = (-) they called after contract expired
# If AHT = (+) they called before contract expiring

sns.set_theme()
sns.set_style("dark")


sns.histplot(data=numeric, x="AHT", hue="Customer_Type").set(ylabel=None)
plt.legend(title='Customer Type', labels=['1', '2', '3'])
plt.legend([],[], frameon=False)

sns.countplot(x='Total_Number_of_Calls', hue='Customer_Type' , data=categorical, hue_order = ['Type 1', 'Type 2', 'Type 3']).set(ylabel=None) 
plt.legend(title="Customer Type", labels=['1', '2', '3'], loc='upper right')
plt.xlabel("Product Type")
plt.legend([],[], frameon=False)

sns.countplot(x='Product_Type', hue='Customer_Type' , data=categorical, order= product_order, hue_order = ['Type 1', 'Type 2', 'Type 3']).set(ylabel=None) 
plt.legend(title="Customer Type", labels=['1', '2', '3'], loc='upper right')
plt.xlabel("Product Type")
plt.legend([],[], frameon=False)

sns.countplot(x='contract_end_y', hue='Customer_Type' , data=categorical, order= year_order, hue_order = ['Type 1', 'Type 2', 'Type 3']).set(ylabel=None) 
plt.legend(title="Customer Type", labels=['1', '2', '3'], loc='upper right')
plt.xlabel("Contract End Year")
plt.legend([],[], frameon=False)

sns.countplot(x='call_y', hue='Customer_Type' , data=categorical, order= year_order2, hue_order = ['Type 1', 'Type 2', 'Type 3']).set(ylabel=None) 
plt.legend(title="Customer Type", labels=['1', '2', '3'], loc='upper right')
plt.xlabel("Call Year")
plt.legend([],[], frameon=False)


sns.countplot(x='Customer_Type', data=categorical, order=["Type 1", "Type 2", "Type 3"]).set(ylabel=None) 
plt.xlabel("Customer Type")


sns.despine()

#%%


a = pd.DataFrame(columns = ['Number of Calls', 'AHT', 'Product 2', 'Product 3', 'Product 4', 'Contract End 2021', 'Contract End 2022', 'Contract End 25/26', 'Called in 2021'])
importances = mutual_info_classif(X_resampled, y_resampled)
feat_importances = pd.Series(importances, a.columns[0:len(a.columns)])
feat_importances = feat_importances.sort_values(ascending=False)
feat_importances.plot(kind='barh', color= 'teal')
plt.show








    
    
    
    
    
    