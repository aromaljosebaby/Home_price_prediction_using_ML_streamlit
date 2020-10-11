


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

house_data=pd.read_csv('data_for_macjine_learning/py/DataScience/BangloreHomePrices/model/bengaluru_house_prices.csv')
house_data=house_data.drop('society',axis='columns')

location_unique_names=house_data.location.unique()
area_type_unique_names=house_data.area_type.unique()

#house_data


# In[24]:



#making dummy values or one hot encding for area type feature
dummies_for_area_type=pd.get_dummies(house_data.area_type)
house_data=pd.concat([house_data,dummies_for_area_type],axis='columns')
house_data=house_data.drop('area_type',axis='columns')

#changing the availability into 1 or 0
house_data['availability']=house_data['availability'].apply(lambda x: 1 if x=='Ready To Move' else 0)



col_with_null_values=list(house_data.columns[house_data.isna().any()])  # gives columns name of elememts with null value




#print(house_data.isna().sum())

house_data=house_data.dropna()  # drops all rows with null data,we do this here becoz total number of rows with null data is less


#print(house_data['size'].unique())


house_data['bhk']=house_data['size'].apply(lambda x: int(x.split()[0]))# x iterates on each row value of size column
house_data=house_data.drop('size',axis='columns')



# In[25]:



def is_float(x):
    try:
        float(x)
    except:
        return  False
    return True

#print( house_data[~house_data['total_sqft'].apply(is_float)])


#now we see that the house data column has irregular datat ie someplaces it has 1245-5678 ,someplaces it has another unit so we gotta clean  tat up

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2  # if its like a range 345-658
    try:
        return float(x)
    except:
        return None
      # ie if it is a value like e563cm2


house_data['total_sqft']=house_data['total_sqft'].apply(convert_sqft_to_num)  # applying above function


#make a new col which says price per sq ft,note here price is in lacks

house_data['price_per_sqft']=house_data['price']*100000/house_data['total_sqft']



data_showing_locations_vs_numbers=house_data.groupby('location')['location'].agg('count').sort_values(ascending=False)

location_stats_less_than_10=data_showing_locations_vs_numbers[data_showing_locations_vs_numbers<10]


house_data['location']=house_data['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x )


#removing outliner data
#print(house_data[house_data.total_sqft/house_data.bhk<300])




house_data = house_data[~(house_data.total_sqft/house_data.bhk<300)]#ie taking opp of all data which has tat ratio greater than 300 or taking all data which has tat ratio <300





# In[26]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
house_data = remove_pps_outliers(house_data)




# In[27]:


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    #plt.show()



plot_scatter_chart(house_data, "Hebbal")


# In[28]:


# remving 3 bhk flats which have less cost than 2bhk flats with same area

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
house_data = remove_bhk_outliers(house_data)
# df8 = df7.copy()


plot_scatter_chart(house_data,"Hebbal")


# In[29]:


#histogram

matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(house_data.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
#plt.show()


# In[30]:


house_data=house_data[house_data.bath<house_data.bhk+2]


house_data=house_data.drop(['price_per_sqft'],axis='columns')


location_dummies=pd.get_dummies(house_data.location)


house_data=pd.concat([house_data,location_dummies],axis='columns')
house_data=house_data.drop(['location','other'],axis='columns')



# In[31]:


X=house_data.drop(['price','availability'],axis='columns')


y=house_data['price']


X[['Built-up  Area','bhk']]=X[['bhk','Built-up  Area']]
X=X.rename(columns={'bhk':'temp','Built-up  Area':'bhk'})
X=X.rename(columns={'temp':'Built-up  Area'})



# In[32]:


def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },


        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [1, 5, 10, 40]
            }
        },


    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

scores=find_best_model_using_gridsearchcv(X,y)
print(scores[['model','best_score']])


# In[33]:




# In[34]:


#X.iloc[:10,:15]


# In[35]:


loc_index = np.where(X.columns=='1st Block Koramangala')[0][0]



# In[ ]:



def predicting_using_model(total_sqft,no_of_bathrooms,no_of_balcony,area_type,bhk,location):
    loc_index_of_area_type = np.where(X.columns==area_type)[0][0]
    loc_index_of_location = np.where(X.columns==location)[0][0]
    predicting_values=np.zeros(len(X.columns))
    predicting_values[0]=total_sqft
    predicting_values[1]=no_of_bathrooms
    predicting_values[2]=no_of_balcony
    predicting_values[3]=bhk
    
    if loc_index_of_area_type>0:        
        predicting_values[loc_index_of_area_type]=1
    if loc_index_of_location>0:        
        predicting_values[loc_index_of_location]=1
    
    return predicting_values


# In[49]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


final_data=predicting_using_model(1056.0,2.0,1.0,'Super built-up  Area',2,'Electronic City Phase II')
model=LinearRegression()
model.fit(X_train,y_train)
model.predict([final_data])


# In[50]:


import pickle

with open('pickle_file/model.pickle','wb') as f:
    pickle.dump(model,f)


# In[51]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns],'area_type_column_names':area_type_unique_names,'location_column_names':location_unique_names
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




