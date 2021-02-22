


# importing libraries 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


# importing ml libraries 
from sklearn.model_selection import train_test_split 

# loading the data set and spliting it for getting a test data 
df = pd.read_csv('D:\my_folders\Git_local_repo\Hackathone-Life_Expectancy_Prediction\Life Expectancy Data.csv') 
data,test_data = train_test_split(df,test_size = 0.2,random_state= 42)


# droping the na values in the test data 
# test_data.dropna(inplace = True) 
# test_data.to_csv('Life Expectancy Data test.csv', index = False) 

# Imputing missing values of 'Schooling' column 
def impute_schooling(c):
    s=c[0]
    l=c[1]
    if pd.isnull(s):
        if l<= 40:
            return 8.0
        elif 40<l<=44:
            return 7.5
        elif 44<l<50:
            return 8.1
        elif 50<l<=60:
            return 8.2
        elif 60<l<=70:
            return 10.5
        elif 70<l<=80:
            return 13.4
        elif l>80:
            return 16.5
    else:
        return s
    
data['Schooling']=data[['Schooling','Life expectancy ']].apply(impute_schooling,axis=1)


# Imputing missing values of 'Alcohol' column 
def impute_Alcohol(cols):
    al=cols[0]
    sc=cols[1]
    if pd.isnull(al):
        if sc<=2.5:
            return 4.0
        elif 2.5<sc<=5.0:
            return 1.5
        elif 5.0<sc<=7.5:
            return 2.5
        elif 7.5<sc<=10.0:
            return 3.0
        elif 10.0<sc<=15:
            return 4.0
        elif sc>15:
            return 10.0
    else:
        return al
    
data['Alcohol']=data[['Alcohol','Schooling']].apply(impute_Alcohol,axis=1)


# Imputing missing values of ''Income composition of resources'' column 
def impute_Income(c):
    i=c[0]
    l=c[1]
    if pd.isnull(i):
        if l<=40:
            return 0.4
        elif 40<l<=50:
            return 0.42
        elif 50<l<=60:
            return 0.402
        elif 60<l<=70:
            return 0.54
        elif 70<l<=80:
            return 0.71
        elif l>80:
            return 0.88
    else:
        return i
        
data['Income composition of resources']=data[['Income composition of resources','Life expectancy ']].apply(impute_Income,axis=1)


def outlier_replace(col):
   for i in countries:
       for j in groups.get_group(i)[col]:
               threshold = 3
               mean = np.mean(groups.get_group(i)[col])
               std = np.std(groups.get_group(i)[col])
       if std != 0:                     
           z_score = (j - mean) / std
           if np.abs(z_score) > threshold:
               j = data[col][data['Country'] == i].mean() 


data.dropna(subset=['Life expectancy '],inplace = True) 
countries = data['Country'].unique()

# we are creating groups of countries 

groups = data.groupby('Country')

# from the avialble data we know that values depends on country , 
# so we are going to handle the missing values and outliers of some columns  with respect to the country


# creating a new list contains gdp null values greater than 10.. this simply means that we cannot update the null with 
# the respective country mean .. 
gdpnull_c = []
for i in countries:
    if groups.get_group(i)['GDP'].isna().sum() >10:
        gdpnull_c.append(i)
        
        
        
# for countries with less gdp null then fill it with mean of gdp  values with respect to each country
for i in countries:
    if i not in gdpnull_c:
        for j in groups.get_group(i)['GDP']:
            data['GDP'][data['Country'] == i]= groups.get_group(i)['GDP'].fillna(groups.get_group(i)['GDP'].mean()) 
            
# for those countries null values more than 10 fill it with  mean of 'GDP'  of entire dataframe
for i in gdpnull_c:
    data['GDP'][data['Country'] == i]=groups.get_group(i)['GDP'].fillna(data['GDP'].mean())
    
# replacing outlier with mean of the rest values in the respective country:

outlier_replace('GDP')

# there are some countries for which we dont have the 15 years data.. so eventhough we did above steps, we may not replace
# null values of such coutries... 

# so , we are droping rest na values ( 5 rows)
data.dropna(subset=['GDP'],inplace = True) 


# hepatities     outlier  and null analysis

# same process in the case of gdp data handling ( refer )
countries = data['Country'].unique()
groups = data.groupby('Country')
gnull_c = []
for i in countries:
    if groups.get_group(i)['Hepatitis B'].isna().sum() >10:
        gnull_c.append(i) 


# treating outlier 'Hepatitis B'values among countries which contain less number of nulls
outlier_replace('Hepatitis B') 

# we replace all null values by mean 'Hepatities B' of the corresponding countries ( countries not in gnull_c)    
for i in countries:
    if i not in gnull_c:
        for j in groups.get_group(i)['Hepatitis B']:
            data['Hepatitis B'][data['Country'] == i]= groups.get_group(i)['Hepatitis B'].fillna(groups.get_group(i)['Hepatitis B'].mean()) 
# for those countries in gnull_c we replace it with mean of 'Hepatitis B'  in the entire dataframe
for i in gnull_c:   
    data['Hepatitis B'][data['Country'] == i]=groups.get_group(i)['Hepatitis B'].fillna(data['Hepatitis B'].mean())

# same processing ( refer gdp data handling process)   
data.dropna(subset=['Hepatitis B'],inplace = True) 


gnull_c = []
for i in countries:
    if groups.get_group(i)['Total expenditure'].isna().sum() >10:
        gnull_c.append(i)

        
outlier_replace('Total expenditure') 


for i in countries:
    if i not in gnull_c:
        for j in groups.get_group(i)['Total expenditure']:
            data['Total expenditure'][data['Country'] == i]= groups.get_group(i)['Total expenditure'].fillna(groups.get_group(i)['Total expenditure'].mean()) 

for i in gnull_c:   
    data['Total expenditure'][data['Country'] == i]=groups.get_group(i)['Total expenditure'].fillna(data['Total expenditure'].mean())

data.dropna(subset=['Total expenditure'],inplace = True) 


# Another imputation technique on bmi 
data = data.drop(' thinness 5-9 years',axis = 1)
def impute_BMI(c):
    b=c[0]
    l=c[1]
    if pd.isnull(b):
        if l<=50:
            return 25.0
        elif 50<l<=60:
            return 25.0
        elif 60<l<=70:
            return 32.0
        elif 70<l<=80:
            return 46.8
        elif 80<l<=100:
            return 60.0
    else:
        return b  
data[' BMI ']=data[[' BMI ','Life expectancy ']].apply(impute_BMI,axis=1) 


# handling population feature 
def impute_population(c):
    p=c[0]
    i=c[1]
    if pd.isnull(p):
        if i<=100:
            return 0.19*((10)**9)
        elif 100<i<=250:
            return 0.18*((10)**9)
        elif 250<i<=350:
            return 0.02*((10)**9)
        elif 350<i<=900:
            return 0.1*((10)**9)
        elif 900<i<=1100:
            return 0.18*((10)**9)
        elif 1100<i<=1250:
            return 0.05*((10)**9)
        elif 1250<i<=1500:
            return 0.19*((10)**9)
        elif 1500<i<=1750:
            return 0.05*((10)**9)
        elif i>1750:
            return 0.1*((10)**9)
    else:
        return p
data['Population']=data[['Population','infant deaths']].apply(impute_population,axis=1) 


#  handling missing data in thinness feature 
def impute_Thin_1(c):
    t=c[0]
    b=c[1]
    if pd.isnull(t):
        if b<=10:
            return 5.0
        elif 10<b<=20:
            return 10.0
        elif 20<b<=30:
            return 8.0
        elif 30<b<=40:
            return 6.0
        elif 40<b<=50:
            return 3.0
        elif 50<b<=70:
            return 4.0
        elif b>70:
            return 1.0
    else:
        return t
    
data[' thinness  1-19 years']=data[[' thinness  1-19 years',' BMI ']].apply(impute_Thin_1,axis=1) 



# polio feature data handling 
countries = data['Country'].unique()
groups = data.groupby('Country')
gnull_c = []
for i in countries:
    if groups.get_group(i)['Polio'].isna().sum() >10:
        gnull_c.append(i)

outlier_replace('Polio') 

for i in countries:
    if i not in gnull_c:
        for j in groups.get_group(i)['Polio']:
            data['Polio'][data['Country'] == i]= groups.get_group(i)['Polio'].fillna(groups.get_group(i)['Polio'].mean()) 
for i in gnull_c:   
    data['Polio'][data['Country'] == i]=groups.get_group(i)['Polio'].fillna(data['Polio'].mean())
data.dropna(subset=['Polio'],inplace = True) 



# dipheria outlier hadnling 
countries = data['Country'].unique()
groups = data.groupby('Country')
gnull_c = []
for i in countries:
    if groups.get_group(i)['Diphtheria '].isna().sum() >10:
        gnull_c.append(i)

outlier_replace('Diphtheria ') 


# handling missing dat ain diphtheria 
for i in countries:
    if i not in gnull_c:
        for j in groups.get_group(i)['Diphtheria ']:
            data['Diphtheria '][data['Country'] == i]= groups.get_group(i)['Diphtheria '].fillna(groups.get_group(i)['Diphtheria '].mean())
            
           
for i in gnull_c:   
    data['Diphtheria '][data['Country'] == i]=groups.get_group(i)['Diphtheria '].fillna(data['Diphtheria '].mean())
data.dropna(subset=['Diphtheria '],inplace = True) 



feature = ['Adult Mortality',
       'Alcohol', ' BMI ',
       'under-five deaths ', 'Polio',
       ' HIV/AIDS', ' thinness  1-19 years', 'Schooling','Income composition of resources']  

# model training 
from sklearn.ensemble import GradientBoostingRegressor
mdl = GradientBoostingRegressor(max_depth=2, random_state=0)


# splitting the data into X and y 
X_train = data[feature]
y_train = data['Life expectancy ']

mdl.fit(X_train,y_train)
print(mdl.score(X_train,y_train)) 



# loading the test data and splitting it 
test_data = pd.read_csv('D:\my_folders\Git_local_repo\Hackathone-Life_Expectancy_Prediction\Life Expectancy Data test.csv')

# splitting X and y 
X_test = test_data[feature]
y_test = test_data['Life expectancy ']

# printing the test accuracy 
print(mdl.score(X_test,y_test))


# saving the model 
import pickle   
# using pickle to save the model to disk 
pickle.dump(mdl,open('D:\my_folders\Git_local_repo\Hackathone-Life_Expectancy_Prediction\life_expectancy_model.pkl','wb'))

