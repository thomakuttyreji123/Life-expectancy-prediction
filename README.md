# Life-expectancy-prediction
This is a hackathone project where I used DLKT  and weka library for the life expectancy prediction 

## Aim 
The main aim of the project is to predict the life expectancy of people according to various parameters. 
## Data
Dataset contains 22 columns and 2056 rows
## Data Preprocessing Plan
### Train Test Split
Before building the machine learning model we must have train and test data. Since the data set contains many null values, we can split the data with  0.7 fraction. 
### Correlation analysis
For understanding the correlation between features we used seaborn correlation heat map. After pointed out the relationships   we removed the null values. 
### Null Values and outlier Handling
1. Imputation technique : Correlated feature values are used to impute the corresponding missing values. 
2. Replacing with grouped mean : The columns which did not have a robust visible correlation were filled according to the              country group mean. If the null values for a particular country is less than 10, then we replace the value with the entire column    mean. (Note:for each country we have 15 data rows) 
3. Dropping: Through we have replaced many values. There can be null in certain country rows. Those values were dropped. 
4. Z scores were used for detecting outliers. 
5. For the data scaling we used  min max scalar ( sklearn )

## Models and test accuracies 
1. model 1 : r2 = 0.7663  (linear regression weka)
2. model 2 : r2 = 0.9494  (random forest weka) 
3. model 3 : r2 = 0.9535  ('')
4. model 4 : r2 = 0.9621  ('')
5. model 5 : r2 = 0.9673  ('')
6. model 6 : r2 = 0.9702  ( final model  random forest - weka)

## Remarks
Using better algorithms we can improve accuracy level.
## Credits
Thomaskutty Reji :https://github.com/thomakuttyreji123
ipshita gosh : https://github.com/ipshitag
