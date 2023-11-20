# MINI-PROJECT-ANALYSIS-AND-ESTIMATION-OF-HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING
## INTRODUCTION
 In past estimating the price of a house or apartment its almost manually
calculated which takes certain amount of time if there is a bulk number of houses.
For the price estimation we will be referring to certain features like no.of. sqft,
bedrooms, living area, amenities etc.…Some of the websites having this information
houses for sale are also mostly not having price estimation option automatically.
Automating the manual calculation with the help of machine learning models which
are concepts of AI quickly and efficiently. Data is the heart of machine learning.
Predictive models use data for training which gives somewhat accurate results.
Without data we can’t train the model. Machine learning involves building these
models from data and uses them to predict new data.
# Statement of the Problem
In the ever-evolving real estate market, the conventional method of manually setting house or apartment prices by owners on platforms like OLX poses significant challenges. This manual approach often results in pricing misalignments, where listed prices may not accurately reflect the true value of the property. To address this issue, the research project focuses on building a forecast model using machine learning algorithms to automate and enhance the accuracy of house price estimations.
Owner-Driven Pricing
Potential Misrepresentation
Data-Driven Solution
Dataset Parameters
Objective

# Purpose Of The Project
The primary purpose of this project is to revolutionize house price estimation in the real estate landscape by harnessing the capabilities of machine learning. The endeavor is driven by the imperative to provide stakeholders, including potential buyers and investors, with a robust and automated solution for precise house price predictions
1.Automated Precision
2.Data-Driven Insights
3.Transparent Transactions
4.Market Intelligence
5.Technological Advancements

# Proposed Methodology
## Diverse Data Aggregation:
Involves gathering a varied and comprehensive dataset from multiple sources within the real estate domain.
Encompasses diverse factors such as location, size, amenities, and historical pricing data for a holistic understanding.
## Algorithm Application:
Utilizes machine learning algorithms, including Decision Trees and possibly ensemble methods, to process the aggregated data.
Applies algorithms to learn patterns and relationships within the data, crucial for accurate house price predictions.
## Rigorous Model Optimization:
Focuses on refining and fine-tuning the machine learning models to enhance their predictive accuracy.
Involves iterative processes of evaluation, adjustment, and optimization to ensure robust and reliable house price predictions.

# Flow Diagram
![image](https://github.com/PAARKAVYB/MINI-PROJECT-ANALYSIS-AND-ESTIMATION-OF-HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING/assets/93509383/0dbac465-4782-441f-a0f5-b0abf7ba6b3f)

# Expected results and its Implications
Squared Error, the model ensures reliable predictions, benefiting both buyers and sellers in the real estate market. Buyers gain fair and transparent price estimates, while sellers can strategically price properties for enhanced competitiveness. Industry professionals can leverage the model for precise valuations and strategic insights, advancing real estate analytics. The project's adaptability, with potential extensions for real-time data integration and market trend analysis, ensures relevance in the ever-evolving real estate landscape, providing stakeholders with a valuable tool for navigating property valuation complexities in today's dynamic housing market.

# Architecture Diagram
![image](https://github.com/PAARKAVYB/MINI-PROJECT-ANALYSIS-AND-ESTIMATION-OF-HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING/assets/93509383/8c6e13c7-1293-4e30-80fb-2a23080362c1)


## Data Collection and Data Cleaning
Data collection or data gathering is the process of gathering and measuring information on targeted variables in an established system, which then enables one to answer relevant questions and evaluate outcomes.

The gathered dataset is being cleaned using python and pandas library. The dataset is being read and the null values , duplicate values are removed from the dataset and conversion of the terms is done and a new cleaned data is generated using pandas library and python.
<img width="507" alt="image" src="https://github.com/PAARKAVYB/MINI-PROJECT-ANALYSIS-AND-ESTIMATION-OF-HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING/assets/93509383/c3b8a233-4ecb-4f29-af2d-74ed9723db42">

# CODE
### Loading Saved Data from Web Scraping
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
raw = pd.read_csv("raw.csv")
raw.head()
raw.shape
raw.info()

```
### Initial Check for Duplicates and Missing Values
```
rawcheck = raw.drop_duplicates()
rawcheck = rawcheck.dropna().reset_index(drop=True)
rawcheck.isnull().sum()

```
### Data Cleaning
```
cleaning columns data by removing extra symbols and other
raw.head()
raw.price.unique()

```
### Price
```
raw['price'] = raw['price'].str.replace(",","").str.replace("₹","").str.strip().astype(float)/1000
raw.facing.unique()

```
### Facing
```
for i in range(0,len(raw)):
 try:
 raw['facing'][i] = raw['facing'][i].strip("['']").lower().replace("-","")
 except:
 continue
raw.floor_no.unique()

```
### Floor_number
```
for i in range(0,len(raw)):
 try:
 raw['floor_no'][i] = raw['floor_no'][i].strip("['']")
 except:
 continue
raw.floor_no = raw.floor_no.astype(float)
raw.constr_info.unique()
 Construction Status
for i in range(0,len(raw)):
 try:
 raw['constr_info'][i] = raw['constr_info'][i].strip("['']").lower().replace(" ","")
 except:
 continue
raw.furnishing.unique()
```
### Furnishing
```
for i in range(0,len(raw)):
 try:
 raw['furnishing'][i] = raw['furnishing'][i].strip("['']").lower().replace("-","")
 except:
 continue
raw.bathrooms.unique()
```
### Bathrooms
```
for i in range(0,len(raw)):
 try:
 raw['bathrooms'][i] = raw['bathrooms'][i].strip("['']")
 except:
 continue
raw.bathrooms.replace('4+', '5', inplace=True)
raw.bathrooms = raw.bathrooms.astype(float)
raw.bedrooms.unique()

```
### Bedrooms
```
for i in range(0,len(raw)):
 try:
 raw['bedrooms'][i] = raw['bedrooms'][i].strip("['']")
 except:
 continue
raw.bedrooms.replace('4+', '5', inplace=True)
raw.bedrooms = raw.bedrooms.astype(float)
raw.sqft.unique()

```
### Sqft
```
for i in range(0,len(raw)):
 try:
 raw['sqft'][i] = raw['sqft'][i].strip("['']")
 except:
 continue
raw.sqft = raw.sqft.astype(float)
raw.house_type.unique()
House_type
for i in range(0,len(raw)):
 try:
 raw['house_type'][i] = raw['house_type'][i].strip("['']").lower().replace(" 
","").replace("&","")
 except:
 continue
raw.location.unique()

```
### Location
```
raw['city'] = np.nan
raw['state'] = np.nan
for i in range(0,len(raw)):
 try:
 city, state = raw['location'][i].strip().split(",")[-2:]
 raw['city'][i] = city
 raw['state'][i] = state 
 except:
 continue
raw.head()
```
### Cleaned data
```
data = raw[['state','city','house_type','sqft','bedrooms','bathrooms',
 'furnishing','constr_info','floor_no','facing','price']]
data.head()
data.to_csv("cleaned.csv", index=False)
Missing Values
data.isnull().sum()
data.dtypes
Replacing Missing values with one single loop for both numerical and categorical
for i in data.columns:
 if data[i].dtype == 'object':
 modevalue = data[i].mode()[0]
 data[i].fillna(modevalue, inplace = True)
 elif data[i].dtype == 'int32' or 'int64' or 'float64':
 data[i].fillna(data[i].median(), inplace = True)

```
### Exploratory Data Analysis
```
data.head()
data.info()
data.describe(include = 'all')
Visualizations
sns.scatterplot(x = 'sqft', y = 'price', data = data)
Data Pre-processing
Outlier treatment
data.info()
sns.boxplot(data.price)
def outlier_detect(df):
 for i in df.describe().columns:
 Q1=df[i].quantile(0.25)
 Q3=df[i].quantile(0.75)
 low=df[i].quantile(0.05)
 high=df[i].quantile(0.95)
 IQR=Q3 - Q1
 LTV=Q1 - 1.5 * IQR
 UTV=Q3 + 1.5 * IQR
 df[i] = df[i].mask(df[i]<LTV,low)
 df[i] = df[i].mask(df[i]>UTV,high)
 return df
data = outlier_detect(data)
sns.boxplot(data.price)

```
### One_Hot Encoding
```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore',sparse = False)
cats = pd.DataFrame(ohe.fit_transform(data.iloc[:,[0,1,2,6,7,9]]))
cats.columns = ohe.get_feature_names_out()
data.iloc[:,[3,4,5,8,10]]
mdata = pd.concat([data.iloc[:,[3,4,5,8,10]],cats], axis=1)
mdata
X & Y
X = mdata.drop('price',axis=1)
y = mdata['price']
Train & Test Split of Data for ML Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Random Forest Regressor Model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_model.score(X_train, y_train)
y_pred = rf_model.predict(X_test)
rf_model.score(X_test, y_test)
sns.distplot(y_test - y_pred)
from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred))
print("MSE: ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('y_test')
plt.ylabel('y_label')
plt.show()
metrics.r2_score(y_test, y_pred)

```
### Realtime Estimation
```
data.head()
One Example for Prediction
check = [['kerala','kochi','apartments',1000,3,2,'furnished','newlaunch',5,'east']]
check = pd.DataFrame(check, columns = ['state', 'city', 'house_type', 'sqft', 'bedrooms', 
'bathrooms',
 'furnishing', 'constr_info','floor_no','facing'])
cats = pd.DataFrame(ohe.transform(check.iloc[:,[0,1,2,6,7,9]]))
cats
cats.columns = ohe.get_feature_names_out()
check = pd.concat([check.iloc[:,[3,4,5,8]],cats], axis=1)
check
rf_model.predict(check)[0]
3592*1000 # multiplying with 1000 to get actual value in rupees
```

### One Function for Predict
```
data.columns
def predict_houseprice(): 
 print("Enter the Following Details To Estimate a House Value:")
 print()
 state = input("Enter State Name: ")
 city = input("Enter City Name from State: ")
 house_type = input("Enter House Type (apartments/housevillas/builderfloors/farmhouses): ")
 sqft = float(input("Enter number of Sqft: "))
 bedrooms = int(input("Enter number of Bedrooms: "))
 bathrooms = int(input("Enter number of Bathrooms: "))
 furnishing = input("Enter Furnishing Info (unfurnished/semifurnsihed/furnished): ")
 constr = input("Enter Construction Info (newlaunch/readytomove/underconstruction): ")
 floor = int(input("Enter floor number: "))
 facing = input("Enter Facing: ") 
 print() 
 data = [[state,city,house_type,sqft,bedrooms,bathrooms,furnishing,constr,floor,facing]]
 data = pd.DataFrame(data, columns = ['state', 'city', 'house_type', 'sqft', 'bedrooms', 
'bathrooms',
 'furnishing', 'constr_info','floor_no','facing']) 
 print("Given data: ")
 display(data)
 cats = pd.DataFrame(ohe.transform(data.iloc[:,[0,1,2,6,7,9]]))
 cats.columns = ohe.get_feature_names_out()
 row = pd.concat([data.iloc[:,[3,4,5,8]],cats], axis=1) 
 Predictions 
 price = rf_model.predict(row)[0]
 roundedprice = round((price*1000)/100000,2) 
 print()
 print("Estimated Price Value: ", "₹ "+str(roundedprice)+" lacs")
 predict_houseprice()
```
```

