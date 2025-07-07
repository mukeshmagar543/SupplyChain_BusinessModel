# Importing Data Manipulation Libraries
import pandas as pd
import numpy as np

# importing Data visualization libraries

import matplotlib.pyplot as plt
import seaborn as sns

# Importing warinings libraires
import warnings
warnings.filterwarnings("ignore")

# importing loggins 
import logging
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    filename='app.log',
    format='%(asctime)s - %(levelname)s - %(message)s'

)


url="https://raw.githubusercontent.com/MontyVasita18/SupplyChain_BusinessModel/refs/heads/main/SCM.csv"
df=pd.read_csv(url)

numeric_data=df.select_dtypes(exclude='object')

Categorical_data=df.select_dtypes(include='object')

df['workers_num']=df['workers_num'].fillna(df['workers_num'].median())

df.drop(columns='wh_est_year',inplace=True)

df.drop(columns=['WH_Manager_ID','Ware_house_ID'])

df['approved_wh_govt_certificate']=df['approved_wh_govt_certificate'].fillna('C')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['Ware_house_ID']=le.fit_transform(df['Ware_house_ID'])
df['WH_Manager_ID']=le.fit_transform(df['WH_Manager_ID'])
df['Location_type']=le.fit_transform(df['Location_type'])
df['WH_capacity_size']=le.fit_transform(df['WH_capacity_size'])
df['zone']=le.fit_transform(df['zone'])
df['WH_regional_zone']=le.fit_transform(df['WH_regional_zone'])
df['wh_owner_type']=le.fit_transform(df['wh_owner_type'])
df['approved_wh_govt_certificate']=le.fit_transform(df['approved_wh_govt_certificate'])


# Replace Outliers with Median Statergy

for col in df.select_dtypes(include='number').columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_count = outliers.sum()

    if outlier_count > 0:
        replacement = df[col].median()  
        df.loc[outliers, col] = replacement


from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
# Using PCA Concept:

# Step 1: Standardize the data

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

# Step 2: Determine number of components to retain 90% variance

for i in range(1, df.shape[1] + 1):
    pca = PCA(n_components=i)
    pca.fit(X_scaled)
    evr = np.cumsum(pca.explained_variance_ratio_)
    if evr[i - 1] >= 0.90:
        pcs = i
        break



# Step 3: Apply PCA

pca = PCA(n_components=pcs)
pca_data = pca.fit_transform(X_scaled)

# Step 4: Create DataFrame

pca_columns = [f'PC{j+1}' for j in range(pcs)]
pca_df = pd.DataFrame(pca_data, columns=pca_columns)

# Step 5: Join Target Column with PCA:

pca_df = pca_df.join(df['product_wg_ton'], how = 'left')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,minmax_scale,StandardScaler

X=pca_df.drop(columns='product_wg_ton')
y=pca_df["product_wg_ton"]

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.70)


from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Initialize the model
xgb_model = XGBRegressor(random_state=42)

# Fit the model
xgb_model.fit(X_train, y_train)

# Predict on training data
y_pred_xgb = xgb_model.predict(X_train)

# Evaluate using R² score
r2_score_xgb = r2_score(y_train, y_pred_xgb)
print("R² Score:", r2_score_xgb)



