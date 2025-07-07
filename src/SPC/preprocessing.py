import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess(df):
    df['workers_num'] = df['workers_num'].fillna(df['workers_num'].median())
    df.drop(columns='wh_est_year', inplace=True)
    df['approved_wh_govt_certificate'] = df['approved_wh_govt_certificate'].fillna('C')

    # Label Encoding
    le = LabelEncoder()
    for col in ['Ware_house_ID', 'WH_Manager_ID', 'Location_type', 'WH_capacity_size', 
                'zone', 'WH_regional_zone', 'wh_owner_type', 'approved_wh_govt_certificate']:
        df[col] = le.fit_transform(df[col])

    # Handle Outliers with Median Strategy
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        if outliers.sum() > 0:
            df.loc[outliers, col] = df[col].median()

    return df

def apply_pca(df):
    from sklearn.decomposition import PCA

    target = df['product_wg_ton']
    df_features = df.drop(columns=['product_wg_ton'])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_features)

    for i in range(1, df_features.shape[1] + 1):
        pca = PCA(n_components=i)
        pca.fit(X_scaled)
        if np.cumsum(pca.explained_variance_ratio_)[i - 1] >= 0.90:
            pcs = i
            break

    pca = PCA(n_components=pcs)
    X_pca = pca.fit_transform(X_scaled)
    pca_columns = [f'PC{j+1}' for j in range(pcs)]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    pca_df['product_wg_ton'] = target.values
    return pca_df