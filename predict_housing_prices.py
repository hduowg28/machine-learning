import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, train_test_split,GridSearchCV
import warnings
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
import mlflow
import os

print("Current directory:", os.getcwd())
print("Tracking URI:", mlflow.get_tracking_uri())
warnings.filterwarnings('ignore')
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Ames_Housing_Prices")

class LogSkewedFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=0.75):
        self.skew_threshold = skew_threshold
        self.skewed_features_ = []

    def fit(self, X, y=None):
        # Chỉ tính độ lệch trên các cột số
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        # Tính skewness và lọc các cột vượt ngưỡng
        skewness = X[numeric_cols].apply(lambda col: col.skew()).abs()
        self.skewed_features_ = skewness[skewness > self.skew_threshold].index.tolist()
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.skewed_features_:
            if col in X_out.columns:
                # np.clip để đảm bảo không có số âm trước khi log1p
                X_out[col] = np.log1p(np.clip(X_out[col], a_min=0, a_max=None))
        return X_out

class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        X_out['Overall Qual'] = pd.to_numeric(X_out['Overall Qual'], errors='coerce')
        X_out['Overall Cond'] = pd.to_numeric(X_out['Overall Cond'], errors='coerce')
        X_out['Gr Liv Area'] = pd.to_numeric(X_out['Gr Liv Area'], errors='coerce')
        X_out['Total Bsmt SF'] = pd.to_numeric(X_out['Total Bsmt SF'], errors='coerce')

        # Interaction features
        X_out['Quality_Adjusted_Area'] = X_out['Overall Qual'] * X_out['Gr Liv Area']
        X_out['Total_SqFt'] = X_out['Gr Liv Area'] + X_out['Total Bsmt SF']
        X_out['Overall_Grade'] = X_out['Overall Qual'] + X_out['Overall Cond']
        X_out['Total_Bathrooms'] = (
            X_out.get('Full Bath', 0) + (0.5 * X_out.get('Half Bath', 0)) +
            X_out.get('Bsmt Full Bath', 0) + (0.5 * X_out.get('Bsmt Half Bath', 0))
        )
        X_out['House_Age'] = X_out['Yr Sold'] - X_out['Year Built']
        X_out['Is_Remodeled'] = (X_out['Year Remod/Add'] != X_out['Year Built']).astype(int)

        return X_out

# 2. CLASS: Mã hóa Ordinal
class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict, cols):
        self.mapping_dict = mapping_dict
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].replace(self.mapping_dict)
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce').fillna(0)
        return X_out

# 3. CLASS: OOF Target Encoder
class OOFSmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, m=10.0, n_splits=5):
        self.m = m
        self.n_splits = n_splits
        self.mappings_ = {}
        self.global_mean_ = 0.0

    def fit(self, X, y):
        self.global_mean_ = y.mean()
        df = pd.DataFrame(X).copy()
        df["__target"] = y.values
        for col in df.columns:
            if col == "__target": continue
            stats = df.groupby(col)["__target"].agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + self.m * self.global_mean_) / (stats["count"] + self.m)
            self.mappings_[col] = smoothed.to_dict()
        return self
    def fit_transform(self, X, y):
        self.fit(X, y)
        df = pd.DataFrame(X).copy()
        df["__target"] = y.values
        X_out = pd.DataFrame(X).copy()

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for col in df.columns:
            if col == "__target": continue
            X_out[col] = np.nan
            for train_idx, val_idx in kf.split(df):
                train_fold = df.iloc[train_idx]
                val_fold = df.iloc[val_idx]

                fold_global_mean = train_fold["__target"].mean()
                stats = train_fold.groupby(col)["__target"].agg(["count", "mean"])
                smoothed = (stats["count"] * stats["mean"] + self.m * fold_global_mean) / (stats["count"] + self.m)
                X_out.loc[val_fold.index, col] = val_fold[col].map(smoothed).fillna(fold_global_mean)
        return X_out

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        for col in X_out.columns:
            if col in self.mappings_:
                X_out[col] = X_out[col].map(self.mappings_[col]).fillna(self.global_mean_)
        return X_out
    
class AmesDataImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Học các giá trị thống kê từ tập Train để tránh rò rỉ dữ liệu (Data Leakage)
        self.lot_frontage_median_ = X["Lot Frontage"].median() if "Lot Frontage" in X.columns else 0
        self.garage_cars_median_ = X["Garage Cars"].median() if "Garage Cars" in X.columns else 0
        self.garage_area_median_ = X["Garage Area"].median() if "Garage Area" in X.columns else 0

        self.electrical_mode_ = X["Electrical"].mode()[0] if "Electrical" in X.columns and not X["Electrical"].mode().empty else "SBrkr"
        self.garage_finish_mode_ = X["Garage Finish"].mode()[0] if "Garage Finish" in X.columns and not X["Garage Finish"].mode().empty else "Unf"
        self.garage_qual_mode_ = X["Garage Qual"].mode()[0] if "Garage Qual" in X.columns and not X["Garage Qual"].mode().empty else "TA"
        self.garage_cond_mode_ = X["Garage Cond"].mode()[0] if "Garage Cond" in X.columns and not X["Garage Cond"].mode().empty else "TA"
        self.garage_yr_blt_mode_ = X["Garage Yr Blt"].mode()[0] if "Garage Yr Blt" in X.columns and not X["Garage Yr Blt"].mode().empty else 1975
        return self

    def transform(self, X):
        X_out = X.copy()
        if "Lot Frontage" in X_out.columns:
            X_out["Lot Frontage"] = X_out["Lot Frontage"].fillna(self.lot_frontage_median_)
        if "Alley" in X_out.columns:
            X_out["Alley"] = X_out["Alley"].fillna("noAlley")
        if "Mas Vnr Type" in X_out.columns:
            X_out["Mas Vnr Type"] = X_out["Mas Vnr Type"].fillna("none")
        if "Mas Vnr Area" in X_out.columns:
            X_out["Mas Vnr Area"] = X_out["Mas Vnr Area"].fillna(0)

        bsmt_cols = ["Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2"]
        for col in bsmt_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna("noBsmt")

        for col in ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath"]:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(0)

        if "Electrical" in X_out.columns:
            X_out["Electrical"] = X_out["Electrical"].fillna(self.electrical_mode_)
        if "Fireplace Qu" in X_out.columns:
            X_out["Fireplace Qu"] = X_out["Fireplace Qu"].fillna("NoFireplace")

        # Logic xử lý cụm từ Garage đặc biệt của bạn
        if "Garage Type" in X_out.columns:
            mask = X_out["Garage Type"].isnull()
            garage_cat_cols = ["Garage Type", "Garage Finish", "Garage Qual", "Garage Cond", "Garage Yr Blt"]
            for col in garage_cat_cols:
                if col in X_out.columns:
                    X_out.loc[mask, col] = X_out.loc[mask, col].fillna("NoGarage")

            X_out["Garage Yr Blt"] = X_out["Garage Yr Blt"].replace("NoGarage", -1)
            X_out["Garage Yr Blt"] = pd.to_numeric(X_out["Garage Yr Blt"], errors='coerce')

            X_out["Garage Finish"] = X_out["Garage Finish"].fillna(self.garage_finish_mode_)
            X_out["Garage Qual"] = X_out["Garage Qual"].fillna(self.garage_qual_mode_)
            X_out["Garage Cond"] = X_out["Garage Cond"].fillna(self.garage_cond_mode_)
            X_out["Garage Yr Blt"] = X_out["Garage Yr Blt"].fillna(self.garage_yr_blt_mode_)

        if "Garage Cars" in X_out.columns:
            X_out["Garage Cars"] = X_out["Garage Cars"].fillna(self.garage_cars_median_)
        if "Garage Area" in X_out.columns:
            X_out["Garage Area"] = X_out["Garage Area"].fillna(self.garage_area_median_)

        if "Pool QC" in X_out.columns: X_out["Pool QC"] = X_out["Pool QC"].fillna("NoPool")
        if "Fence" in X_out.columns: X_out["Fence"] = X_out["Fence"].fillna("NoFence")
        if "Misc Feature" in X_out.columns: X_out["Misc Feature"] = X_out["Misc Feature"].fillna("None")

        return X_out

def remove_outliers(df):
    print(f"Kích thước ban đầu: {df.shape}")

    df_clean = df.drop(df[df['Gr Liv Area'] > 4000].index, errors='ignore')

    temp_df = df_clean.copy()

    temp_df['Total_SqFt'] = temp_df['Gr Liv Area'] + temp_df['Total Bsmt SF'].fillna(0)
    temp_df['House_Age'] = temp_df['Yr Sold'] - temp_df['Year Built']

    feature_cols = ['Gr Liv Area', 'Total_SqFt', 'Garage Area', 'Total Bsmt SF', 'Lot Frontage', 'House_Age']
    feature_cols = [c for c in feature_cols if c in df_clean.columns]

    temp_df = df_clean[feature_cols].copy()
    for col in temp_df.columns:
        temp_df[col] = temp_df[col].fillna(temp_df[col].median())

    iso = IsolationForest(contamination=0.01, random_state=42, n_estimators=150)
    labels = iso.fit_predict(temp_df)

    df_clean = df_clean[labels == 1].reset_index(drop=True)
    print(f"Kích thước sau khi lọc Outlier: {df_clean.shape}")
    return df_clean


def train_housing_pipeline(data_path, model_name="Ridge"):
    dataset = pd.read_csv(data_path)
    dataset = remove_outliers(dataset)

    X = dataset.drop(columns=["Order", "PID", "SalePrice"], errors="ignore")
    y = dataset["SalePrice"]
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    ordinal_cols = ["Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC",
                    "Kitchen Qual", "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC"]
    qual_mapping = {"NoPool": 0, "NoFireplace": 0, "noBsmt": 0, "NoGarage": 0,
                    "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    target_enc_cols = ["Neighborhood", "Exterior 1st", "Exterior 2nd"]
    all_categorical = X_train.select_dtypes(include=["object"]).columns.tolist()
    ohe_cols = [c for c in all_categorical if c not in ordinal_cols and c not in target_enc_cols]

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    new_features = ['Quality_Adjusted_Area', 'Total_SqFt', 'Overall_Grade',
                    'Total_Bathrooms', 'House_Age', 'Is_Remodeled']
    all_numeric_cols = numeric_cols + new_features
    all_numeric_cols = [c for c in all_numeric_cols if c not in ordinal_cols]

    numeric_transformer = Pipeline(steps=[
        ('skew_log', LogSkewedFeaturesTransformer(skew_threshold=0.75)),
        ('scaler', StandardScaler()) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', CustomOrdinalEncoder(mapping_dict=qual_mapping, cols=ordinal_cols), ordinal_cols),
            ('target_enc', OOFSmoothedTargetEncoder(m=10.0, n_splits=5), target_enc_cols),
            ('ohe', OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore'), ohe_cols),
            ('numeric', numeric_transformer, all_numeric_cols)
        ],
        remainder='drop'
    )
    if model_name == "LinearRegression":
        steps = [
            ('data_imputer', AmesDataImputer()),
            ('feature_engineering', AmesFeatureEngineer()),
            ('preprocessor', preprocessor),
            ('feature_selection', SelectFromModel(Lasso(alpha=0.005, random_state=42))), 
            ('pca', PCA(n_components=0.95, random_state=42)),
            ('lr', LinearRegression())
        ]
        param_grid = {} 

    elif model_name == "Ridge":
        steps = [
            ('data_imputer', AmesDataImputer()),
            ('feature_engineering', AmesFeatureEngineer()),
            ('preprocessor', preprocessor),
            ('feature_selection', SelectFromModel(Lasso(alpha=0.005, random_state=42))),
            ('pca', PCA(n_components=0.95, random_state=42)),
            ('ridge', Ridge())
        ]
        param_grid = {'ridge__alpha': [0.1, 1.0, 10.0, 20.0]}

    elif model_name == "RandomForest":
        steps = [
            ('data_imputer', AmesDataImputer()),
            ('feature_engineering', AmesFeatureEngineer()),
            ('preprocessor', preprocessor),
            ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
        ]
        param_grid = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [None, 15, 25],
            'rf__min_samples_split': [2, 5]
        }

    full_pipeline = Pipeline(steps=steps)

    with mlflow.start_run(run_name=f"CV_Pipeline_{model_name}"):
        

        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("pipeline_version", "v1.0")

        print(f"\nĐang tinh chỉnh và huấn luyện mô hình {model_name} (5-Fold CV)...")
        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train_log)
        best_model = grid_search.best_estimator_

        # Dự đoán đánh giá
        y_pred_log = best_model.predict(X_test)
        y_test_real = np.expm1(y_test_log)
        y_pred_real = np.expm1(y_pred_log)

        rmse = root_mean_squared_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)

        print(f"--- KẾT QUẢ {model_name} ---")
        print(f"Best Params: {grid_search.best_params_}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R2 Score: {r2:.4f}")

        # --- TỐI ƯU LOGGING MLFLOW ---
        if grid_search.best_params_:
            mlflow.log_params(grid_search.best_params_)
        
        mlflow.log_param("cv_splits", 5)
        mlflow.log_param("outliers_removed", "GrLivArea > 4000 & IsoForest")

        # Log Metrics
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2_score", r2)
        mlflow.log_metric("best_cv_rmse_log", -grid_search.best_score_)
        
        input_example = X_train.iloc[[0]] 
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model_pipeline",
            serialization_format="cloudpickle",
            input_example=input_example
        )

        return best_model

data_path = r"C:/learning/3rd year/2 semerter/machine learning/machine-learning\AmesHousing (1).csv"

pipeline_lr = train_housing_pipeline(data_path, model_name="LinearRegression")
pipeline_ridge = train_housing_pipeline(data_path, model_name="Ridge")
pipeline_rf = train_housing_pipeline(data_path, model_name="RandomForest")