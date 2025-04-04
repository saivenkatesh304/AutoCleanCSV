import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import io

# ----- Enhanced Cleaning Logic -----
class AutoDataCleaner:
    def __init__(self, df, scale_method="standard", impute_strategy="most_frequent", outlier_method="zscore"):
        self.df = df.copy()
        self.scale_method = scale_method
        self.impute_strategy = impute_strategy
        self.outlier_method = outlier_method

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)

    def standardize_text_fields(self):
        obj_cols = self.df.select_dtypes(include=["object", "category"]).columns
        for col in obj_cols:
            self.df[col] = (
                self.df[col].astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
                .str.title()
            )

    def handle_missing(self):
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns

        # Numeric imputation
        if self.impute_strategy == "knn":
            num_imputer = KNNImputer()
        elif self.impute_strategy == "mean":
            num_imputer = SimpleImputer(strategy="mean")
        else:
            num_imputer = SimpleImputer(strategy="most_frequent")
        self.df[num_cols] = num_imputer.fit_transform(self.df[num_cols])

        # Categorical imputation
        cat_imputer = SimpleImputer(strategy="most_frequent")
        self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])

    def handle_outliers(self):
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if self.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(self.df[num_cols]))
            self.df = self.df[(z_scores < 3).all(axis=1)]
        elif self.outlier_method == "iqr":
            Q1 = self.df[num_cols].quantile(0.25)
            Q3 = self.df[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[~((self.df[num_cols] < (Q1 - 1.5 * IQR)) | (self.df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    def format_dates(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except:
                    continue

    def encode_categoricals(self):
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns
        self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)

    def scale_features(self):
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        scaler = StandardScaler() if self.scale_method == "standard" else MinMaxScaler()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])

    def run_all(self):
        initial_shape = self.df.shape
        self.remove_duplicates()
        self.standardize_text_fields()
        self.handle_missing()
        self.handle_outliers()
        self.format_dates()
        self.encode_categoricals()
        self.scale_features()
        final_shape = self.df.shape
        return self.df, initial_shape, final_shape

# ----- Streamlit App -----
st.set_page_config(page_title="Auto Data Cleaner", layout="wide")
st.title("🧹 Automated Data Cleaning and Preprocessing Tool")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df, use_container_width=True)

    with st.sidebar:
        st.header("⚙️ Cleaning Configuration")
        impute_strategy = st.selectbox("Missing Value Strategy", ["most_frequent", "mean", "knn"])
        outlier_method = st.selectbox("Outlier Detection", ["zscore", "iqr"])
        scale_method = st.selectbox("Scaling Method", ["standard", "minmax"])
        clean_button = st.button("🚀 Run Cleaner")

    if clean_button:
        cleaner = AutoDataCleaner(df, scale_method=scale_method,
                                  impute_strategy=impute_strategy,
                                  outlier_method=outlier_method)
        cleaned_df, before_shape, after_shape = cleaner.run_all()

        st.success("✅ Data cleaned successfully!")
        st.markdown(f"**Rows Before Cleaning:** {before_shape[0]}  \n**Rows After Cleaning:** {after_shape[0]}")
        st.subheader("✅ Cleaned Data Preview")
        st.dataframe(cleaned_df, use_container_width=True)

        # Download button
        csv = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned Data", data=csv, file_name="cleaned_data.csv", mime="text/csv")
else:
    st.info("👆 Upload a CSV file to get started.")
