import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import warnings
import io
import time
from datetime import datetime

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import AutoClean
from autoclean import AutoClean

# Fix Arrow Compatibility for Streamlit display
def make_arrow_compatible(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype('int64', errors='ignore')
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

# Page configuration
st.set_page_config(
    page_title="AutoClean CSV Processor",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { max-width: 100%; padding: 2rem; }
    .sidebar .sidebar-content { padding: 1rem; }
    .stButton>button, .stDownloadButton>button {
        width: 100%; padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🧹 AutoClean CSV Processor")
st.markdown("""
Upload your CSV file and customize the cleaning parameters below. 
The app will process your data using the AutoClean library and provide a cleaned version for download.
""")

# Sidebar for file upload and options
with st.sidebar:
    st.header("1. Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    st.header("2. Processing Options")
    mode = st.radio("Cleaning Mode", ["auto", "manual"], index=0)

    if mode == "manual":
        st.subheader("Manual Cleaning Options")
        duplicates = "auto" if st.checkbox("Remove duplicates", value=True) else False

        missing_num = st.selectbox(
            "Numerical missing values",
            [False, "auto", "linreg", "knn", "mean", "median", "most_frequent", "delete"],
            index=1
        )
        missing_categ = st.selectbox(
            "Categorical missing values",
            [False, "auto", "logreg", "knn", "most_frequent", "delete"],
            index=1
        )
        outliers = st.selectbox(
            "Outlier handling",
            [False, "auto", "winz", "delete"],
            index=2
        )
        outlier_param = st.number_input(
            "Outlier parameter (IQR multiplier)",
            min_value=0.1,
            max_value=10.0,
            value=1.5,
            step=0.1
        ) if outliers else 1.5

        encode_options = st.multiselect(
            "Categorical encoding",
            ["auto", "onehot", "label"],
            default=["auto"]
        )

        extract_datetime = st.selectbox(
            "Datetime extraction",
            [False, "auto", "D", "M", "Y", "h", "m", "s"],
            index=1
        )
    else:
        duplicates = "auto"
        missing_num = "auto"
        missing_categ = "auto"
        outliers = "winz"
        outlier_param = 1.5
        encode_options = ["auto"]
        extract_datetime = "s"

    st.header("3. Logging Options")
    logfile = st.checkbox("Create log file", value=True)
    verbose = st.checkbox("Show detailed logs", value=True)

# Main content area
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()  # Clean up column names
        df = df.convert_dtypes()
        st.success("File successfully uploaded!")

        with st.expander("Preview original data"):
            st.dataframe(make_arrow_compatible(df.head()))
            st.write(f"Shape: {df.shape}")

        st.header("Data Cleaning Process")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Data Summary")
            st.write("**Missing values:**")
            missing_df = pd.DataFrame(df.isna().sum(), columns=["Missing Values"])
            st.dataframe(make_arrow_compatible(missing_df))

            st.write("**Data types:**")
            dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
            dtype_df["Data Type"] = dtype_df["Data Type"].astype(str)
            st.dataframe(make_arrow_compatible(dtype_df))

        if st.button("Clean Data", type="primary"):
            with st.spinner("Cleaning your data. Please wait..."):
                start_time = time.time()
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("Initializing AutoClean...")
                    progress_bar.progress(10)

                    encode_categ = (
                        "auto" if "auto" in encode_options else encode_options
                    ) if encode_options else False

                    status_text.text("Processing data...")
                    progress_bar.progress(40)

                    pipeline = AutoClean(
                        df,
                        mode=mode,
                        duplicates=duplicates,
                        missing_num=missing_num,
                        missing_categ=missing_categ,
                        encode_categ=encode_categ,
                        extract_datetime=extract_datetime,
                        outliers=outliers,
                        outlier_param=outlier_param,
                        logfile=logfile,
                        verbose=verbose
                    )

                    cleaned_df = pipeline.output
                    progress_bar.progress(80)

                    with col2:
                        st.subheader("Cleaned Data Summary")
                        cleaned_missing = pd.DataFrame(cleaned_df.isna().sum(), columns=["Missing Values"])
                        st.write("**Missing values after cleaning:**")
                        st.dataframe(make_arrow_compatible(cleaned_missing))

                        cleaned_dtypes = pd.DataFrame(cleaned_df.dtypes, columns=["Data Type"])
                        cleaned_dtypes["Data Type"] = cleaned_dtypes["Data Type"].astype(str)
                        st.write("**Data types after cleaning:**")
                        st.dataframe(make_arrow_compatible(cleaned_dtypes))

                    with st.expander("Preview cleaned data"):
                        st.dataframe(make_arrow_compatible(cleaned_df.head()))
                        st.write(f"Shape: {cleaned_df.shape}")

                    # Debugging info
                    st.subheader("🛠 Debug: Null Values in Cleaned Data")
                    st.write(cleaned_df.isnull().sum())

                    st.header("Download Cleaned Data")
                    csv = cleaned_df.to_csv(index=False).encode('utf-8')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_filename = f"cleaned_data_{timestamp}.csv"

                    st.download_button(
                        label="Download cleaned CSV",
                        data=csv,
                        file_name=download_filename,
                        mime="text/csv"
                    )

                    progress_bar.progress(100)
                    status_text.text(f"Cleaning completed in {time.time() - start_time:.2f} seconds!")
                    st.success("Data cleaning completed successfully!")

                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    progress_bar.progress(0)
                    status_text.text("Processing failed")

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin processing.")

# Help & Guide
with st.expander("How to use this app"):
    st.markdown("""
    ### AutoClean CSV Processor Guide

    1. **Upload your CSV file** using the uploader in the sidebar.
    2. **Select cleaning mode**:
       - *Auto mode*: The app will automatically handle all cleaning steps.
       - *Manual mode*: You can customize each step.
    3. **Adjust parameters** in the sidebar.
    4. **Click 'Clean Data'** to process your file.
    5. **Download the cleaned data** when processing completes.

    ### Options Explained

    - **Duplicates**: Remove exact duplicate rows.
    - **Missing Values**:
      - Numerical: Fill using KNN, regression, mean, median, etc.
      - Categorical: Fill using mode, logistic regression, etc.
    - **Outliers**: Handle using Winsorization or deletion.
    - **Categorical Encoding**: Transform text to numbers using label or one-hot encoding.
    - **Datetime Extraction**: Extract parts (day, month, etc.) from datetime columns.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using [AutoClean](https://github.com/elisemercury/AutoClean) and [Streamlit](https://streamlit.io/)")
