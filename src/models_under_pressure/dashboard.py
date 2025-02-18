import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit Page Config
st.set_page_config(page_title="CSV Data Dashboard", layout="wide")

# Title
st.title("📊 CSV Data Dashboard")

# File Uploader
file_path = "../../dataset.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)


# Show dataset preview
st.subheader("Dataset Preview")
st.dataframe(df)

# Filtering Options
st.sidebar.header("🔍 Filter Options")

# Get column names
filter_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

if filter_columns:
    selected_column = st.sidebar.selectbox("Select a column to filter", filter_columns)
    unique_values = df[selected_column].unique()
    selected_values = st.sidebar.multiselect(f"Filter {selected_column}", unique_values)

    # Apply filter if values selected
    if selected_values:
        df = df[df[selected_column].isin(selected_values)]

# Text-based search filter
search_column = st.sidebar.selectbox("Select a column for text search", df.columns)
search_text = st.sidebar.text_input(f"Search in {search_column}")

if search_text:
    df = df[
        df[search_column].astype(str).str.contains(search_text, case=False, na=False)
    ]

# Display Filtered Data
st.subheader("📊 High Stakes Distribution")

if "high_stakes" in df.columns:
    fig = px.histogram(
        df,
        x="high_stakes",
        nbins=2,
        title="Distribution of High Stakes (0/1)",
        labels={"high_stakes": "High Stakes"},
        category_orders={"high_stakes": [0, 1]},
    )
    st.plotly_chart(fig)
else:
    st.warning("Column 'high_stakes' not found in the dataset!")
# Download Filtered Data
st.subheader("📥 Download Filtered Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")
