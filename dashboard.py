import argparse

import pandas as pd
import plotly.express as px
import streamlit as st


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CSV data using Streamlit")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file to visualize")
    return parser.parse_args()


def main():
    args = parse_args()

    # Read the CSV file
    df = pd.read_csv(args.csv_path)

    # Streamlit Page Config
    st.set_page_config(page_title="CSV Data Dashboard", layout="wide")

    # Title
    st.title("📊 CSV Data Dashboard")

    # Show dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Filtering Options
    st.sidebar.header("🔍 Filter Options")

    # Get column names
    filter_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if filter_columns:
        selected_column = st.sidebar.selectbox(
            "Select a column to filter", filter_columns
        )
        unique_values = df[selected_column].unique()
        selected_values = st.sidebar.multiselect(
            f"Filter {selected_column}", unique_values
        )

        # Apply filter if values selected
        if selected_values:
            df = df[df[selected_column].isin(selected_values)]

    # Text-based search filter
    search_column = st.sidebar.selectbox("Select a column for text search", df.columns)
    search_text = st.sidebar.text_input(f"Search in {search_column}")

    if search_text:
        df = df[
            df[search_column]
            .astype(str)
            .str.contains(search_text, case=False, na=False)
        ]

    # Display numeric columns distributions
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_columns) > 0:
        st.subheader("📊 Numeric Columns Distribution")
        selected_numeric = st.selectbox(
            "Select a numeric column to visualize", numeric_columns
        )

        fig = px.histogram(
            df,
            x=selected_numeric,
            title=f"Distribution of {selected_numeric}",
            labels={selected_numeric: selected_numeric},
            marginal="box",
        )
        st.plotly_chart(fig)

        # Display summary statistics
        st.subheader("📈 Summary Statistics")
        st.write(df[selected_numeric].describe())

    # Download Filtered Data
    st.subheader("📥 Download Filtered Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")


if __name__ == "__main__":
    main()
