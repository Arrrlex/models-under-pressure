import argparse
import os

import pandas as pd
import plotly.express as px
import streamlit as st


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Data using Streamlit")
    parser.add_argument("file_path", type=str, help="Path to the dataset to visualize")
    return parser.parse_args()


class DashboardDataset:
    def __init__(self, dataset: pd.DataFrame):
        """
        Pandas DataFrame wrapper class for additional functionality that may be required for the dashboard.
        """

        self.dataset = dataset

    @classmethod
    def load_from(
        cls,
        file_path: str,
    ) -> "DashboardDataset":
        # Infer the file type from the file extension
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == ".csv":
            return cls(dataset=pd.read_csv(file_path))
        elif file_extension == ".jsonl":
            return cls(dataset=pd.read_json(file_path, lines=True))

    @property
    def data(self) -> pd.DataFrame:
        return self.dataset


def main():
    # Parse arguments
    args = parse_args()

    # Load the dataset
    data = DashboardDataset.load_from(args.file_path)
    df_display = data.data

    # Streamlit Page Config
    st.set_page_config(page_title="Data Dashboard", layout="wide")

    # Title
    st.title("📊Data Dashboard")

    # Show dataset preview
    st.subheader("Dataset")

    # Filtering Options
    st.sidebar.header("🔍 Filter Options")

    # Get column names
    data_columns = df_display.columns.tolist()
    filter_columns = []
    for col in data_columns:
        try:
            if df_display[col].nunique() < 20:
                filter_columns.append(col)
        except TypeError as e:
            print(e)
            continue

    if len(filter_columns) > 0:
        selected_column = st.sidebar.selectbox(
            "Select a column to filter", filter_columns
        )
        unique_values = df_display[selected_column].unique()
        selected_values = st.sidebar.multiselect(
            f"Filter {selected_column}", unique_values
        )

        # Apply filter if values selected
        if selected_values:
            df_display = df_display[df_display[selected_column].isin(selected_values)]

    # Text-based search filter
    search_column = st.sidebar.selectbox(
        "Select a column for text search", df_display.columns
    )
    search_text = st.sidebar.text_input(f"Search in {search_column}")

    if search_text:
        df_display = df_display[
            df_display[search_column]
            .astype(str)
            .str.contains(search_text, case=False, na=False)
        ]

    # Display the dataset
    st.dataframe(df_display)

    # Display numeric columns distributions
    numeric_columns = data.data.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_columns) > 0:
        st.subheader("📊 Numeric Columns Distribution")
        selected_numeric = st.selectbox(
            "Select a numeric column to visualize", numeric_columns
        )

        fig = px.histogram(
            data.data,
            x=selected_numeric,
            title=f"Distribution of {selected_numeric}",
            labels={selected_numeric: selected_numeric},
            marginal="box",
        )
        st.plotly_chart(fig)

        # Display summary statistics
        st.subheader("📈 Summary Statistics")
        st.write(data.data[selected_numeric].describe())

    # Download Filtered Data
    st.subheader("📥 Download Filtered Data")
    csv = data.data.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")


if __name__ == "__main__":
    main()
