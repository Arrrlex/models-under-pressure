import argparse
import os

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Data using Streamlit")
    parser.add_argument("file_path", type=str, help="Path to the dataset to visualize")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
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
        debug: bool = False,
    ) -> "DashboardDataset":
        # Infer the file type from the file extension
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == ".csv":
            data = cls(dataset=pd.read_csv(file_path))
        elif file_extension == ".jsonl":
            data = cls(dataset=pd.read_json(file_path, lines=True))
        else:
            raise NotImplementedError(f"File type {file_extension} not supported")

        if debug:

            def create_probe_logits(prompt: str) -> np.ndarray:
                # Split the prompt into words:
                words = prompt.split(" ")

                return 2 * np.zeros(len(words))

            # Load the probe logits
            probe_logits = pd.read_csv(
                os.path.join(os.path.dirname(file_path), "probe_logits.csv")
            )
            data["probe_logits"] = probe_logits

        return data

    @property
    def data(self) -> pd.DataFrame:
        return self.dataset


def main():
    # Parse arguments
    args = parse_args()

    # Load the dataset
    data = DashboardDataset.load_from(args.file_path, debug=args.debug)
    df_display = data.data

    # Streamlit Page Config
    st.set_page_config(page_title="Data Dashboard", layout="wide")

    # Title
    st.title("📊Data Dashboard")

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

    # Analyze Row feature controls in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analyze Row")

    # Create a way to select rows
    selected_index = st.sidebar.number_input(
        "Enter row number to analyze:",
        min_value=0,
        max_value=len(df_display) - 1 if len(df_display) > 0 else 0,
        value=0,
    )

    # Add column selection dropdown
    all_columns = df_display.columns.tolist()
    selected_column_to_analyze = st.sidebar.selectbox(
        "Select column to analyze:", all_columns
    )

    analyze_button = st.sidebar.button("Analyze Selected Row")

    # Display the dataset
    st.subheader("Dataset")
    st.dataframe(df_display, use_container_width=True)

    # If a row is selected, show detailed information in the main area
    if analyze_button and len(df_display) > 0:
        st.subheader("Selected Row Analysis")

        # Get the selected row by index
        selected_row = df_display.iloc[selected_index]

        # Only display the selected column
        st.markdown(f"**Row {selected_index}, Column: {selected_column_to_analyze}**")

        # Format the display based on the data type
        value = selected_row[selected_column_to_analyze]

        # Determine if it's a text column
        if df_display[selected_column_to_analyze].dtype == "object":
            st.info(str(value))
        else:
            # For numeric values, provide a bit more context
            st.info(f"Value: {value}")

            # If it's numeric, add some statistics for context
            if pd.api.types.is_numeric_dtype(df_display[selected_column_to_analyze]):
                col_mean = df_display[selected_column_to_analyze].mean()
                col_std = df_display[selected_column_to_analyze].std()
                percentile = (
                    df_display[selected_column_to_analyze] <= value
                ).mean() * 100

                st.markdown(f"Dataset mean for this column: **{col_mean:.2f}**")
                st.markdown(f"Dataset standard deviation: **{col_std:.2f}**")
                st.markdown(f"This value is in the **{percentile:.1f}th** percentile")

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
