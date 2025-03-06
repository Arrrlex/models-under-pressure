"""
Run the dashboard with:

```
uv run dash <path_to_dataset>
```
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import typer


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

            data.data["probe_logits"] = data.data["prompt"].apply(create_probe_logits)

        return data

    @property
    def data(self) -> pd.DataFrame:
        return self.dataset


def setup_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(page_title="Data Dashboard", layout="wide")
    st.title("📊Data Dashboard")


def create_sidebar_filters(df_display: pd.DataFrame) -> pd.DataFrame:
    """Create and apply sidebar filters to the dataframe."""
    st.sidebar.header("🔍 Filter Options")

    # Get column names for filtering
    data_columns = df_display.columns.tolist()
    filter_columns = []
    for col in data_columns:
        try:
            if df_display[col].nunique() < 20:
                filter_columns.append(col)
        except TypeError as e:
            print(e)
            continue

    # Apply category filter if possible
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

    return df_display


def setup_row_analyzer_controls(df_display: pd.DataFrame) -> tuple[int, str, str, bool]:
    """Set up the controls for analyzing individual rows."""
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

    # Add probe logits/probs column selection
    probe_columns = [
        col for col in all_columns if "logit" in col.lower() or "prob" in col.lower()
    ]
    probe_column = (
        st.sidebar.selectbox(
            "Select column of probe logits/probs:", probe_columns, key="probe_column"
        )
        if probe_columns
        else None
    )

    analyze_button = st.sidebar.button("Analyze Selected Row")

    return selected_index, selected_column_to_analyze, probe_column, analyze_button


def display_row_analysis(
    df_display: pd.DataFrame,
    selected_index: int,
    selected_column_to_analyze: str,
    probe_column: str,
) -> None:
    """Display detailed analysis for the selected row."""
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
        display_word_level_visualization(value, selected_row, probe_column)
    else:
        # For numeric values, provide a bit more context
        st.info(f"Value: {value}")
        display_numeric_analysis(df_display, selected_column_to_analyze, value)


def display_word_level_visualization(
    text_value: str, selected_row: pd.Series, probe_column: str
) -> None:
    """Display word-level visualization for text with probe values."""
    if probe_column is None or not isinstance(text_value, str):
        return

    st.subheader("Word-level Visualization")

    # Get the probe values
    probe_values = selected_row[probe_column]

    # Check if probe values is a string representation of a list
    if (
        isinstance(probe_values, str)
        and probe_values.startswith("[")
        and probe_values.endswith("]")
    ):
        try:
            probe_values = eval(probe_values)
        except (ValueError, SyntaxError):
            st.warning("Could not parse probe values as a list.")
            return

    # Split the text into words
    words = text_value.split()

    # If we have valid probe values that match the number of words
    if isinstance(probe_values, (list, np.ndarray)) and len(words) == len(probe_values):
        # Create a visualization of colored words
        min_val = min(probe_values)
        max_val = max(probe_values)
        range_val = max_val - min_val if max_val != min_val else 1

        # Function to get color for a value
        def get_color(val):
            # Normalize to 0-1 scale
            normalized = (val - min_val) / range_val
            # Use a color scale from blue (low) to red (high)
            return f"rgb({int(255 * normalized)}, 0, {int(255 * (1 - normalized))})"

        # Build HTML for colored words
        html = "<div style='font-size: 18px; line-height: 2;'>"
        for word, val in zip(words, probe_values):
            color = get_color(val)
            html += f"<span style='background-color: {color}; padding: 3px; margin: 2px; border-radius: 3px;'>{word}</span> "
        html += "</div>"

        # Display colored words
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.warning(
            f"The number of words ({len(words)}) doesn't match the number of probe values or probe values are not in the correct format."
        )


def display_numeric_analysis(
    df_display: pd.DataFrame, column: str, value: float
) -> None:
    """Display statistical analysis for numeric values."""
    if pd.api.types.is_numeric_dtype(df_display[column]):
        col_mean = df_display[column].mean()
        col_std = df_display[column].std()
        percentile = (df_display[column] <= value).mean() * 100

        st.markdown(f"Dataset mean for this column: **{col_mean:.2f}**")
        st.markdown(f"Dataset standard deviation: **{col_std:.2f}**")
        st.markdown(f"This value is in the **{percentile:.1f}th** percentile")


def display_numeric_distributions(data: DashboardDataset) -> None:
    """Display distributions for numeric columns."""
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


def display_confusion_matrix(data: DashboardDataset) -> None:
    """Display confusion matrix for categorical columns."""
    st.subheader("🔄 Confusion Matrix")

    # Get categorical columns (or columns with few unique values)
    categorical_columns = []
    for col in data.data.columns:
        try:
            if data.data[col].nunique() < 20:
                categorical_columns.append(col)
        except TypeError:
            continue

    if len(categorical_columns) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            conf_col1 = st.selectbox(
                "Select first column:", categorical_columns, key="conf_col1"
            )
        with col2:
            conf_col2 = st.selectbox(
                "Select second column:",
                categorical_columns,
                key="conf_col2",
                index=min(1, len(categorical_columns) - 1),
            )

        if st.button("Generate Confusion Matrix"):
            # Create confusion matrix with raw counts
            raw_counts = pd.crosstab(data.data[conf_col1], data.data[conf_col2])

            # Plot heatmap with text annotations
            fig = px.imshow(
                raw_counts,
                labels=dict(x=conf_col2, y=conf_col1, color="Count"),
                x=raw_counts.columns,
                y=raw_counts.index,
                color_continuous_scale="Blues",
                title=f"Confusion Matrix: {conf_col1} vs {conf_col2}",
            )

            # Add text annotations with the count values
            for i in range(len(raw_counts.index)):
                for j in range(len(raw_counts.columns)):
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=str(raw_counts.iloc[i, j]),
                        showarrow=False,
                        font=dict(
                            color="white"
                            if raw_counts.iloc[i, j] > raw_counts.values.max() / 3
                            else "black",
                            size=14,
                            family="Arial",
                        ),
                    )

            # Improve layout
            fig.update_layout(
                xaxis_title=conf_col2,
                yaxis_title=conf_col1,
                height=500,
                coloraxis_showscale=True,
                margin=dict(l=40, r=40, t=50, b=40),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display the normalized values as well
            with st.expander("Show proportions"):
                normalized_df = pd.crosstab(
                    data.data[conf_col1],
                    data.data[conf_col2],
                    normalize="index",  # Normalize by rows
                )
                st.write(normalized_df)
    else:
        st.info(
            "Need at least 2 categorical columns (with fewer than 20 unique values) to create a confusion matrix."
        )


def add_download_button(data: DashboardDataset) -> None:
    """Add a download button for the filtered data."""
    st.subheader("📥 Download Filtered Data")
    csv = data.data.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")


def main(
    file_path: str = typer.Argument(..., help="Path to the dataset to visualize"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
):
    # Load the dataset
    data = DashboardDataset.load_from(file_path, debug=debug)

    # Setup page
    setup_page()

    # Apply filters
    df_display = create_sidebar_filters(data.data)

    # Setup row analyzer controls
    selected_index, selected_column, probe_column, analyze_button = (
        setup_row_analyzer_controls(df_display)
    )

    # Display the dataset
    st.subheader("Dataset")
    st.dataframe(df_display, use_container_width=True)

    # Display row analysis if requested
    if analyze_button and len(df_display) > 0:
        display_row_analysis(df_display, selected_index, selected_column, probe_column)

    # Display numeric distributions
    display_numeric_distributions(data)

    # Display confusion matrix
    display_confusion_matrix(data)

    # Add download button
    add_download_button(data)


if __name__ == "__main__":
    typer.run(main)
