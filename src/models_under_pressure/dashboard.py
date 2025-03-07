"""
Run the dashboard with:

```
uv run dash <path_to_dataset>
```
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import typer

from models_under_pressure.interfaces.dataset import Label


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


def setup_row_analyzer_controls(
    df_display: pd.DataFrame,
) -> tuple[int, str, str | None, bool]:
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
    """Display token-level visualization for text with probe values using a HuggingFace tokenizer."""
    if probe_column is None or not isinstance(text_value, str):
        return

    st.subheader("Token-level Visualization")

    # Get the probe values
    probe_values = selected_row[probe_column]

    # Handle different formats of probe values
    if isinstance(probe_values, (list, np.ndarray)):
        # Already in the correct format
        pass
    elif isinstance(probe_values, str):
        # Try to parse the string as a list
        try:
            probe_values = json.loads(probe_values)

        except Exception as e:
            st.warning(f"Could not parse probe values as a list: {str(e)}")
            st.write("Raw probe values:", probe_values)
            return
    else:
        st.warning(f"Unexpected type for probe values: {type(probe_values)}")
        st.write("Raw probe values:", probe_values)
        return

    # Import the tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer_name = st.selectbox(
            "Select tokenizer",
            [
                "meta-llama/Llama-3.2-1B-Instruct",
            ],
            index=0,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Check if text value can be json loaded then tokenize with apply_chat_
        try:
            print("Tokenizing json loaded messages")
            tokens = tokenizer.apply_chat_template(
                json.loads(text_value), tokenize=True, add_generation_prompt=True
            )

        except Exception as e:
            st.error(f"Error tokenizing text: {str(e)}, applying standard tokenization")
            tokens = tokenizer.tokenize(text_value)

        # If we have valid probe values that match the number of tokens
        if isinstance(probe_values, (list, np.ndarray)) and len(tokens) == len(
            probe_values
        ):
            # Create a visualization of colored tokens
            min_val = min(probe_values)
            max_val = max(probe_values)
            range_val = max_val - min_val if max_val != min_val else 1

            # Function to get color for a value
            def get_color(val: float) -> str:
                # Normalize to 0-1 scale
                normalized = (val - min_val) / range_val
                # Use a color scale from blue (low) to red (high)
                return f"rgb({int(255 * normalized)}, 0, {int(255 * (1 - normalized))})"

            # Build HTML for colored tokens
            html = "<div style='font-size: 18px; line-height: 2;'>"
            for token, val in zip(tokens, probe_values):
                color = get_color(val)

                # Convert token ID to display text
                if isinstance(token, int):
                    # For integer token IDs, convert to string representation
                    display_token = tokenizer.decode([token], skip_special_tokens=False)
                else:
                    # For string tokens, handle special characters
                    display_token = str(token).replace("Ġ", " ").replace("▁", " ")

                html += f"<span style='background-color: {color}; padding: 3px; margin: 2px; border-radius: 3px;'>{display_token}</span>"
            html += "</div>"

            # Display colored tokens
            st.markdown(html, unsafe_allow_html=True)

            # Display original tokens and their values in a table
            with st.expander("Show token values"):
                token_df = pd.DataFrame({"Token": tokens, "Value": probe_values})
                st.dataframe(token_df)
        else:
            st.warning(
                f"The number of tokens ({len(tokens)}) doesn't match the number of probe values ({len(probe_values) if isinstance(probe_values, (list, np.ndarray)) else 'unknown'})."
            )

            # Show tokens anyway for debugging
            st.write("Tokens:", tokens)
            st.write("Number of tokens:", len(tokens))
            st.write(
                "Number of probe values:",
                len(probe_values)
                if isinstance(probe_values, (list, np.ndarray))
                else "unknown format",
            )

    except ImportError:
        st.error(
            "Please install the transformers library to use token-level visualization: `pip install transformers`"
        )
    except Exception as e:
        st.error(f"Error in token visualization: {str(e)}")
        st.exception(e)


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


def display_model_evaluation_curves(data: DashboardDataset) -> None:
    """Display precision-recall or ROC curves based on probe scores and ground truth."""
    st.subheader("🎯 Model Evaluation Curves")

    # Get columns that might contain probe scores (logits/probs)
    probe_columns = [
        col
        for col in data.data.columns
        if pd.api.types.is_numeric_dtype(data.data[col])
    ]

    # Get columns that might contain ground truth labels
    all_columns = data.data.columns.tolist()

    if not probe_columns:
        st.info(
            "No probe score columns detected. Need columns with 'logit' or 'prob' in their name."
        )
        return

    # Create selection widgets
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_probe = st.selectbox(
            "Select probe score column:", probe_columns, key="eval_probe_col"
        )

    with col2:
        selected_truth = st.selectbox(
            "Select ground truth column:", all_columns, key="eval_truth_col"
        )

    with col3:
        chart_type = st.selectbox(
            "Select chart type:",
            ["Precision-Recall Curve", "ROC Curve"],
            key="eval_chart_type",
        )

    # Button to generate the chart
    if st.button("Generate Evaluation Curve"):
        try:
            assert all(
                x in {"high-stakes", "low-stakes", 0, 1}
                for x in data.data[selected_truth].unique()
            ), "Ground truth must be binary"

            y_true = [
                label if isinstance(label, int) else Label(label).to_int()
                for label in data.data[selected_truth]
            ]

            # Get probe scores and convert to probabilities using sigmoid if they're logits
            probe_scores = data.data[selected_probe].tolist()

            # Check if we need to apply sigmoid (if these are logits)
            if "logit" in selected_probe.lower():
                import scipy.special

                y_scores = scipy.special.expit(probe_scores)  # sigmoid function
            else:
                y_scores = probe_scores

            # Generate the appropriate curve
            import plotly.graph_objects as go
            from sklearn.metrics import auc, precision_recall_curve, roc_curve

            if chart_type == "Precision-Recall Curve":
                precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

                # Calculate average precision
                from sklearn.metrics import average_precision_score

                ap = average_precision_score(y_true, y_scores)

                # Create the plot
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=recall,
                        y=precision,
                        mode="lines",
                        name=f"Precision-Recall (AP={ap:.3f})",
                    )
                )
                fig.update_layout(
                    title="Precision-Recall Curve",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    yaxis=dict(range=[0, 1.05]),
                    xaxis=dict(range=[0, 1.05]),
                )

            else:  # ROC Curve
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)

                # Create the plot
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"
                    )
                )

                # Add diagonal line (random classifier)
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        line=dict(dash="dash", color="gray"),
                        name="Random",
                    )
                )

                fig.update_layout(
                    title="Receiver Operating Characteristic (ROC) Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    yaxis=dict(range=[0, 1.05]),
                    xaxis=dict(range=[0, 1.05]),
                )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating curve: {str(e)}")
            st.exception(e)


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

    # Display model evaluation curves
    display_model_evaluation_curves(data)

    # # Display confusion matrix
    # display_confusion_matrix(data)

    # Add download button
    add_download_button(data)


if __name__ == "__main__":
    typer.run(main)
