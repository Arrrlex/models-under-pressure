from typing import cast
import pandas as pd
import plotly.express as px
import streamlit as st

from models_under_pressure.interfaces.prompt import Prompt
from models_under_pressure.config import RunConfig


RUN_ID = "debug"

run_config = RunConfig(run_id=RUN_ID)

# Read the prompts and metadata
annotated_prompts = Prompt.from_jsonl(run_config.prompts_file, run_config.metadata_file)

# Explicitly type the DataFrame
df: pd.DataFrame = pd.DataFrame(
    [{
        **prompt.to_dict(),
        **{f"annotated_{k}" if k == "high_stakes" else k: v 
           for k, v in (prompt.metadata or {}).items()}
    } for prompt in annotated_prompts]
)

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
    selected_column = st.sidebar.selectbox("Select a column to filter", filter_columns)
    unique_values = df[selected_column].unique()
    selected_values = st.sidebar.multiselect(f"Filter {selected_column}", unique_values)

    # Apply filter if values selected
    if selected_values:
        df = cast(pd.DataFrame, df[df[selected_column].isin(selected_values)])

# Text-based search filter
search_column = st.sidebar.selectbox("Select a column for text search", df.columns)
search_text = st.sidebar.text_input(f"Search in {search_column}")

if search_text:
    # Ensure we're working with strings and cast result back to DataFrame
    df = cast(
        pd.DataFrame,
        df[
            df[search_column]
            .astype(str)
            .str.contains(search_text, case=False, na=False)
        ],
    )

# Display Filtered Data
st.subheader("📊 High Stakes Distribution")

if "high_stakes" in df.columns:
    # Original high stakes histogram
    fig = px.histogram(
        df,
        x="high_stakes",
        nbins=2,
        title="Distribution of High Stakes (0/1)",
        labels={"high_stakes": "High Stakes"},
        category_orders={"high_stakes": [0, 1]},
    )
    st.plotly_chart(fig)

    # Add character length histograms
    st.subheader("📏 Prompt Length Distribution by Stakes")

    # Calculate character lengths
    df["char_length"] = df["prompt"].str.len()

    # Create separate dataframes for high and low stakes
    high_stakes_df = df[df["high_stakes"] == 1]
    low_stakes_df = df[df["high_stakes"] == 0]

    # Create subplots for length distributions
    fig = px.histogram(
        df,
        x="char_length",
        color="high_stakes",
        nbins=30,
        title="Character Length Distribution by Stakes",
        labels={"char_length": "Character Length", "high_stakes": "High Stakes"},
        color_discrete_map={1: "red", 0: "blue"},
        marginal="box",  # Adds box plots on the margin
    )

    # Update layout for better readability
    fig.update_layout(
        barmode="overlay",  # Overlapping bars
        # opacity=0.7,        # Make bars semi-transparent
    )

    st.plotly_chart(fig)

    # Display summary statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "High Stakes Avg Length",
            f"{high_stakes_df['char_length'].mean():.0f} chars",
        )
    with col2:
        st.metric(
            "Low Stakes Avg Length", f"{low_stakes_df['char_length'].mean():.0f} chars"
        )

else:
    st.warning("Column 'high_stakes' not found in the dataset!")

# Download Filtered Data
st.subheader("📥 Download Filtered Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")
