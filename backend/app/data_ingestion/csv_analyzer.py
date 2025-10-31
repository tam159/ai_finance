"""Analyze csv file."""

import pandas as pd


def summarize_csv(file_path: str) -> str:
    """
    Comprehensively analyzes a CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        str: Formatted comprehensive analysis of the dataset

    """
    df = pd.read_csv(file_path)
    summary = []

    # Basic info
    summary.append("=" * 60)
    summary.append("📊 DATA OVERVIEW")
    summary.append("=" * 60)
    summary.append(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    summary.append(f"\nColumns: {', '.join(df.columns.tolist())}")

    # Data types
    summary.append("\n📋 DATA TYPES:")
    for col, dtype in df.dtypes.items():
        summary.append(f"  • {col}: {dtype}")

    # Missing data analysis
    missing = df.isnull().sum().sum()  # type: ignore # noqa: PGH003, PD003
    missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
    summary.append("\n🔍 DATA QUALITY:")
    if missing:
        summary.append(
            f"Missing values: {missing:,} ({missing_pct:.2f}% of total data)"
        )
        summary.append("Missing by column:")
        for col in df.columns:
            col_missing = df[col].isnull().sum()  # type: ignore # noqa: PGH003, PD003
            if col_missing > 0:
                col_pct = (col_missing / len(df)) * 100
                summary.append(f"  • {col}: {col_missing:,} ({col_pct:.1f}%)")
    else:
        summary.append("✓ No missing values - dataset is complete!")

    # Numeric analysis
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        summary.append("\n📈 NUMERICAL ANALYSIS:")
        summary.append(str(df[numeric_cols].describe()))

        # Correlations if multiple numeric columns
        if len(numeric_cols) > 1:
            summary.append("\n🔗 CORRELATIONS:")
            corr_matrix = df[numeric_cols].corr()
            summary.append(str(corr_matrix))

    # Categorical analysis
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if "id" not in c.lower()]

    if categorical_cols:
        summary.append("\n📊 CATEGORICAL ANALYSIS:")
        for col in categorical_cols[:5]:  # Limit to first 5
            value_counts = df[col].value_counts()
            summary.append(f"\n{col}:")
            for val, count in value_counts.head(10).items():
                pct = (count / len(df)) * 100
                summary.append(f"  • {val}: {count:,} ({pct:.1f}%)")

    # Time series analysis
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        summary.append("\n📅 TIME SERIES ANALYSIS:")
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        date_range = df[date_col].max() - df[date_col].min()
        summary.append(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        summary.append(f"Span: {date_range.days} days")

    summary.append("\n" + "=" * 60)
    summary.append("✅ COMPREHENSIVE ANALYSIS COMPLETE")
    summary.append("=" * 60)

    return "\n".join(summary)
