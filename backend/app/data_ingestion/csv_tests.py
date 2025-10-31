# ruff: noqa: T201
"""
Example usage of csv_to_markdown for LLM tool integration.

This demonstrates how to use the CSV to markdown converter
as part of an LLM tool that needs to present CSV data.
"""

from csv_analyzer import summarize_csv
from csv_loader import csv_to_markdown, csv_to_markdown_with_summary

if __name__ == "__main__":
    # Example 1: Load all data for detailed analysis
    print("Example 1: Full data load")
    print("=" * 80)
    full_data = csv_to_markdown("mock/showcase_financial_pl_data.csv")
    print(full_data)
    print("\n... (full data would be passed to LLM)\n")

    # Example 2: Quick summary for initial analysis
    print("\nExample 2: Quick summary (first 5 rows)")
    print("=" * 80)
    summary = csv_to_markdown_with_summary("mock/showcase_financial_pl_data.csv")
    print(summary)

    # Example 3: Comprehensive analysis
    print("\nExample 3: Comprehensive analysis")
    print("=" * 80)
    summary = summarize_csv("mock/showcase_financial_pl_data.csv")
    print(summary)
