"""CSV to Markdown converter for LLM-friendly data representation."""

from __future__ import annotations

import csv
from pathlib import Path


def csv_to_markdown(
    file_path: str,
    max_rows: int | None = None,
    *,
    show_row_numbers: bool = False,
) -> str:
    """
    Read a CSV file and convert it to markdown table format.

    This function is optimized for LLM tool use, producing clean markdown
    tables that language models can easily parse and understand.

    Args:
        file_path: Path to the CSV file to read
        max_rows: Maximum number of data rows to include (None = all rows)
        show_row_numbers: Whether to add a row number column

    Returns:
        A markdown-formatted table as a string

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file is empty or malformed

    Example:
        >>> markdown = csv_to_markdown("data.csv", max_rows=10)
        >>> print(markdown)
        | Column1 | Column2 | Column3 |
        |---------|---------|---------|
        | value1  | value2  | value3  |

    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        with path.open(encoding="utf-8") as f:
            reader = csv.reader(f)

            # Read headers
            try:
                headers = next(reader)
            except StopIteration as err:
                raise ValueError("CSV file is empty") from err

            if not headers:
                raise ValueError("CSV file has no headers")

            # Add row number column if requested
            if show_row_numbers:
                headers = ["#", *headers]

            # Build markdown table
            lines = []

            # Header row
            header_row = "| " + " | ".join(headers) + " |"
            lines.append(header_row)

            # Separator row
            separator = "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|"
            lines.append(separator)

            # Data rows
            row_count = 0
            for row in reader:
                if max_rows is not None and row_count >= max_rows:
                    lines.append(
                        f"\n*... {sum(1 for _ in reader) + 1} more rows truncated ...*"
                    )
                    break

                if row:  # Skip empty rows
                    if show_row_numbers:
                        row = [str(row_count + 1), *row]

                    # Escape pipe characters in cell values
                    escaped_row = [str(cell).replace("|", "\\|") for cell in row]
                    data_row = "| " + " | ".join(escaped_row) + " |"
                    lines.append(data_row)
                    row_count += 1

            if row_count == 0:
                raise ValueError("CSV file has no data rows")

            return "\n".join(lines)

    except csv.Error as e:
        raise ValueError(f"Error parsing CSV file: {e}") from e
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}") from e


def csv_to_markdown_with_summary(
    file_path: str,
    max_rows: int | None = 10,
    *,
    show_row_numbers: bool = False,
) -> str:
    """
    Read a CSV file and convert to markdown with a summary header.

    This variant includes metadata about the CSV file, which can be helpful
    for LLMs to understand the data context.

    Args:
        file_path: Path to the CSV file to read
        max_rows: Maximum number of data rows to include (None = all rows)
        show_row_numbers: Whether to add a row number column

    Returns:
        A markdown-formatted table with summary information

    """
    path = Path(file_path)

    # Get row count first
    with path.open(encoding="utf-8") as f:
        total_rows = sum(1 for _ in csv.reader(f)) - 1  # Exclude header

    # Get column count
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        total_columns = len(headers)

    # Build summary
    summary_lines = [
        f"# CSV Data: {path.name}",
        "",
        f"**File:** `{file_path}`",
        f"**Total Rows:** {total_rows}",
        f"**Total Columns:** {total_columns}",
        f"**Columns:** {', '.join(headers)}",
        "",
    ]

    if max_rows and total_rows > max_rows:
        summary_lines.append(f"*Showing first {max_rows} of {total_rows} rows*")
        summary_lines.append("")

    # Get the table
    table = csv_to_markdown(file_path, max_rows, show_row_numbers=show_row_numbers)

    return "\n".join(summary_lines) + "\n" + table
