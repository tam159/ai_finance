"""
Module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Any, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langchain_tavily import TavilySearch

from app.data_ingestion.csv_analyzer import summarize_csv
from app.data_ingestion.csv_loader import csv_to_markdown
from app.finance_agent.configuration import Configuration


class FinancialDataType(StrEnum):
    """Finalcial Data Type."""

    PNL = "PNL"


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> list[dict[str, Any]] | None:
    """
    Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    tool = TavilySearch(max_results=configuration.max_search_results)
    result = await tool.ainvoke({"query": query})
    return cast("list[dict[str, Any]]", result)


@tool(parse_docstring=True)
def retrieve_financial_data(  # noqa: D417
    data_type: FinancialDataType = FinancialDataType.PNL,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """
    Retrieve financial data.

    Args:
        data_type: option is PNL.

    Returns:
        Full data set.

    """
    configuration = Configuration.from_runnable_config(config)
    if configuration.role == "accounting_manager":
        if data_type == FinancialDataType.PNL:
            return csv_to_markdown(
                "app/data_ingestion/mock/showcase_financial_pl_data.csv"
            )
        return "Invalid data type"
    return "You are not authorized to access this data. Contact IT team for support."


@tool(parse_docstring=True)
def summarize_financial_data(  # noqa: D417
    data_type: FinancialDataType = FinancialDataType.PNL,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """
    Summarize financial data.

    Args:
        data_type: option is PNL.

    Returns:
        Full data set.

    """
    configuration = Configuration.from_runnable_config(config)
    if configuration.role == "accounting_manager":
        if data_type == FinancialDataType.PNL:
            return summarize_csv(
                "app/data_ingestion/mock/showcase_financial_pl_data.csv"
            )
        return "Invalid data type"
    return "You are not authorized to access this data. Contact IT team for support."


TOOLS: list[Callable[..., Any]] = [retrieve_financial_data, summarize_financial_data]
