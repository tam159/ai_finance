"""
Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from app.finance_agent.configuration import Configuration
from app.finance_agent.state import InputState, State
from app.finance_agent.tools import TOOLS
from app.finance_agent.utils import load_chat_model

# Define the function that calls the model


async def call_model(
    state: State, config: RunnableConfig
) -> dict[str, list[AIMessage]]:
    """
    Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.

    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    if configuration.use_tools:
        model = load_chat_model(configuration.model).bind_tools(TOOLS)
    else:
        model = load_chat_model(configuration.model)

    if configuration.system_prompt == "None":
        prompt = ChatPromptTemplate.from_messages([("placeholder", "{messages}")])
        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
            },
            config,
        )
    else:
        # Create a prompt template. Customize this to change the agent's behavior.
        prompt = ChatPromptTemplate.from_messages(
            [("system", configuration.system_prompt), ("placeholder", "{messages}")]
        )

        # Prepare the input for the model, including the current system time
        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
                "system_time": datetime.now(tz=UTC).isoformat(),
            },
            config,
        )

    # Get the model's response
    response = cast("AIMessage", await model.ainvoke(message_value, config))

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph

workflow = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
workflow.add_node(call_model)
workflow.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
workflow.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """
    Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").

    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
workflow.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
workflow.add_edge("tools", "call_model")

# Compile the workflow into an executable graph
# You can customize this by adding interrupt points for state updates
graph = workflow.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "Finance Agent"  # This customizes the name in LangSmith
