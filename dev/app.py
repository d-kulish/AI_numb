# Loading libraries
import warnings

warnings.filterwarnings(
    "ignore",
    message="python-telegram-bot is using upstream urllib3. This is allowed but not supported by python-telegram-bot maintainers",
)
import asyncio

# import getpass
import os
import pymongo
from openai import OpenAI
from langchain_openai import ChatOpenAI

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from typing import Literal
from langgraph.graph import StateGraph, MessagesState
from scipy.spatial import distance

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import json
from langchain_core.messages import ToolMessage

import uuid
import re
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage

import telegram

# from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

import os
from dotenv import load_dotenv

openai_token = os.getenv("OPENAI_API_KEY")
mongodb_uri = os.getenv("MONGODB_URI")

mongodb_client = pymongo.MongoClient(mongodb_uri)
coll_chat_history = mongodb_client["AI_numb"]["chat_history"]

model = ChatOpenAI(model="gpt-4o", api_key=openai_token, temperature=0.5)
openai_client = OpenAI(api_key=openai_token)

memory = MemorySaver()


# Classes
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


class AgentState(MessagesState):
    # The 'next' field indicates where to route to next
    next: str


# Functions
def create_embeddings(texts):
    response = openai_client.embeddings.create(
        model="text-embedding-3-large", input=texts
    )
    response_dict = response.model_dump()
    return response_dict["data"][0]["embedding"]


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def get_last_ai_message_content(messages):
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            print(message.content)
    return None


async def send_and_receive(dictionary):
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("CHAT_ID")  # Integer chat ID

    bot = None

    try:
        # Initialize the bot
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

        # Create inline keyboard
        keyboard = [
            [
                InlineKeyboardButton("Approve", callback_data="approve"),
                InlineKeyboardButton("Deny", callback_data="deny"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Send request message
        message_text = f"Please approve the following request: {dictionary}"
        await bot.send_message(
            chat_id=CHAT_ID, text=message_text, reply_markup=reply_markup
        )
        # print("Waiting for response...")

        # Get initial update_id
        updates = await bot.get_updates(timeout=1)
        update_id = updates[-1].update_id + 1 if updates else None

        while True:
            updates = await bot.get_updates(offset=update_id, timeout=5)

            for update in updates:
                update_id = update.update_id + 1

                if update.callback_query:
                    callback = update.callback_query
                    if callback.message.chat.id == CHAT_ID:
                        response = callback.data
                        # Acknowledge callback
                        await bot.answer_callback_query(callback.id)
                        # Send thank you message
                        await bot.send_message(
                            chat_id=CHAT_ID, text="Thank you for your reaction"
                        )
                        return response

            await asyncio.sleep(1)

    except Exception as e:
        # print(f"Error: {e}")
        return None
    finally:
        # Clean up only if bot was created
        if bot is not None:
            del bot


# Tools


@tool
def count_unique_conversations():
    """
    Aggregates the total number of unique conversation saved MongoDB collection for AI chat.

    Args:
        None

    Returns:
        int: The total number of unique IDs.
    """
    unique_ids = coll_chat_history.distinct("_id")
    # return f'Number of unique conversation saved is {len(unique_ids)}'
    return len(unique_ids)


@tool
def approval_money_transfer(sentence: str) -> str:
    """
    This function asks for approval of money transfer.

    Args:
        sentence - string, this is the sentence that needs approval;

    Returns:
        str - function returns the approval status.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a system that helps to get approval or denial for money transfers.",
        },
        {"role": "user", "content": f"Can I ask you a question - {sentence}"},
        {
            "role": "system",
            "content": f"""Answer within the following guidance: 
                        1. Define from {sentence} the 'recipient_name' and the 'amount_to_send'; 
                        2. Create a JSON object with the recipient and the amount like {{"Recipient": 'recipient_name', "Amount": 'amount_to_send'}}; 
                        3. Return only the final JSON object; 
                        """,
        },
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```") and content.endswith("```"):
        content_lines = content.strip().split("\n")
        if content_lines[0].startswith("```"):
            content_lines = content_lines[1:]
        if content_lines[-1].startswith("```"):
            content_lines = content_lines[:-1]
        content = "\n".join(content_lines)

    data = json.loads(content)

    result = asyncio.run(send_and_receive(data))
    if result == "approve":
        return "Your request was Approved."
    elif result == "deny":
        return "Your request was Denied."
    else:
        return "There was an error processing your request."


tools = [
    count_unique_conversations,
    approval_money_transfer,
]
tool_node = ToolNode(tools)

# App
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
         You are an AI assistant for Numberz company. You follow these instructions: 
            1. answer questions in the same language the clients asks them; 
            2. be kind, professional and polite; 
            3. format your answers in a structured way; 
            4. answer questions only related to Numberz; 
                  """,
        ),
        ("placeholder", "{messages}"),
    ]
)

chat_agent = create_react_agent(model, tools=tools, state_modifier=prompt)


def chat_node(state: AgentState) -> AgentState:
    result = chat_agent.invoke(state)
    return {"messages": result["messages"]}


graph_builder.add_node("chatbot", chat_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}


def main():
    conversation = []
    thread_id = uuid.uuid4()  # Generate a unique thread ID for the conversation

    try:
        while True:
            question = input("You: ")
            if question.lower() in ["exit", "quit"]:
                break
            question = question.encode("utf-8", errors="ignore").decode("utf-8")

            state = {"messages": [HumanMessage(content=question)]}
            result_state = graph.invoke(state, config=config)
            ai_message = result_state["messages"][-1]
            ai_message_content = ai_message.content.encode(
                "utf-8", errors="ignore"
            ).decode("utf-8")

            print(f"AI: {ai_message_content}")
            conversation.append(
                {
                    "question": question,
                    "answer": ai_message_content,
                    "timestamp": uuid.uuid4().time,
                }
            )
    finally:
        coll_chat_history.insert_one(
            {
                "thread_id": str(thread_id),
                "conversation": conversation,
                "timestamp": uuid.uuid4().time,
            }
        )


if __name__ == "__main__":
    main()
