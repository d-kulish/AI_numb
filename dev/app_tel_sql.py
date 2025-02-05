# Loading libraries
import warnings
from dotenv import load_dotenv

load_dotenv()

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
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

from telegram.ext import Application, CommandHandler, MessageHandler, filters
import html

# Connects
openai_token = os.getenv("OPENAI_API_KEY")
mongodb_client = pymongo.MongoClient(os.getenv("MONGODB_URI"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))


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



tools = [
    count_unique_conversations,
    
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


async def handle_message(update, context):
    question = update.message.text
    question = question.encode("utf-8", errors="ignore").decode("utf-8")

    state = {"messages": [HumanMessage(content=question)]}
    result_state = graph.invoke(state, config=config)
    ai_message = result_state["messages"][-1]
    ai_message_content = ai_message.content.encode("utf-8", errors="ignore").decode("utf-8")

    # Store in MongoDB
    coll_chat_history.insert_one({
        "thread_id": str(thread_id),
        "conversation": {
            "question": question,
            "answer": ai_message_content,
            "timestamp": uuid.uuid4().time,
            "chat_id": update.effective_chat.id,
            "user_id": update.effective_user.id
        },
        "timestamp": uuid.uuid4().time,
    })

    # Send response back to Telegram
    await update.message.reply_text(html.escape(ai_message_content), parse_mode='HTML')

async def start_command(update, context):
    await update.message.reply_text('Hello! I am your Numberz AI assistant. How can I help you today?')

def main():
    # Get the token from environment variables
    token = os.getenv("TELEGRAM_CHAT_BOT_TOKEN")
    
    # Create application
    application = Application.builder().token(token).build()
    
    # Add handlers
    application.add_handler(CommandHandler('start', start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    print("Starting bot...")
    application.run_polling()  # Removed the allowed_updates parameter as it's not necessary

if __name__ == "__main__":
    main()
