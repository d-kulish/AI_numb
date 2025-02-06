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

from sqlalchemy import create_engine, text
from pathlib import Path
from sqlalchemy.orm import sessionmaker


# Connects
openai_token = os.getenv("OPENAI_API_KEY")
mongodb_client = pymongo.MongoClient(os.getenv("MONGODB_URI"))

# Update the token variable name to match .env file
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_CHAT_BOT_TOKEN")  # Changed from TELEGRAM_BOT_TOKEN
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_CHAT_BOT_TOKEN not found in environment variables")

print("Bot token validation:", "✓" if TELEGRAM_BOT_TOKEN else "✗")
print(f"Token prefix: {TELEGRAM_BOT_TOKEN[:8] if TELEGRAM_BOT_TOKEN else 'None'}")

coll_chat_history = mongodb_client["AI_numb"]["chat_history"]

model = ChatOpenAI(model="gpt-4o", api_key=openai_token, temperature=0.5)
openai_client = OpenAI(api_key=openai_token)

# load_dotenv(Path.cwd().parent / ".env")

# Get all connection parameters from environment variables
db_ip = os.getenv("DB_IP")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
db_port = os.getenv("DB_PORT")

# Create the connection string
connection_string = f"postgresql://{db_user}:{db_password}@{db_ip}:{db_port}/{db_name}"

# Create the SQLAlchemy engine
engine = create_engine(connection_string)
Session = sessionmaker(bind=engine)


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

@tool 
def table_structure(table_name: str):
    """
    This funciton returns first 5 rows of any table from datatabase 'am_db'.

    Args:
        table_name (str): The name of the table to query. 

    Returns:
        python dictionary with 5 rows from the table.
    """
    
    session = Session()

    try:
        sql = text(f"SELECT * FROM {table_name} LIMIT 5")
        result = session.execute(sql)
        columns = result.keys()
        records = [dict(zip(columns, row)) for row in result]
    finally:
        session.close()
    
    return records


tools = [
    count_unique_conversations,
    table_structure, 
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
    try:
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        question = update.message.text
        question = question.encode("utf-8", errors="ignore").decode("utf-8")

        # Send typing action to show the bot is processing
        await update.message.chat.send_action(action="typing")

        print(f"Processing message: '{question}' from User: {username} (ID: {user_id})")

        try:
            state = {"messages": [HumanMessage(content=question)]}
            result_state = graph.invoke(state, config=config)
            ai_message = result_state["messages"][-1]
            ai_message_content = ai_message.content.encode("utf-8", errors="ignore").decode("utf-8")
        except Exception as api_error:
            print(f"OpenAI API Error: {str(api_error)}")
            raise

        # Store in MongoDB
        try:
            coll_chat_history.insert_one({
                "thread_id": str(thread_id),
                "conversation": {
                    "question": question,
                    "answer": ai_message_content,
                    "timestamp": time.time(),  # Use actual timestamp instead of UUID time
                    "chat_id": update.effective_chat.id,
                    "user_id": user_id,
                    "username": username
                },
                "timestamp": time.time(),
            })
        except Exception as db_error:
            print(f"MongoDB Error: {str(db_error)}")
            # Continue even if storage fails
            
        # Send response back to Telegram
        if ai_message_content:
            await update.message.reply_text(ai_message_content)
        else:
            await update.message.reply_text("I apologize, but I couldn't generate a response. Please try again.")

    except Exception as e:
        error_message = f"Error processing message: {str(e)}"
        print(error_message)
        await update.message.reply_text("I encountered an error processing your message. Please try again.")

async def start_command(update, context):
    user = update.effective_user
    welcome_message = (
        f"Hello {user.first_name}! I am your Numberz AI assistant.\n"
        "I'm here to help you with any questions you have.\n"
        "Feel free to ask me anything!"
    )
    await update.message.reply_text(welcome_message)

def main():
    try:
        if not TELEGRAM_BOT_TOKEN:
            raise ValueError("Bot token is missing or invalid")
            
        print(f"Initializing bot with token starting with: {TELEGRAM_BOT_TOKEN[:8]}...")
        
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler('start', start_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        print("Bot is starting up...")
        print("To test the bot, send /start in Telegram")
        application.run_polling(drop_pending_updates=True)
    except Exception as e:
        print(f"Failed to start bot: {str(e)}")
        raise

if __name__ == "__main__":
    main()
