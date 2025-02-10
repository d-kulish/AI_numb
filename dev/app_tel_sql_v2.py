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
from datetime import datetime, timedelta


# Connects
openai_token = os.getenv("OPENAI_API_KEY")
mongodb_client = pymongo.MongoClient(os.getenv("MONGODB_URI"))

# Update the token variable name to match .env file
TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_CHAT_BOT_TOKEN"
)  # Changed from TELEGRAM_BOT_TOKEN
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


@tool
def top10_product_sales_day(target_date=None):
    """
    Returns the top 10 products by sales for a specific day.

    Args:
        target_date (str or None): The target date in 'YYYY-MM-DD' format.
                                 If None, returns yesterday's data.
                                 Only accepts dates within the last 30 days.

    Returns:
        List[Dict]: A list of dictionaries with product_id, name, and sales.
        Dict: Error message if date is invalid or no data found.
    """
    try:
        today = datetime.now().date()

        # Date handling
        if target_date is None:
            target_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        elif isinstance(target_date, str):
            try:
                parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
                # Validate date range
                if parsed_date > today:
                    return {"error": "Cannot query future dates"}
                if parsed_date < (today - timedelta(days=30)):
                    return {
                        "error": f"Can only query dates between {(today - timedelta(days=30)).strftime('%Y-%m-%d')} and {today.strftime('%Y-%m-%d')}"
                    }
            except ValueError:
                return {"error": "Invalid date format. Please use YYYY-MM-DD"}

        session = Session()

        query = """
        SELECT 
            aaa.article_id as product_id, 
            aaa2.translation as name, 
            COALESCE(SUM(aaa.price), 0) as sales
        FROM am_app_articlesales aaa
        JOIN am_app_articlestockunitdictionary aaa2 
            ON aaa.article_id = aaa2.id 
        WHERE DATE(aaa.date_of_sales) = DATE(:target_date)
        GROUP BY aaa.article_id, aaa2.translation 
        ORDER BY sales DESC
        LIMIT 10
        """

        result = session.execute(text(query), {"target_date": target_date})
        columns = result.keys()
        records = [dict(zip(columns, row)) for row in result]

        if not records:
            return {"message": f"No sales data found for date {target_date}"}

        return records

    except Exception as e:
        print(f"Error in top10_product_sales_day: {str(e)}")
        return {"error": f"Database query failed: {str(e)}"}
    finally:
        if "session" in locals():
            session.close()


tools = [
    count_unique_conversations,
    table_structure,
    top10_product_sales_day,
]

tool_node = ToolNode(tools)

# App
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant for Numberz company, operating in {current_year}. Follow these instructions:
1. Answer questions in the same language the clients asks them
2. Be kind, professional and polite
3. Format your answers in a structured way
4. Answer questions only related to Numberz
5. Date handling requirements:
   - Today's date is {current_date}
   - You can ONLY access data from the last 30 days from today
   - All dates MUST be from {current_year}
   - Never reference or use dates from 2023 or earlier years
   - If no specific date is mentioned, use yesterday ({yesterday_date})
   - Always use YYYY-MM-DD format
   - Always validate that requested dates are within the 30-day window from today
6. For sales data queries:
   - Convert any relative date references (e.g., "last Monday") to actual dates in YYYY-MM-DD format
   - Only allow dates between {thirty_days_ago} and {current_date}
   - Clearly explain date constraints when users request data outside this range
        """.format(
                current_date=datetime.now().strftime("%Y-%m-%d"),
                current_year=datetime.now().year,
                yesterday_date=(datetime.now() - timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
                thirty_days_ago=(datetime.now() - timedelta(days=30)).strftime(
                    "%Y-%m-%d"
                ),
            ),
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
        # Enhanced user identification
        user = update.effective_user
        user_id = user.id
        username = user.username or "Unknown"
        first_name = user.first_name or "Unknown"
        last_name = user.last_name or "Unknown"

        # Detailed user logging
        print(
            f"""
User Details:
- ID: {user_id}
- Username: {username}
- First Name: {first_name}
- Last Name: {last_name}
- Is Bot: {user.is_bot}
        """
        )

        question = update.message.text
        question = question.encode("utf-8", errors="ignore").decode("utf-8")

        # Send typing action to show the bot is processing
        await update.message.chat.send_action(action="typing")

        print(f"Processing message: '{question}' from User: {username} (ID: {user_id})")

        try:
            state = {"messages": [HumanMessage(content=question)]}
            result_state = graph.invoke(state, config=config)
            ai_message = result_state["messages"][-1]
            ai_message_content = ai_message.content.encode(
                "utf-8", errors="ignore"
            ).decode("utf-8")
        except Exception as api_error:
            print(f"OpenAI API Error for user {user_id}: {str(api_error)}")
            raise

        # Store in MongoDB with enhanced user info
        try:
            coll_chat_history.insert_one(
                {
                    "thread_id": str(thread_id),
                    "conversation": {
                        "question": question,
                        "answer": ai_message_content,
                        "timestamp": time.time(),  # Use actual timestamp instead of UUID time
                        "chat_id": update.effective_chat.id,
                        "user_id": user_id,
                        "username": username,
                        "user_details": {
                            "first_name": first_name,
                            "last_name": last_name,
                            "is_bot": user.is_bot,
                        },
                    },
                    "timestamp": time.time(),
                }
            )
        except Exception as db_error:
            print(f"MongoDB Error for user {user_id}: {str(db_error)}")
            # Continue even if storage fails

        # Send response back to Telegram with user-specific error handling
        if ai_message_content:
            try:
                await update.message.reply_text(ai_message_content)
                print(f"Successfully sent response to user {user_id}")
            except Exception as send_error:
                print(f"Failed to send message to user {user_id}: {str(send_error)}")
                raise
        else:
            await update.message.reply_text(
                "I apologize, but I couldn't generate a response. Please try again."
            )

    except Exception as e:
        error_message = f"Error processing message for user {user_id}: {str(e)}"
        print(error_message)
        await update.message.reply_text(
            "I encountered an error processing your message. "
            "Please try again or contact support if the issue persists."
        )


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
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )

        print("Bot is starting up...")
        print("To test the bot, send /start in Telegram")
        application.run_polling(drop_pending_updates=True)
    except Exception as e:
        print(f"Failed to start bot: {str(e)}")
        raise


if __name__ == "__main__":
    main()
