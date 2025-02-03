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

# Load environment variables
load_dotenv()

openai_token = os.getenv('OPENAI_API_KEY')
mongodb_uri = os.getenv('MONGODB_URI')

mongodb_client = pymongo.MongoClient(
    mongodb_uri
)

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
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    CHAT_ID = os.getenv('CHAT_ID') 

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
def general_questions(question: str) -> str:
    """
    This funciton answers 3 questions about 'Consulting for Retail' (C4R) company:
    - Company history, specialization and mission;
    - Management - team, positions, roles and linkedin;
    - Offices - addresses, phones, e-mails.

    Args:
        inquery - string, this is text with question about the company;

    Returns:
        str - function answers the question based on selected description.
    """
    query_embedding = create_embeddings(question)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": 1,
            }
        },
        {"$project": {"_id": 0, "chunk": 1}},
    ]

    results = coll_corp_gen.aggregate(pipeline)
    result = next(results, None)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assitant  Consulting for Ukraine (C4R, in short) company.",
        },
        {"role": "user", "content": f"Can I ask you a question - {question}"},
        {
            "role": "system",
            "content": f"""Answer within the following guidance: 
                1. Find the answer to the question ({question})in information provided in tiple backticks '''{result['chunk']}''' - try not change answers you receive from tools; 
                2. Be polite and professional; 
                3. Present only finished sentences; 
                4. If asked about contacts of Managers direct to their linkedin accounts, if links are known; 
                """,
        },
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=300, temperature=0.5
    )

    return response.choices[0].message.content.strip()

@tool
def internal_rules_general(inquery: str) -> str:
    """
    This funciton answers quesitons about internal policies and rules, some of them:
        1. Planning and procedure for granting vacations in the company
        2. Project management rules for the implementation department
        3. List of company rules
        4. Interaction rules within the implementation department
        5. Instructions for reward payments
        6. Communication rules
        7. Invoicing in ERM (IRCG/C4R) NEW
        8. Interaction rules during remote work
        9. Business trips
        10. RULES FOR USING CORPORATE PRESENTATION C4R and IRCG etc.

    Args:
        inquery - string, this is text with question about the company;

    Returns:
        str - function answers the question based on selected description.
    """
    query_embedding = create_embeddings(inquery)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "general_xmls_vector_search",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": 1,
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "link": 1,
                "doc_name": 1,
            }
        },
    ]

    results = coll_internal_rules_general.aggregate(pipeline)
    result = next(results, None)

    if result:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assitant who answers GENERAL questions about corporate policies",
            },
            {"role": "user", "content": f"can I ask you a question - {inquery}"},
            {
                "role": "system",
                "content": f"""Answer within the following guidance: 
                    1. Find the answer to the question ({inquery}) in information provided in tiple backticks '''{result['text']}'''; 
                    2. Be polite and professional; 
                    3. Present only finished sentences; 
                    4. In the end, suggest reading source link in the tiple backticks '''{result['link']}'''; 
                    5. If needed, you can mention the document name in the tiple backticks '''{result['doc_name']}''';
                    """,
            },
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=1000, temperature=0.5
        )

    return response.choices[0].message.content.strip()

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

@tool 
def erm_cases(inquery: str) -> str:
    """
    Description: This function is based on Customer Support ERM cases and how those cases were resolved. 
        It looks at how clients' problems were described, how problems were analysed and how specialits resolved escalated problems.  

    Args:
        inquery - string, this is text with question about the company;

    Returns:
        str - function answers the question based on selected description.
    """
    query_embedding = create_embeddings(inquery) 
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    
    documents = list(coll_erm_cases.find(
        {"embedding": {"$exists": True}},
        {"_id": 1, "embedding": 1}
    ))
    
    embeddings = np.array([doc["embedding"] for doc in documents])
    ids = [doc["_id"] for doc in documents]

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    most_similar_idx = np.argmax(similarities)
    best_match_id = ids[most_similar_idx]
    
    result_doc = coll_erm_cases.find_one(
        {"_id": best_match_id},
        {"text": 1}
    )

    if result_doc:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assitant who consults on similar ERM cases and how those cases were resolved",
            },
            {"role": "user", "content": f"can I ask you a question - {inquery}"},
            {
                "role": "system",
                "content": f"""Answer within the following guidance: 
                    1. Refer to the a similar problem and give a short description, it could be found '''{result_doc['text']}'''; 
                    2. Explain how similar problem was solved, the solution you can find here - '''{result_doc['text']}''';
                    3. Suggests steps to fix the problem; 
                    4. Answer in a structured way and in the same language the question is formulated;  
                    5. If recommended steps do not solve the problem, suggest to contact Customer Support and create a ticket; 
                    """,
            },
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=1000, temperature=1
        )

    return(response.choices[0].message.content.strip())

@tool
def manuals_pdf(inquery: str) -> str:
    """
    Description: This function provides detailed answers on User manuals for GOLD Stock and GOLD Central (retail ERP systems). It covers the following manuals - Administration, 
    Allotment, Basic Data - Article Third Party, Warehouse Map, Charts, Basic Data Articles, Basic Data Third Party, Packing Management, Parameter Management, 
    Stock Management, DO Management, Route Management, Histories, Inventories, Scheduling, Cross-Docking, Alcohol Parameterisation, Report Parameterisation, 
    Distribution Parameterisation, Inventory Parameterisation, Preparation Parameterisation, Stock Status Parameterisation, Basic Data - Picking, Standard Preparation, 
    Immidiate Release Preparation, Dock Occupation, Radio, Receptions, Replenishment, Distribution, Preparation Supervision, Post-preparation Processing, Traceability, 
    Vocal, Preparation Areas and Path. 

    Args:
        inquery - string, this is text with question about the company;

    Returns:
        str - function answers the question based on selected description.
    """
    query_embedding = create_embeddings(inquery) 

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": 1,
            }
        },
        {
            "$project": {
                "_id": 0,
                "document": 1,
                "path": 1,
            }
        },
    ]

    results = coll_manuals_pdf.aggregate(pipeline)
    result = next(results, None)

    if result:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assitant who consults on similar ERM cases and how those cases were resolved",
            },
            {"role": "user", "content": f"can I ask you a question - {inquery}"},
            {
                "role": "system",
                "content": f"""Answer within the following guidance: 
                    1. Answer the {inquery} in information provided in tiple backticks '''{result['document']}''';
                    4. Answer in a structured way and in the same language the question is formulated;  
                    5. In the end of the your answer recommend to read the full document in the tiple backticks '''{result['path']}''' and provide 'section_title' from {result};
                    """,
            },
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=1000, temperature=1
        )

    return(response.choices[0].message.content.strip())


tools = [
    general_questions,
    internal_rules_general,
    count_unique_conversations,
    approval_money_transfer, 
    erm_cases, 
    manuals_pdf
]
tool_node = ToolNode(tools)

# App
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
         You are an AI assistant for Consulting for Retail (C4R, in short) company. You follow these instructions: 
            1. answer questions in the same language the clients asks them; 
            2. be kind, professional and polite; 
            3. format your answers in a structured way; 
            4. answer questions only related to C4R; 
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
