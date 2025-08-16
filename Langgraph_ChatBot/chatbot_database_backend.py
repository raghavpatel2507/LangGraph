from langgraph.graph import StateGraph,START,END,add_messages
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
import os
import sqlite3


load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="llama3-70b-8192",api_key=GROQ_API_KEY)

#state
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    #take user query from state
    messages=state['messages']
    #send to llm
    response=llm.invoke(messages)
    #response store llm
    return {'messages':[response]}

#edage and nodes
conn=sqlite3.connect(database="chatbot.db",check_same_thread=False)
checkpointer=SqliteSaver(conn)
graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')

graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)

CONFIG = {'configurable': {'thread_id': "thread-1"}}

response=chatbot.invoke(
                {'messages': [HumanMessage(content="what is my name")]},
                config= CONFIG,
                stream_mode= 'messages'
            )
print(response)