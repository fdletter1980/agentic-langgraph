#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install -U langgraph langchain_openai langchain_community langchain-google-vertexai langchain-google-genai chromadb langgraph pypdf langchain_google_community unstructured[pdf] streamlit sentence-transformers


# In[ ]:


#!sudo apt update


# In[ ]:


#!sudo apt install python3-pip


# In[ ]:


#!pip3 install --upgrade pip setuptools wheel


# In[ ]:


#Libraries to be installed in addition on GCP
#!sudo apt-get update
#!apt-get install poppler-utils
#! apt install tesseract-ocr
#! apt install libtesseract-dev


# In[ ]:


#get_ipython().system('python --version')


# In[ ]:


# Install python 3.11
#!sudo apt-get update -y
#!sudo apt-get install python3.11


# In[ ]:


# Change default python3 to 3.11
#!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Confirm version
#!python3 --version

# #Import Libraries

# In[ ]:


import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
import sqlite3
import streamlit as st
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.document_loaders import DirectoryLoader
from langchain_google_community import GCSDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentType, Tool, initialize_agent, AgentExecutor
from typing import Union, Literal, Sequence, TypedDict, Annotated, List, Dict, Optional,Tuple
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import TypedDict
import functools
import operator
from langgraph.graph import END, StateGraph, START, MessagesState


# # Load Non-Structured Data

# In[ ]:


import os
os.environ["GOOGLE_API_KEY"] ='YOUR GOOGLE API KEY'


# In[ ]:
#code to provision integration with our LLM Ops platform Arize
#import arize_otel

# Import open-telemetry dependencies
#from arize_otel import register_otel, Endpoints

# Setup OTEL via our convenience function
#register_otel(
#    endpoints = Endpoints.ARIZE,
#    space_id = "", # in app space settings page
#    api_key = "", # in app space settings page
#    model_id = "churn_project", # name this to whatever you would like
#    #log_to_console = True
#)
# Import the automatic instrumentor from OpenInference
#from openinference.instrumentation.langchain import LangChainInstrumentor

# Finish automatic instrumentation
#LangChainInstrumentor().instrument()

# In[ ]:

#from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader("./downloads/", glob="**/*.pdf")
documents = loader.load()
#print(documents[0])

# In[ ]:


#def download_blob(bucket_name, source_blob_name, destination_file_name):
#    """Downloads a blob from the bucket."""

#    storage_client = storage.Client()
#    bucket = storage_client.bucket(bucket_name)
#    blob = bucket.blob(source_blob_name)

#    blob.download_to_filename(destination_file_name)

#    print(
#        f"Blob {source_blob_name} downloaded to {destination_file_name}."
 #   )

# Example usage
#bucket_name = "kw-data-science-scratch"
#source_blob_name = "fred/KB/coaching_cap.pdf"
#destination_file_name = "./coaching_cap.pdf"


#download_blob(bucket_name, source_blob_name, destination_file_name)

#documents = destination_file_name

# In[ ]:


def split_docs(documents,chunk_size=100000,chunk_overlap=200):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  #text_splitter = SemanticChunker(VertexAIEmbeddings(), breakpoint_threshold_type="percentile")
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

# In[ ]:

import logging

logging.basicConfig(filename='my_app.log', level=logging.DEBUG)
logging.debug('This message should go to the log file')

# In[ ]:


#from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

persist_directory = "chroma_db"
#embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

#from langchain_huggingface import HuggingFaceEmbeddings
#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)

retriever = vectordb.as_retriever()


# In[ ]:

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# # Load Structured Data

# In[ ]:


client = bigquery.Client()


# In[ ]:


QUERY1 = (
'''
SELECT kw_uid as kw_uid, 
heatseeker,
CASE 
  when heatseeker > 0.1374 THEN 1
  else 0
END AS FLAGGED_CHURN_REALTORS,
cap as CAP_SEGMENT,
units_last_12 as SOLD_UNITS
FROM `data-lake-prod-pres-2c917b19.people.agent_retention__internal_agent_targets_tbl` LIMIT 1000 
'''
)

@st.cache_data
def fetch_churn_data():
    data = client.query_and_wait(QUERY1).to_dataframe()
    return data

df_c = client.query_and_wait(QUERY1).to_dataframe()


# In[ ]:


df_c[df_c.duplicated(['kw_uid'], keep=False)]


# In[ ]:


df_c = df_c.drop_duplicates(subset='kw_uid', keep="first")


# In[ ]:


df_c['kw_uid'] = pd.to_numeric(df_c['kw_uid'], errors='coerce')

# In[ ]:


QUERY2 = (
'SELECT distinct kw_uid, first_name, last_name, luxury, commercial, land, military ,maps_client, maps_mastery_client,  FROM `data-lake-prod-pres-2c917b19.people.ams__person_details_tbl` where active = 1 and country_id = 235 and person_type = 1'
)

@st.cache_data
def fetch_agent_ams_data():
    data = client.query_and_wait(QUERY2).to_dataframe()
    return data

df_a = client.query_and_wait(QUERY2).to_dataframe()


# In[ ]:


QUERY3 = (
'''
SELECT distinct kw_uid,
max(case when source = "Designs" THEN count end) as Designs,
max(case when source = "SmartPlans" THEN count end) as SmartPlans,
max(case when source = "Tasks" THEN count end) as Tasks,
max(case when source = "Interactions" THEN count end) as Interactions,
max(case when source = "Referrals" THEN count end) as Referrals,
max(case when source = "Campaigns" THEN count end) as Campaigns,
max(case when source = "Opportunities" THEN count end) as Opportunities,
max(case when source = "Contacts" THEN count end) as Contacts
FROM `data-lake-prod-pres-2c917b19.bi_rd.command_activity_fixed`
WHERE type != 'deleted' and date >= '2024-01-01'
group by kw_uid
;
'''
)

@st.cache_data
def fetch_command_data():
    data = client.query_and_wait(QUERY3).to_dataframe()
    return data

df_u = client.query_and_wait(QUERY3).to_dataframe()


# In[ ]:


# Replace null values with 0
df_u.fillna(0, inplace=True)


# In[ ]:


df_u[df_u.duplicated(['kw_uid'], keep=False)]


# In[ ]:


df_u['kw_uid'] = pd.to_numeric(df_u['kw_uid'], errors='coerce')


# In[ ]:


QUERY4 = (
'''
WITH contacts_data_set AS (
  SELECT c.*
  FROM (SELECT cs.*, owner.id as owner_id, author.created_by as created_by
            , CASE WHEN p.type = 'CLAIMED' THEN p.granted_to END as claimed_by
            , p.granted_to
      FROM `data-lake-prod-pres-2c917b19.contacts.contacts_snapshot` cs LEFT JOIN UNNEST(permissions) p) c
  WHERE c.archived_date IS NULL AND c.deleted_at IS NULL -- active (not deleted or archived) contacts only
)

, primary_assignee AS (
  SELECT c._id
        , CASE WHEN owner.type = 'AGENT' THEN owner.id
              WHEN claimed_by IS NOT NULL THEN claimed_by
              WHEN in_lead_route = true THEN NULL
              WHEN granted_count = 1 THEN granted_to
              ELSE created_by END as kw_uid -- primary assignee
  FROM contacts_data_set c
        JOIN (SELECT _id, COALESCE(COUNT(DISTINCT granted_to),0) as granted_count FROM contacts_data_set GROUP BY 1) a ON c._id = a._id
)

, contacts_per_kwuid AS (
  SELECT kw_uid, COUNT(DISTINCT _id) as active_contact_count
  FROM primary_assignee
  GROUP BY ALL
)

SELECT *
FROM contacts_per_kwuid
'''
)

@st.cache_data
def fetch_contacts_data():
    data = client.query_and_wait(QUERY4).to_dataframe()
    return data

df_co = client.query_and_wait(QUERY4).to_dataframe()


# In[ ]:


df_co['kw_uid'] = pd.to_numeric(df_co['kw_uid'], errors='coerce')


# In[ ]:


merged_df = df_c.merge(df_a, on='kw_uid').merge(df_u, on='kw_uid').merge(df_co, on='kw_uid')


# In[ ]:


merged_df.fillna(0, inplace=True)


# # Inject DataFrame into DB

# In[ ]:


connection = sqlite3.connect("realtors.db")
connection.execute("DROP TABLE realtors")


# In[ ]:


merged_df_str = merged_df.applymap(str)


# In[ ]:


merged_df_str.to_sql(name="realtors", con=connection)


# In[ ]:


db = SQLDatabase.from_uri("sqlite:///realtors.db")


# In[ ]:


#llm = ChatVertexAI(model="gemini-1.5-flash-002")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0.2)


# In[ ]:


toolkit = SQLDatabaseToolkit(db=db, llm=llm)


# # Set-up SQL Agent

# In[ ]:


MSSQL_AGENT_SUFFIX_WITH_MEMORY= """While generating SQL for the above query, pay attention to the below:



- General Instructions:

  - Do not use any LIMIT statements in SQL.
  - Round answers to two decimal places.
  - Avoid complicated SQL queries such as those involving division within a query.
  - Perform operations step by step.
  - Pay attention to all conditions mentioned in the query. Do not infer conditions.
  - For questions on share or market share, use column="Amount" unless stated otherwise explicitly.
  - YTD or ytd = Year to Date
   - Don't assume year as current year unless indicated so. Take data for all years unless its indicated to use a specific year"""


# In[ ]:


sql_agent_executor = create_react_agent(llm,tools=toolkit.get_tools(),state_modifier=MSSQL_AGENT_SUFFIX_WITH_MEMORY)


# In[ ]:


#Test a qeury with the SQL agent
query="How many realtors are present in the database and what is the average number of contacts?"

events = sql_agent_executor.stream(
     {"messages": [HumanMessage(content=query)]},
     stream_mode="values",
)

for event in events:
   event["messages"][-1].pretty_print()
   message_content= event["messages"][-1].content
   if "Answer:" in message_content:
      final_answer=message_content.split("Answer:",1)[1].strip()


# # Set-up RAG Agent

# In[ ]:


from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

Use only the provided context information to form your response. If an answer can not be found within the provided context information respond with 'The answer could not be found in the provided context.".

"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)


# In[ ]:


from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser

from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


# In[ ]:


#rag_chain.invoke({"question" : "What are some of the most popular coaching programs in the Full Cap status?"})


# In[ ]:


from typing import Annotated, List, Tuple, Union
from langchain_core.tools import tool

@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
    ):
  """Use Retrieval Augmented Generation to retrieve information."""
  return rag_chain.invoke(query)


# In[ ]:


tools = [retrieve_information]


# # Utilities

# In[ ]:


members = ["PDF_analyst", "Sql_agent"]
options = ["FINISH"] + members

# The agent state is the input to each node in the graph
class AgentState(MessagesState):
    # The 'next' field indicates where to route to next
    next: str


def make_supervisor_node(llm: llm, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(BaseModel):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[(*options,)]

    def supervisor_node(state: MessagesState) -> MessagesState:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        next_ = response.next
        if next_ == "FINISH":
            next_ = END

        return {"next": next_}

    return supervisor_node


# In[ ]:

## Initialize the Research Agents

rag_agent = create_react_agent(llm, tools=tools, state_modifier="You should provide RAG search.")
#rag_node = functools.partial(agent_node, agent=rag_agent, name="PDF_analyst")

def rag_node(state: AgentState) -> AgentState:
    result = rag_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="PDF_Analyst")
        ]
    }


sql_agent = create_react_agent(llm, tools=toolkit.get_tools(), state_modifier=MSSQL_AGENT_SUFFIX_WITH_MEMORY)
#sql_node = functools.partial(agent_node, agent=sql_agent, name="Sql_agent")
def sql_node(state: AgentState) -> AgentState:
    result = sql_agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="Sql_agent")]
    }


research_supervisor_node = make_supervisor_node(llm, ["PDF_Analyst", "Sql_agent"])

## Create the Research Workflow

# In[ ]:


research_builder = StateGraph(MessagesState)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("PDF_Analyst", rag_node)
research_builder.add_node("Sql_agent", sql_node)

# Define the control flow
research_builder.add_edge(START, "supervisor")
# We want our workers to ALWAYS "report back" to the supervisor when done
research_builder.add_edge("Sql_agent", "supervisor")
research_builder.add_edge("PDF_Analyst", "supervisor")
# Add the edges where routing applies
research_builder.add_conditional_edges("supervisor", lambda state: state["next"])

research_graph = research_builder.compile()


# In[ ]:


#from IPython.display import Image, display

#display(Image(research_graph.get_graph().draw_mermaid_png()))


# # Research Results

# In[ ]:


#query = "What are the most popular full cap coaching programs?"

#input_data = {"query": query}

#for s in research_graph.stream(
#    {
#        "messages": [
#            HumanMessage(content=input_data['query'])
#        ]
#    }
#):
#    if "__end__" not in s:
#        print(s)
#        print("----")


# In[ ]:


#query = "How many realtors are present in the database and what is the average count contacts?"

#input_data = {"query": query}

#for s in research_graph.stream(
#    {
#        "messages": [
#            HumanMessage(content=input_data['query'])
#        ]
#    }
#):
#    if "__end__" not in s:
#        print(s)
#        print("----")


# # Initialize the Writing Agents

# In[ ]:


def call_model(state: MessagesState):
    # add any logic to customize model system message etc here
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder_writer = StateGraph(MessagesState)
builder_writer.add_node(call_model)
builder_writer.add_edge(START, "call_model")

writer_graph = builder_writer.compile()


# In[ ]:


#from IPython.display import Image, display
#from langchain_core.runnables.graph import MermaidDrawMethod

#display(
#    Image(
#        writer_graph.get_graph().draw_mermaid_png(
#            draw_method=MermaidDrawMethod.API,
#        )
#    )
#)


# In[ ]:


#query = "Describe what a realtor does?"

#input_data = {"query": query}

#for s in writer_graph.stream(
#    {
#        "messages": [
#            HumanMessage(content=input_data['query'])
#        ]
#    }
#):
#    if "__end__" not in s:
#        print(s)
#        print("----")

# # Add Layers

# In[ ]:


prompt_message_super="""You are a supervisor tasked with managing a conversation between the following workers: {{members_super}}.
Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. If the conversation is over, respond with 'FINISH'.

Create a nicely formatted example email for the first realtor that is identified as going to churn based upon the
        first kw_uid from the list. Make sure to adhere to the following:
        1. Address the realtor with the content of the "First Name" field.
        2. Indicate in which CAP segment the realtor is currently residing in.
        3. Recommend a couple of coaching programs that realtors are taking in the next CAP segment up.
          Suggest that if they are looking for something else suggest contacting their MAPS coach.
        4. If the number of active_contact_count from the realtor is less than 201 please call this out in that this has been proven out to be correlated with higher production.
        5. Evalaute if the realtor has any usage with SmartPlans and if 0 suggest the use of SmartPlans as a tool to engage with their contacts.
        6. Make sure it is a positive and constructive message.
        7. Sign off with The Keller Williams Team

 """


# In[ ]:

teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])


# In[ ]:


def call_research_team(state: AgentState) -> AgentState:
    response = research_graph.invoke({"messages": state["messages"][-1]})
    return {
        "messages": [
            HumanMessage(content=response["messages"][-1].content, name="research_team")
        ]
    }


def call_paper_writing_team(state: AgentState) -> AgentState:
    response = writer_graph.invoke({"messages": state["messages"][-1]})
    return {
        "messages": [
            HumanMessage(content=response["messages"][-1].content, name="writing_team")
        ]
    }



# In[ ]:


# Define the graph.
super_builder = StateGraph(AgentState)
super_builder.add_node("supervisor_super", teams_supervisor_node)
super_builder.add_node("research_team", call_research_team)
super_builder.add_node("writing_team", call_paper_writing_team)


# In[ ]:


# Define the control flow
super_builder.add_edge(START, "supervisor_super")
# We want our teams to ALWAYS "report back" to the top-level supervisor when done
super_builder.add_edge("research_team", "supervisor_super")
super_builder.add_edge("writing_team", "supervisor_super")
# Add the edges where routing applies
super_builder.add_conditional_edges("supervisor_super", lambda state: state["next"])
super_graph = super_builder.compile()


# In[ ]:


from IPython.display import Image, display

display(Image(super_graph.get_graph().draw_mermaid_png()))


# In[ ]:


#for s in super_graph.stream(
#    {
#        "messages": [
#            ("user", "select first realtor and write email.")
#        ],
#    },
#    {"recursion_limit": 150},
#):
#    print(s)
#    print("---")

# In[ ]:

# Function to invoke the compiled graph externally
def invoke_our_graph(st_messages, callables):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return super_graph.invoke({"messages": st_messages}, config={"callbacks": callables, "recursion_limit": 50})

