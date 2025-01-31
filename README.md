# agentic-langgraph

Repo contains both a streamlit app version for conversational use (streamlit run __main__.py) and a python notebook version without the app (AgenticAI_LangGraph_Hier.ipynb).

![image](https://github.com/user-attachments/assets/8f8c1f43-25de-417d-8fc8-bb031ff76e47)

Solution design includes the following components: 
  1. Hierarchical LangGraph design pattern with multiple supervisors. See online docs: https://langchain-ai.github.io/langgraph/tutorials/introduction/
  2. Agents are using ReAct (Reasoning and Acting) framework.
  3. Structured data sets are using a schema tool for the AI agents to retrieve the data.
  4. Unstructured data retrieves documents from a bucket and uses RAG architecture.
  5. Vector DB = ChromaDB
  6. Has a Streamlit app UI on top of it.

Prerequisites:
  1. Create a virtual environment and install requirements.txt (or use notebook IDE)
  2. Get Google API key covering GenerativeAI services


