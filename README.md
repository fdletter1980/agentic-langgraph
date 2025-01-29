# agentic-langgraph

Repo contains both a streamlit app version for conversational use (streamlit run __main__.py) and a python notebook version without the app (AgenticAI_LangGraph_Hier.ipynb).

![Hierarchical](https://github.com/user-attachments/assets/487b5af0-89a2-4317-92fe-a52cef5ab140)

Solution design includes the following components: 
  1. Hierarchical LangGraph design pattern with multiple supervisors. See online docs: https://langchain-ai.github.io/langgraph/tutorials/introduction/
  2. Agents are using ReAct (Reasoning and Acting) framework.
  3. Structured data sets are residing in our GCP BQ environment and are using a schema tool for the AI agents to retrieve the data.
  4. Unstructured data retrieves documents from a GCP bucket and uses RAG architecture.
  5. Vector DB = ChromaDB
  6. Has a Streamlit app UI on top of it.

Prerequisites:
  1. Create a virtual environment and install requirements.txt (or use notebook IDE)
  2. Get Google API key covering GenerativeAI services
  3. The app version has integration code commented out re our LLM Ops platform (Arize)

Example using the super prompt embedded in the code triggering the graph:

![image](https://github.com/user-attachments/assets/59da1883-4c7d-4c52-acae-514ce34bd81f)
