import streamlit as st
import os
from langchain.utilities import SQLDatabase
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.memory import ConversationBufferMemory

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

# Database connection using environment variable
db = SQLDatabase.from_uri(os.getenv("DATABASE_URI"))

# Initialize LLM with environment variables
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model_name="gpt-4",
    temperature=0,
    openai_api_version="2024-03-01-preview",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE")
)

# Initialize SQL agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    memory=st.session_state.memory,
    agent_type="openai-tools"
)

# Streamlit UI
st.title("Health Hackathon Chatbot")
st.write("Ask about claims in the HealthHackathan database.")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user query
if prompt := st.chat_input("Ask a question about claims"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                response = agent_executor.run(prompt)
                st.markdown(response)
                # Add assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")