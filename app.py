# streamlit_app.py
import streamlit as st
from rag_pipeline import load_llm, augmented_prompt, build_vectorstore
from langchain.schema import SystemMessage, AIMessage, HumanMessage

# Set page configuration
st.set_page_config(page_title="RAG Chat with Llama 2", page_icon="🤖", layout="wide")

# Custom CSS styling for a modern look
custom_css = """
<style>
body {
    background: linear-gradient(135deg, #ece9e6, #ffffff);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.chat-bubble {
    padding: 10px 15px;
    border-radius: 20px;
    margin: 5px 0;
    max-width: 80%;
}
.user-bubble {
    background-color: #DCF8C6;
    align-self: flex-end;
}
.ai-bubble {
    background-color: #FFF;
    border: 1px solid #ddd;
    align-self: flex-start;
}
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px;
    height: 400px;
    overflow-y: auto;
    background: #f7f7f7;
    border-radius: 10px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# App title and description
st.title("🤖 Retrieval-Augmented Llama 2 Chatbot")
st.markdown("An interactive chatbot using Llama 2 via the Hugging Face Inference API with Pinecone-backed context retrieval.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Build/retrieve the vectorstore if not already loaded in session state
if "vectorstore" not in st.session_state:
    with st.spinner("Building vectorstore... (this may take a while)"):
        vectorstore, pc, index_name = build_vectorstore()
        st.session_state.vectorstore = vectorstore

# Load the LLM (assumes API tokens are already set in environment)
llm = load_llm()

# Chat interface
st.subheader("Chat with the Bot")
query_input = st.text_input("Enter your query:")
if st.button("Send Query"):
    if query_input.strip() == "":
        st.warning("Please enter a valid query!")
    else:
        vectorstore = st.session_state.vectorstore
        prompt_text = augmented_prompt(query_input, vectorstore)
        # Optionally show the augmented prompt for debugging/inspection
        with st.expander("Show Augmented Prompt"):
            st.code(prompt_text, language="text")
        with st.spinner("Generating response..."):
            response = llm(prompt_text)
        st.session_state.chat_history.append(("User", query_input))
        st.session_state.chat_history.append(("AI", response))

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Display chat history in a custom chat bubble style
st.subheader("Conversation")
for role, message in st.session_state.chat_history:
    bubble_class = "user-bubble" if role == "User" else "ai-bubble"
    st.markdown(
        f'<div class="chat-bubble {bubble_class}"><strong>{role}:</strong> {message}</div>',
        unsafe_allow_html=True
    )
