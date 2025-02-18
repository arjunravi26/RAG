import streamlit as st
from rag_pipeline import load_llm, augmented_prompt, build_vectorstore

# Set page configuration
st.set_page_config(
    page_title="RAG Chat with Llama 2",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS styling for a modern, creative look
custom_css = """
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
<style>
body {
    background: linear-gradient(135deg, #ece9e6, #ffffff);
    font-family: 'Roboto', sans-serif;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(180deg, #ff9a9e, #fad0c4);
    color: #fff;
    padding: 20px;
}
.main-header {
    text-align: center;
    padding: 10px 0;
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.chat-container {
    background: #ffffff;
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    padding: 20px;
    height: 500px;
    overflow-y: auto;
}
.chat-bubble {
    padding: 12px 18px;
    border-radius: 20px;
    margin: 8px 0;
    max-width: 75%;
    animation: fadeIn 0.5s ease-in-out;
}
.user-bubble {
    background-color: #DCF8C6;
    margin-left: auto;
    text-align: right;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.ai-bubble {
    background-color: #f1f0f0;
    margin-right: auto;
    text-align: left;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
.input-section {
    margin-top: 20px;
    background: #fff;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
footer {
    text-align: center;
    padding: 10px;
    color: #888;
    font-size: 0.9em;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar with instructions and info
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    **RAG Chat with Llama 2** is an interactive chatbot that uses retrieval-augmented generation.  
    **How it works:**  
    - Your query is augmented with context retrieved from a vector store.  
    - Llama 2 generates a response based on the augmented prompt.  
    """)
    st.markdown("### Enjoy your chat experience!")
    st.info("Note: Building the vectorstore may take a moment on first run.")

# Main header area
st.markdown('<div class="main-header"><h1>🤖 Retrieval-Augmented Llama 2 Chatbot</h1><p>An innovative chatbot using Llama 2, powered by context retrieval!</p></div>', unsafe_allow_html=True)

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

# Layout: Two columns for chat interface and history
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Send a Query")
    with st.container():
        query_input = st.text_input("Your query:", key="query")
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
                    response = llm.invoke(prompt_text)
                    print(response)
                st.session_state.chat_history.append(("User", query_input))
                st.session_state.chat_history.append(("AI", response))
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()

with col2:
    st.subheader("Conversation")
    chat_container = st.empty()
    # Render chat history inside a scrollable container
    chat_history_html = '<div class="chat-container">'
    for role, message in st.session_state.chat_history:
        bubble_class = "user-bubble" if role == "User" else "ai-bubble"
        icon = "🧑" if role == "User" else "🤖"
        chat_history_html += f'<div class="chat-bubble {bubble_class}"><strong>{icon} {role}:</strong><br>{message}</div>'
    chat_history_html += '</div>'
    chat_container.markdown(chat_history_html, unsafe_allow_html=True)

# Footer
st.markdown('<footer>Developed with ❤️ using Streamlit, Python, and cutting-edge AI.</footer>', unsafe_allow_html=True)
