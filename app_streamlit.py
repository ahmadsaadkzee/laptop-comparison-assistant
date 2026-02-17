import streamlit as st
import rag
import os

# Page configuration
st.set_page_config(
    page_title="Laptop Comparison RAG",
    page_icon="💻",
    layout="wide"
)

# Title and description
st.title("💻 Laptop Specification & Comparison Assistant")
st.markdown("""
Ask questions about laptops or compare two models. 
The system uses local documents and falls back to web search for missing information.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message and message["source"]:
            with st.expander("Source Information"):
                st.write(message["source"])

# React to user input
if prompt := st.chat_input("Ex: Compare Dell XPS 13 vs Dell Latitude 7400"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Pass history excluding the current prompt to avoid duplication
                history_to_pass = st.session_state.messages[:-1]
                print(f"STREAMLIT DEBUG: Passing history of length {len(history_to_pass)}")
                response, source = rag.answer_question(prompt, history=history_to_pass)
                st.markdown(response)
                if source:
                    with st.expander("Source Information"):
                         st.write(source)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source": source
                })
            except Exception as e:
                import traceback
                st.error(f"An error occurred: {e}")
                st.code(traceback.format_exc())
