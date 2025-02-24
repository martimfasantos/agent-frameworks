import streamlit as st
import importlib
from utils import get_available_agents


# --------------------------------------
#           Streamlit UI
# --------------------------------------

# Page page title
st.set_page_config(page_title="Chat Tester 🤖")
st.title("Chat Tester 🤖")

# Fix annoying UI issues
st.markdown(
    """
    <style>
    .stAppDeployButton {
        visibility: hidden;
    }
    .stSidebar {
        min-width: 200px;
        max-width: 250px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------
#     Side Bar Config
# ------------------------
# - Select the agent framework to interact with
# - Button to clear the chat history

with st.sidebar:
    # get available agents
    available_agents = get_available_agents()

    # Select agent 
    selected_agent = st.sidebar.selectbox(
        "Select Agent",
        options=list(available_agents.keys()),
        format_func=lambda x: available_agents[x],
        key="agent_selector"
    )

    # Dynamic import of selected agent
    if selected_agent not in st.session_state:
        try: 
            module = importlib.import_module(selected_agent)
            st.session_state[selected_agent] = {
                "agent": module.Agent(),
                "messages": []
            }
        except Exception as e:
            st.error(f"Error loading agent: {str(e)}")

    current_agent = st.session_state[selected_agent]["agent"]
    current_chat_history = st.session_state[selected_agent]["messages"]

    st.caption(f"\U00002139\U0000FE0F Streaming is not enabled for this {current_agent.name}")

    st.divider()

    # Clear chat history
    if st.button(f"🗑️ Clear {current_agent.name}'s Chat History"):
        current_agent.clear_chat()
        current_chat_history = []



# ------------------------
#     Chat Interface
# ------------------------
# - Display a caption with the agent name you are chatting with
# - Display previous chat history for the selected agent
# - Ask user to type write a message
# - Get the response from the agent
# - Display the agents's response

# Display title with agent name and its tools
st.caption(f"Chatting with :blue[{current_agent.name}] \
            with the following tools: \n\n{current_agent.tools_descriptions}")

# Display chat history
for msg in current_chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state[selected_agent]["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from agent
    with st.chat_message("assistant"):

        try:
            response = current_agent.chat(user_input)
        except Exception as e:
            response = f"Error: {e}"

        # Display the response
        st.markdown(response)

        # Add the message to the chat history
        st.session_state[selected_agent]["messages"].append(
            {"role": "assistant", "content": response}
        )
