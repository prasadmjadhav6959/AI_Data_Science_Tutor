import os
import streamlit as st
import traceback
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()
key = os.getenv("GEMINI_API_KEY")

if not key:
    st.error("Error: Gemini API key not found. Set GEMINI_API_KEY in a .env file.")
    st.stop()

# Configure Gemini API
try:
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()

# Initialize memory
memory = ConversationBufferMemory()

# Streamlit UI Setup
st.set_page_config(page_title="AI Data Science Tutor", page_icon=":robot_face:")
st.title("💡 AI Conversational Data Science Tutor")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Function to execute Python code safely
def execute_code(code):
    try:
        local_env = {}
        exec(code, {"np": np, "pd": pd, "plt": plt}, local_env)

        results = {}
        for var_name, value in local_env.items():
            if isinstance(value, (int, float, str, list, dict, tuple, pd.DataFrame)):
                results[var_name] = value
            elif isinstance(value, plt.Figure):
                results["plot"] = value
        return results

    except Exception as e:
        return {"error": traceback.format_exc()}

# User Input and Button
user_input = st.text_area("Ask a data science question or enter Python code:")
if st.button("Ask"):  # Added 'Ask' button
    if user_input:
        if "```python" in user_input and "```" in user_input:
            # Extract and run Python code
            code = user_input.split("```python")[1].split("```")[0].strip()
            output = execute_code(code)

            if "error" in output:
                response = f"⚠️ Error in code execution:\n```\n{output['error']}\n```"
            else:
                response = "✅ Code executed successfully!"
                
                # Display variables
                for var_name, value in output.items():
                    if isinstance(value, pd.DataFrame):
                        st.write(f"**{var_name} (DataFrame):**")
                        st.dataframe(value)
                    elif isinstance(value, plt.Figure):
                        st.pyplot(value)
                    else:
                        st.write(f"**{var_name}** = {value}")

        else:
            # Generate AI response
            try:
                ai_response = model.generate_content(user_input)
                response = ai_response.text if ai_response and hasattr(ai_response, 'text') else "❌ No valid response from AI."
            except Exception as e:
                response = f"⚠️ AI response error:\n```\n{traceback.format_exc()}\n```"

        # Store chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
