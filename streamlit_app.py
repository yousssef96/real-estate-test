# ===================================================================
# Libraries
# ===================================================================

import streamlit as st
import requests
import os


# ===================================================================
# CONFIGURATION & API CONNECTIVITY
# ===================================================================

API_BASE_URL = st.secrets.get("API_URL", "http://localhost:8000")

# ===================================================================
# STREAMLIT UI
# ===================================================================


def main():
    st.set_page_config(page_title="مساعد العقارات", page_icon="🏠", layout="centered")
    
    # Custom CSS for Right-to-Left (RTL) Arabic support
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    html, body, [data-testid="stSidebar"], .main {
        font-family: 'Cairo', sans-serif;
        direction: RTL;
        text-align: right;
    }
    .stTextArea textarea {
        direction: RTL;
        text-align: right;
    }
    div[data-testid="stMarkdownContainer"] p {
        text-align: right;
    }
    button {
        direction: RTL;
    }
    /* Fixing the chat input direction */
    .stChatInputContainer textarea {
        direction: RTL;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏠 مساعد مدينتي العقاري")

    # Sidebar Admin
    with st.sidebar:
        st.header("لوحة التحكم")
        if st.button("🔄 تحديث البيانات من Airtable"):
            with st.spinner("جاري تحديث قاعدة البيانات..."):
                try:
                    # Calling the FastAPI /sync endpoint
                    response = requests.post(f"{API_BASE_URL}/sync")
                    if response.status_code == 200:
                        st.success("تم التحديث بنجاح!")
                    else:
                        st.error(f"فشل التحديث: {response.text}")
                except Exception as e:
                    st.error(f"تعذر الاتصال بالسيرفر: {e}")

    # Chat Interface Logic
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat Input
    if prompt := st.chat_input("اسأل عن الشقق المتاحة..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            # Placeholder for the streaming response
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # 1. Send POST request to FastAPI /chat with streaming enabled
                with requests.post(
                    f"{API_BASE_URL}/chat", 
                    json={"query": prompt}, 
                    stream=True
                ) as r:
                    if r.status_code == 200:
                        # 2. Iterate over chunks coming from the API
                        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                full_response += chunk
                                # 3. Update UI in real-time to mimic typing
                                response_placeholder.markdown(full_response + "▌")
                        
                        # Final update without the cursor
                        response_placeholder.markdown(full_response)
                    else:
                        st.error("السيرفر لا يستجيب حالياً. حاول لاحقاً.")
            
            except Exception as e:
                st.error(f"حدث خطأ في الاتصال: {str(e)}")
            
        # Store the complete answer in session history
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":

    main()

