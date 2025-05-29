import streamlit as st
from inference.qa_pipeline import get_answer

st.set_page_config(page_title="Q/A Chatbot", page_icon="🤖")

st.title("🧠 Distilled Q/A Chatbot")
st.markdown("Ask a question based on the provided context passage.")

context = st.text_area("📘 Context:", height=200)
question = st.text_input("❓ Question:")

if st.button("Get Answer"):
    if not context.strip() or not question.strip():
        st.warning("Please enter both context and question.")
    else:
        answer = get_answer(question, context)
        st.success(f"💬 **Answer**: {answer}")
