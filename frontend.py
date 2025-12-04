# import streamlit as st
# from backend import build_vector_db_from_pdfs, answer_query
# import os

# st.set_page_config(page_title="Multi-Modal RAG System", layout="wide")

# st.title("📘 Multi-Modal Retrieval-Augmented QA System")

# # Create folders if not exist
# os.makedirs("uploaded_files", exist_ok=True)
# os.makedirs("vector_store", exist_ok=True)

# # Sidebar — Upload PDFs
# st.sidebar.header("📂 Upload Documents")
# uploaded_files = st.sidebar.file_uploader(
#     "Upload PDF files",
#     type=["pdf"],
#     accept_multiple_files=True
# )

# build_btn = st.sidebar.button("Build Vector Store")

# if build_btn:
#     if not uploaded_files:
#         st.sidebar.error("Please upload at least one PDF.")
#     else:
#         st.sidebar.success("Processing PDFs...")
#         paths = []
#         for f in uploaded_files:
#             path = f"uploaded_files/{f.name}"
#             with open(path, "wb") as out:
#                 out.write(f.read())
#             paths.append(path)

#         st.session_state.vector_db = build_vector_db_from_pdfs(paths, "vector_store")
#         st.sidebar.success("✅ Vector DB created!")

# # Chat Interface
# st.subheader("💬 Chat with the Document")

# if "vector_db" not in st.session_state:
#     st.warning("Please upload PDFs and build the vector DB first.")
# else:
#     user_input = st.text_input("Ask a question based on the document")

#     if st.button("Ask"):
#         if user_input.strip():
#             answer = answer_query(st.session_state.vector_db, user_input)
#             st.write("### 📌 Answer")
#             st.write(answer)
import streamlit as st
from backend import build_vector_db_from_pdfs, answer_query
import os

st.set_page_config(page_title="Multi-Modal RAG System", layout="wide")

st.title("📘 Multi-Modal Retrieval-Augmented QA System")

# Create folders if not exist
os.makedirs("uploaded_files", exist_ok=True)
os.makedirs("vector_store", exist_ok=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar — Upload PDFs
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

build_btn = st.sidebar.button("Build Vector Store")

if build_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        st.sidebar.success("Processing PDFs...")
        paths = []
        for f in uploaded_files:
            path = f"uploaded_files/{f.name}"
            with open(path, "wb") as out:
                out.write(f.read())
            paths.append(path)

        st.session_state.vector_db = build_vector_db_from_pdfs(paths, "vector_store")
        st.sidebar.success("✅ Vector DB created!")

# Chat Interface
st.subheader("💬 Chat with the Document")

if "vector_db" not in st.session_state:
    st.warning("Please upload PDFs and build the vector DB first.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🧑‍💻 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg['content']}")

    # User input
    user_input = st.text_input("Ask a question based on the document")

    if st.button("Ask"):
        if user_input.strip():
            # Save user question
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Generate answer
            answer = answer_query(st.session_state.vector_db, user_input)

            # Save assistant answer
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Rerun to refresh UI
            st.rerun()
