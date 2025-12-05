import os
import pdfplumber
import pytesseract
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


# ---------- Configuration ----------
GEMINI_API_KEY = "AIzaSyCpfS1czLljxwPsAvBUABiteuhEozFoIwQ"

# Embeddings & LLM models (adjust names to valid ones)


GEMINI_MODEL_NAME = "gemini-2.5-flash"           # correct Gemini model name

# Chunking parameters
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MAX_PAGE_CHARS = 200_000

# ---------- Client init ----------
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.0
)




def table_to_summary(table):
    """
    Converts a pdfplumber table → short summary text.
    """
    try:
        if len(table) > 1:
            df = pd.DataFrame(table[1:], columns=table[0])
            cols = list(df.columns)
            sample = df.head(1).to_dict(orient="records")
            return f"[TABLE] columns={cols} sample={sample}"
        else:
            return f"[TABLE] raw={table}"
    except:
        return "[TABLE] (unreadable)"


def extract_images_with_ocr(pdf_path):
    """
    Extract images → OCR → produce text chunks.
    """
    images_text = []
    pages = convert_from_path(pdf_path)

    for idx, img in enumerate(pages, 1):
        try:
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                images_text.append((ocr_text, idx))
        except:
            pass

    return images_text


def extract_pages_safe(pdf_path):
    """
    Extract text + tables from PDF pages.
    """
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""

            if len(text) > MAX_PAGE_CHARS:
                text = text[:MAX_PAGE_CHARS] + "\n[TRUNCATED]"

            # Extract tables
            table_summaries = []
            try:
                tables = page.extract_tables() or []
                for t in tables:
                    table_summaries.append(table_to_summary(t))
            except:
                pass

            pages.append((text, page.page_number, table_summaries))

    return pages


# ---------------------------------------------------
# MULTI-MODAL INGESTION + VECTOR DB
# ---------------------------------------------------

def build_vector_db_from_pdfs(pdf_filepaths, persist_path=None):
    """
    Full multi-modal ingestion:
    - Extract text
    - Extract tables
    - Extract images + OCR text
    - Chunk everything
    - Build FAISS vector DB
    """
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    for path in pdf_filepaths:
        basename = os.path.basename(path)

        # 1. Text + tables
        pages = extract_pages_safe(path)

        for text, page_no, table_summaries in pages:
            combined = text
            if table_summaries:
                combined += "\n\n" + "\n".join(table_summaries)

            doc = Document(
                page_content=combined,
                metadata={"source": basename, "page": page_no, "modality": "text/table"}
            )

            chunks = splitter.split_documents([doc])
            all_docs.extend(chunks)

        # 2. Extract image OCR text
        ocr_results = extract_images_with_ocr(path)
        for ocr_text, page_no in ocr_results:
            doc = Document(
                page_content="[OCR_IMAGE]\n" + ocr_text,
                metadata={"source": basename, "page": page_no, "modality": "image"}
            )
            chunks = splitter.split_documents([doc])
            all_docs.extend(chunks)

    if not all_docs:
        return None

    # Build unified multi-modal vector DB
    vector_db = FAISS.from_documents(all_docs, embeddings_model)

    if persist_path:
        vector_db.save_local(persist_path)

    return vector_db


# ---------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------

def retrieve_top_docs(vector_db, query, k=5):
    if vector_db is None:
        return []
    return vector_db.similarity_search(query, k=k)


def answer_query(vector_db, query, k=5):
    hits = retrieve_top_docs(vector_db, query, k)
    if not hits:
        return "No relevant information found."

    context = ""
    for d in hits:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        modality = d.metadata.get("modality", "text")

        context += f"[Source: {src} | Page: {page} | {modality}]\n{d.page_content}\n\n"

    prompt = f"""
Use ONLY the context below to answer the question.
Cite sources exactly like this: (Source: filename | Page: N)

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content


# ---------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------

if __name__ == "__main__":
    pdf_files = ["qatar_test_doc.pdf"]

    print("Building multi-modal vector DB (text + tables + OCR images)...")
    vector_db = build_vector_db_from_pdfs(pdf_files)
    print("Vector DB created successfully!")

    user_q = "What is the main objective of the assignment?"
    print("\nUser Question:", user_q)
    print("\nAnswer:")
    print(answer_query(vector_db, user_q))

