# text_rag_api.py
import os
import uuid
import logging
import argparse
import requests
from typing import List

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse

from google.cloud import aiplatform
from langchain.prompts import PromptTemplate
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_vertexai import (
    ChatVertexAI,
    VertexAI,
    VertexAIEmbeddings,
    VectorSearchVectorStore,
)
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

# ---------------------------------------------------------------------
# Config & Vertex AI init
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID  = os.getenv("PROJECT_ID",  "mycompany-npe")
LOCATION    = os.getenv("LOCATION",    "us-central1")
GCS_BUCKET  = os.getenv("GCS_BUCKET",  "mycompany-npe-dev-bucket")
GCS_BUCKET_URI = f"gs://{GCS_BUCKET}"

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=GCS_BUCKET_URI)

CHAT_MODEL_NAME      = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "text-embedding-005"
CHAT_TOKEN_LIMIT     = 8192
EMBED_TOKEN_LIMIT    = 2048
TOKEN_LIMIT          = min(CHAT_TOKEN_LIMIT, EMBED_TOKEN_LIMIT)
DIMENSIONS           = 768   

# ---------------------------------------------------------------------
# Helper: summarise large chunks before embedding
# ---------------------------------------------------------------------
def generate_text_summaries(texts: List[str], summarise: bool = True) -> List[str]:
    prompt = PromptTemplate.from_template(
        "You are an assistant tasked with summarising text for retrieval. "
        "Give a concise summary optimised for retrieval. Text: {element}"
    )
    llm      = VertexAI(model_name=CHAT_MODEL_NAME, temperature=0, max_output_tokens=TOKEN_LIMIT)
    pipeline = {"element": lambda x: x} | prompt | llm | StrOutputParser()

    return pipeline.batch(texts, {"max_concurrency": 1}) if summarise else texts

# ---------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------
app = FastAPI(title="Text-only RAG API", version="1.0")

@app.post("/rag-query")
async def rag_query(pdf_file: UploadFile, query: str = Form(...)):
    logger.info("Processing PDF and query...")

    # ------------------ 1. Save uploaded PDF temporarily ------------------
    tmp_pdf = f"tmp_{uuid.uuid4()}.pdf"
    with open(tmp_pdf, "wb") as f:
        f.write(await pdf_file.read())

    # ------------------ 2. Parse PDF into raw text elements ---------------
    raw_elements = partition_pdf(
        filename=tmp_pdf,
        extract_images_in_pdf=False, # No images in this example
        infer_table_structure=False, # no tables
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )
    texts = [str(elem) for elem in raw_elements]

    # ------------------ 3. Split into <≈4 k-token chunks ------------------
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=10000, chunk_overlap=0)
    large_text = " ".join(texts)
    text_chunks = splitter.split_text(large_text)

    # ------------------ 4. Summarise each chunk for embedding ------------
    summaries = generate_text_summaries(text_chunks, summarise=True)

    # ------------------ 5. Build Vector Store + Retriever -----------------
    vectorstore = VectorSearchVectorStore.from_components(
        project_id=PROJECT_ID,
        region=LOCATION,
        gcs_bucket_name=GCS_BUCKET,
        index_id="7153759100868755456",
        endpoint_id="8909248161868939264",
        embedding=VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME),
        stream_update=False,
    )
    docstore  = InMemoryStore()
    id_key    = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
    )

    doc_ids = [str(uuid.uuid4()) for _ in text_chunks]
    summary_docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(summaries)
    ]

    retriever.docstore.mset(list(zip(doc_ids, text_chunks)))
    retriever.vectorstore.add_documents(summary_docs)

    # ------------------ 6. Build simple RAG chain ------------------------
    prompt = PromptTemplate.from_template(
        "Use the following retrieved context to answer the question.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | ChatVertexAI(model_name=CHAT_MODEL_NAME, temperature=0, max_output_tokens=TOKEN_LIMIT)
        | StrOutputParser()
    )

    answer = chain.invoke(query)

    # ------------------ 7. Cleanup + respond -----------------------------
    os.remove(tmp_pdf)
    return JSONResponse(content={"response": answer})

# ---------------------------------------------------------------------
# CLI helper for local testing
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run text-only RAG locally.")
    parser.add_argument("--pdf-path", required=True, help="Path to a PDF file")
    parser.add_argument("--query", required=True, help="Query for the RAG system")
    args = parser.parse_args()

    with open(args.pdf_path, "rb") as f:
        files = {"pdf_file": (os.path.basename(args.pdf_path), f, "application/pdf")}
        data = {"query": args.query}
        resp = requests.post("http://localhost:8000/rag-query", files=files, data=data)
    print(resp.json())

if __name__ == "__main__":
    main()
