"""
Medical Chatbot: AI‑Powered Transcription, Summarization, and Q&A System.

This module provides an interactive file assistant that leverages large language models (LLMs) and machine learning techniques for processing, summarizing, translating, and extracting embeddings from uploaded files (PDFs and MP3 audio). It uses Qdrant for vector storage and retrieval, Hugging Face models for text and image embeddings, and integrates various file processing workflows including transcription, translation, and summarization.

Key Features:
- PDF and MP3 file support: Upload and process PDF and audio files.
- Text, image, and table embeddings: Store embeddings of text, images, and tables in Qdrant.
- Translation: Automatically translate PDF and audio files into different languages.
- Summarization: Generate summaries of the document or transcription.
- Chatbot interface: Interact with the file through a chatbot interface to ask questions, get summaries, and retrieve relevant content.

Dependencies:
- torch
- langchain_huggingface
- QdrantClient
- transformers
- librosa
- pymupdf
- streamlit
- content_extract
- deep_translator
- embeddings
- store_embeddings
- retrieve
- translate

Modules:
- `pdf_content_extraction`: Handles PDF content extraction, including text, images, and tables.
- `split_audio`, `transcribe`, `translate_audio`: Audio processing functions including transcription and translation.
- `split_text`, `image_generate_embeddings`, `generate_table_embeddings`: Helper functions to split content and generate embeddings.
- `store_text`, `store_image_embeddings`, `store_table_embeddings`: Functions for storing embeddings in Qdrant.
- `pdf_prompt`, `audio_prompt`, `pdf_summarization`, `audio_summarization`, `retrieve_text`, `image_retrieval`, `table_retrieve`, `reranking`, `generate_response`: Functions for querying and processing the extracted content.
- `resize_pdf`, `translate_pdf`: Functions for resizing and translating PDF files.

Usage:
- Upload PDF or MP3 files and interact with the chatbot interface for various actions like summarization, translation, and querying.
- The application supports a multilingual chatbot, capable of translating content and answering user questions based on the content.

Session State:
- The session state is used to track the file content, embeddings, translation mode, chat history, and other session-related information.

Note:
- This module assumes a pre-configured Qdrant client for storing and retrieving vector embeddings.
"""

import sys
import types

import torch

if "torch.classes" in sys.modules:
    if isinstance(sys.modules["torch.classes"], types.ModuleType):
        del sys.modules["torch.classes"]

import shutil
from io import BytesIO

import librosa
import pymupdf
import streamlit as st
from deep_translator import GoogleTranslator
from langchain_huggingface import HuggingFaceEmbeddings

# Qdrant Client and Models
from qdrant_client import QdrantClient

# Transformers for Models
from transformers import (
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    Qwen2VLForConditionalGeneration,
)

from content_extract import pdf_content_extraction
from embeddings import (
    generate_table_embeddings,
    image_generate_embeddings,
    split_text,
)
from retrieve import (
    audio_prompt,
    audio_summarization,
    generate_response,
    image_retrieval,
    pdf_prompt,
    pdf_summarization,
    reranking,
    retrieve_text,
    table_retrieve,
)
from store_embeddings import (
    store_image_embeddings,
    store_table_embeddings,
    store_text,
)
from transcribe import split_audio, transcribe, translate_audio
from translate import resize_pdf, translate_pdf

CHUNK_DURATION = 30  # 30 seconds per chunk
TEXT_COLLECTION = "text_collection"  # Text Embedding Storage
IMAGE_COLLECTION = "image_collection"  # Image Embedding Storage
TABLE_COLLECTION = "table_collection"  # Table Embedding Storage
IMAGE_VECTOR_SIZE = 512
TABLE_VECTOR_SIZE = 768

if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = QdrantClient(":memory:")

if "qwen_model" not in st.session_state:
    # Text embedding model
    st.session_state.text_embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    # Image embedding model
    st.session_state.clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    st.session_state.clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    # Qwen model + processor
    st.session_state.qwen_model = (
        Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto"
        )
    )
    st.session_state.qwen_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct"
    )

    st.session_state.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    st.session_state.qwen_model.to(st.session_state.device)
    st.session_state.qwen_model.eval()


# App title
st.set_page_config(page_title="Medical Bot", layout="wide")
st.title(
    "Medical Chatbot: AI‑Powered Transcription, Summarization, and Q&A System"
)

# Session state to track conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file" not in st.session_state:
    st.session_state.file = None
if "translation_mode" not in st.session_state:
    st.session_state.translation_mode = False

# Upload file (now at top center)
uploaded_file = st.file_uploader(
    "📁 Upload a PDF or MP3 file to begin",
    type=["pdf", "mp3"],
    key="upload_widget",
)
if uploaded_file:
    st.session_state.file = uploaded_file
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        if "doc" not in st.session_state:
            with st.spinner("Loading PDF file..."):
                pdf_stream = BytesIO(st.session_state.file.read())
                doc = pymupdf.open(
                    stream=pdf_stream, filetype="pdf"
                )  # Load the PDF file
                content = pdf_content_extraction(doc, pdf_stream)
                st.session_state.content = content
                st.session_state.doc = doc

                # Storing Text Embeddings
                if "text_embedding_store" not in st.session_state:
                    text_chunks = split_text(st.session_state.content["text"])
                    st.session_state.text_vector_store = store_text(
                        text_chunks,
                        st.session_state.text_embedding_model,
                        TEXT_COLLECTION,
                    )
                    st.session_state.text_chunks = text_chunks
                    st.text_embedding_store = True
                if "image_embedding_store" not in st.session_state:
                    if st.session_state.content["images"]:
                        image_embeddings, px, size = image_generate_embeddings(
                            st.session_state.content["images"],
                            st.session_state.clip_processor,
                            st.session_state.clip_model,
                        )
                        store_image_embeddings(
                            st.session_state.qdrant_client,
                            st.session_state.content["images"],
                            image_embeddings,
                            px,
                            size,
                            IMAGE_COLLECTION,
                            IMAGE_VECTOR_SIZE,
                        )
                        st.image_embedding_store = True
                    else:
                        st.image_embedding_store = False
                if "table_embedding_store" not in st.session_state:
                    if st.session_state.content["tables"]:
                        table_embeddings = generate_table_embeddings(
                            st.session_state.content,
                            st.session_state.text_embedding_model,
                        )
                        store_table_embeddings(
                            st.session_state.qdrant_client,
                            TABLE_COLLECTION,
                            TABLE_VECTOR_SIZE,
                            table_embeddings,
                        )
                        st.session_state.table_embedding_store = True
                    else:
                        st.session_state.table_embedding_store = False

        st.success("✅ PDF file uploaded.")

    elif file_type == "audio/mpeg":
        if "audio" not in st.session_state:
            with st.spinner("Loading audio file..."):
                audio, sr = librosa.load(st.session_state.file, sr=16000)
                chunk_samples = CHUNK_DURATION * sr
                chunks = split_audio(audio, chunk_samples)
                transcriptions = transcribe(chunks)
                st.session_state.audio = audio
                st.session_state.sr = sr
                st.session_state.transcriptions = transcriptions
                if "text_embedding_store" not in st.session_state:
                    text_chunks = split_text(
                        " ".join(st.session_state.transcriptions)
                    )
                    st.session_state.text_vector_store = store_text(
                        text_chunks,
                        st.session_state.text_embedding_model,
                        TEXT_COLLECTION,
                    )
                    st.text_embedding_store = True
            st.success("✅ MP3 file uploaded.")

    else:
        st.error("❌ Unsupported file type. Please upload a PDF or MP3.")


# Show chat and controls only if file is uploaded
if st.session_state.file:
    st.divider()
    st.subheader("💬 Chat with your file")

    # Control buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🌐 Translate"):
            st.session_state.translation_mode = True

    with col2:
        if st.button("📝 Summarize"):
            with st.spinner("Summarizing......"):
                partial_summary = []
                if file_type == "application/pdf":
                    pdf_summary_prompt = pdf_summarization()
                    images = st.session_state.content.get("images", [])
                    partial_summary.append(
                        generate_response(
                            st.session_state.qwen_processor,
                            st.session_state.qwen_model,
                            st.session_state.device,
                            pdf_summary_prompt,
                            "Summarize the Information",
                            st.session_state.content["text"],
                            images,
                        )
                    )
                if file_type == "audio/mpeg":
                    audio_summary_prompt = audio_summarization()
                    partial_summary.append(
                        generate_response(
                            st.session_state.qwen_processor,
                            st.session_state.qwen_model,
                            st.session_state.device,
                            audio_summary_prompt,
                            "Summarize the Information",
                            st.session_state.transcriptions,
                        )
                    )
            final_summary = "\n".join(f"- {s}" for s in partial_summary)
            st.session_state.chat_history.append(
                ("You", "Summarize the document")
            )
            st.session_state.chat_history.append(("Bot", final_summary))

    with col3:
        if st.button("🔄 Restart"):
            with st.spinner("🔄 Restarting..."):
                shutil.rmtree("extracted_images", ignore_errors=True)
                st.session_state.clear()
                st.session_state.pop("upload_widget", None)
                st.rerun()

    # Language dropdown if Translate was clicked
    if st.session_state.translation_mode:
        languages = GoogleTranslator().get_supported_languages(as_dict=True)
        lang = st.selectbox(
            "Select language to translate to",
            list(languages.keys()),
            key="language_select",
        )
        if st.button("Translate Now"):
            language_code = languages.get(lang.lower())
            translator = GoogleTranslator(
                source="auto", target=language_code
            )  # Configure the translator
            if file_type == "application/pdf":
                with st.spinner("Translating PDF file..."):
                    WHITE = pymupdf.pdfcolor["white"]  # Define color "white"

                    textflags = (
                        pymupdf.TEXT_DEHYPHENATE
                        | pymupdf.TEXT_PRESERVE_WHITESPACE
                    )  # This flag ensures that text will be dehyphenated after extraction.
                    try:
                        resized = resize_pdf(
                            st.session_state.doc, "pdf_resized.pdf"
                        )  # Resize the PDF
                        translate_pdf(
                            resized, f"pdf-{lang}.pdf", lang, translator
                        )  # Translate the PDF

                    except Exception:
                        pass
                st.success(f"**✅ PDF translated to {lang.capitalize()}!**")
            elif file_type == "audio/mpeg":
                translate_audio(st.session_state.transcriptions, translator)
                st.success(
                    f"**✅ Audio transcribed and translated to {lang.capitalize()}!**"
                )

    st.divider()

    # Chat input and history
    user_input = st.text_input("💭 You:", key="user_input")
    if user_input:
        output_text = ""
        with st.spinner("Generating Response..."):
            if file_type == "application/pdf":
                if "pdf_promt_template" not in st.session_state:
                    st.session_state.prompt_template = pdf_prompt()
                try:
                    table_retriever = table_retrieve(
                        st.session_state.qdrant_client,
                        TABLE_COLLECTION,
                        st.session_state.text_embedding_model,
                    )
                    image_results = image_retrieval(
                        st.session_state.qdrant_client,
                        user_input,
                        st.session_state.clip_processor,
                        st.session_state.clip_model,
                        IMAGE_COLLECTION,
                    )
                except ValueError as e:
                    st.warning(
                        "⚠️ Table collection not found. Skipping table retrieval."
                    )
                    table_retriever = None
                relevant_text = retrieve_text(
                    st.session_state.text_vector_store
                )
                # table_retriever = table_retrieve(client, TABLE_COLLECTION, text_embedding_model)
                # Use reranking with conditional retrievers
                output_text = reranking(
                    user_input, relevant_text, table_retriever or None
                )
                response = generate_response(
                    st.session_state.qwen_processor,
                    st.session_state.qwen_model,
                    st.session_state.device,
                    st.session_state.prompt_template,
                    user_input,
                    output_text,
                    image_path=image_results if image_results else None,
                )
            elif file_type == "audio/mpeg":
                if "audio_promt_template" not in st.session_state:
                    st.session_state.prompt_template = audio_prompt()
                text_retriever = retrieve_text(
                    st.session_state.text_vector_store
                )
                docs = text_retriever.invoke(user_input)
                context = " ".join([d.page_content for d in docs])
                response = generate_response(
                    st.session_state.qwen_processor,
                    st.session_state.qwen_model,
                    st.session_state.device,
                    st.session_state.prompt_template,
                    user_input,
                    context,
                )
        bot_response = f"{response}"
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", bot_response))

    # Display chat history
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑‍💻 You:** {msg}")
        else:
            st.markdown(f"**🤖 Bot:** {msg}")
else:
    st.info("Please upload a file to get started with the chatbot.")
