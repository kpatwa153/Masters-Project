import sys
import types
import torch

if "torch.classes" in sys.modules:
    if isinstance(sys.modules["torch.classes"], types.ModuleType):
        del sys.modules["torch.classes"]

from io import BytesIO
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

# Qdrant Client and Models
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
# Transformers for Models
from transformers import CLIPProcessor, CLIPModel
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info

import librosa
import pymupdf
import streamlit as st
from content_extract import pdf_content_extraction
from deep_translator import GoogleTranslator
from transcribe import (
    save_transcription,
    split_audio,
    transcribe,
    translate_audio,
)
from embeddings import split_text, image_generate_embeddings, generate_table_embeddings
from store_embeddings import store_text, store_image_embeddings, store_table_embeddings
from retrieve import prompt,pdf_summarization, audio_summarization, retrieve_text, image_retrieval, table_retrieve, reranking, generate_response
from translate import resize_pdf, translate_pdf

# import random
# import numpy as np
# import torch

# SEED = 42  # or any fixed number

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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
    st.session_state.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    st.session_state.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Qwen model + processor
    st.session_state.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto"
    )
    st.session_state.qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.session_state.qwen_model.to(st.session_state.device)
    st.session_state.qwen_model.eval()


# App title
st.set_page_config(page_title="Medical Bot", layout="wide")
st.title("LLM-Powered File Assistant")

# Session state to track conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file" not in st.session_state:
    st.session_state.file = None
if "translation_mode" not in st.session_state:
    st.session_state.translation_mode = False

# Upload file (now at top center)
uploaded_file = st.file_uploader(
    "📁 Upload a PDF or MP3 file to begin", type=["pdf", "mp3"]
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
                    st.session_state.text_vector_store = store_text(text_chunks, st.session_state.text_embedding_model, TEXT_COLLECTION)
                    st.session_state.text_chunks = text_chunks
                    st.text_embedding_store = True
                if "image_embedding_store" not in st.session_state:
                    if st.session_state.content["images"]:
                        image_embeddings, px, size = image_generate_embeddings(
                            st.session_state.content["images"], st.session_state.clip_processor, st.session_state.clip_model
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
                        table_embeddings = generate_table_embeddings(st.session_state.content, st.session_state.text_embedding_model)
                        store_table_embeddings(
                            st.session_state.qdrant_client, TABLE_COLLECTION, TABLE_VECTOR_SIZE, table_embeddings
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
                    text_chunks = split_text(" ".join(st.session_state.transcriptions))
                    st.session_state.text_vector_store = store_text(text_chunks, st.session_state.text_embedding_model, TEXT_COLLECTION)
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
            partial_summary = []
            if file_type == "application/pdf":
                pdf_summary_prompt = pdf_summarization()
                partial_summary.append(generate_response(st.session_state.qwen_processor, st.session_state.qwen_model,
                    st.session_state.device, pdf_summary_prompt,
                    "Summarize the Information",
                    st.session_state.content["text"]))
            if file_type == "audio/mpeg":
                audio_summary_prompt = audio_summarization()
                partial_summary.append(generate_response(st.session_state.qwen_processor, st.session_state.qwen_model,
                    st.session_state.device, audio_summary_prompt,
                    "Summarize the Information",
                    st.session_state.transcriptions))
            final_summary = "\n".join(f"- {s}" for s in partial_summary)
            st.session_state.chat_history.append(("You", "Summarize the document"))
            st.session_state.chat_history.append(("Bot", final_summary))
            

    with col3:
        if st.button("🔄 Restart"):
            st.session_state.chat_history = []
            st.session_state.file = None
            st.session_state.translation_mode = False
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
        if "promt_template" not in st.session_state:
            st.session_state.prompt_template = prompt()
        output_text = ""
        with st.spinner("Generating Response..."):
            if file_type == "application/pdf":
                try:
                    table_retriever = table_retrieve(st.session_state.qdrant_client, TABLE_COLLECTION, st.session_state.text_embedding_model)
                    image_results = image_retrieval(
                        st.session_state.qdrant_client, user_input, st.session_state.clip_processor, st.session_state.clip_model, IMAGE_COLLECTION
                    )
                except ValueError as e:
                    st.warning("⚠️ Table collection not found. Skipping table retrieval.")
                    table_retriever = None
                relevant_text = retrieve_text(st.session_state.text_vector_store)
                # table_retriever = table_retrieve(client, TABLE_COLLECTION, text_embedding_model)
                # Use reranking with conditional retrievers
                output_text = reranking(user_input, relevant_text, table_retriever or None)

                # Use generate_response with optional image input
                response = generate_response(
                    st.session_state.qwen_processor, 
                    st.session_state.qwen_model, 
                    st.session_state.device,
                    st.session_state.prompt_template,
                    user_input,
                    output_text,
                    image_path=image_results if image_results else None
                )
            elif file_type == "audio/mpeg":
                relevant_text = retrieve_text(st.session_state.text_vector_store)
                response = generate_response(
                    st.session_state.qwen_processor, 
                    st.session_state.qwen_model, 
                    st.session_state.device,
                    st.session_state.prompt_template,
                    user_input,
                    relevant_text,
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
