# Medical Chatbot: AI‑Powered Transcription, Summarization, and Q&A System

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Architecture](#architecture)  
4. [Components](#components)  
5. [Requirements](#requirements)  
6. [Installation & Setup](#installation--setup)  
7. [Usage](#usage)  
   - [Fine‑Tune Whisper](#fine-tune-whisper)  
   - [Launch Streamlit UI](#launch-streamlit-ui)    
8. [Configuration](#configuration)  
9. [Contributing](#contributing)    

---

## Project Overview
A modular, scalable system that ingests medical audio (doctor–patient dialogues, prescriptions) and PDFs (reports, instructions) to deliver:
- **Accurate medical transcription** (fine‑tuned ASR)  
- **Concise summarization** of lengthy documents or conversations  
- **Multilingual translation** (text & audio)  
- **Context‑aware Q&A** via retrieval‑augmented generation  

This solution streamlines clinical workflows, breaks language barriers, and provides an interactive assistant for healthcare professionals.

---

## Features

### 1. Medical ASR Transcription  
Converts spoken medical conversations into accurate text.  
- **Fine‑Tuned Whisper:** Adapted on a medical audio dataset (30 s segments) to reduce WER on domain‑specific terms citeturn0file2.  
- **Chunking Pipeline:** Audio is split with 10 % overlap to preserve context, then transcribed via `transcribe.py` citeturn0file6.  
- **Benefit:** Up to 30 % WER reduction on medical jargon; faster, more reliable charting.

### 2. Document & Audio Summarization  
Automatically condenses text or speech into high‑level overviews.  
- **Prompt Templates:** Custom LangChain prompts (`pdf_summarization`, `audio_summarization`) ensure focus on essential facts without hallucination citeturn0file4.  
- **Merged Summaries:** Combines chunk‑level summaries into a cohesive narrative.  
- **Benefit:** Saves clinicians significant reading time and highlights key insights.

### 3. Layout‑Preserving Multilingual Translation  
Translates PDFs and audio transcripts while keeping original formatting.  
- **PDFs:** Pages scaled (default 1.2×) to fit translated text, overlaid with white blocks, and replaced by HTML blocks under Optional Content Groups citeturn0file7.  
- **Audio:** Line‑numbered translated transcripts via Deep Translator.  
- **Benefit:** Maintains the integrity of medical forms, prescriptions, and reports.

### 4. Interactive Multimodal Q&A  
Allows free‑form queries against uploaded files.  
- **RAG Pipeline:** Retrieves top‑k text & table chunks (MMR), fetches relevant images via CLIP, merges them, and prompts Qwen2‑VL citeturn0file4.  
- **Benefit:** Grounded, context‑aware answers; reduces hallucinations; supports text, tables, and images.

### 5. Vector‑Based Retrieval & Storage  
Enables efficient similarity search across text, images, and tables.  
- **Embeddings:**  
  - Text → BGE (768‑dim)  
  - Images → CLIP (512‑dim) citeturn0file1  
  - Tables → formatted text → embeddings  
- **Storage:** Qdrant collections with rich metadata (`filename`, pixels, table_text) citeturn0file5.  
- **Benefit:** Sub‑second retrieval for large datasets; unified multimodal search.

### 6. Streamlit Web UI  
User-friendly interface orchestrating the entire pipeline.  
- **File Upload:** PDF/MP3 triggers full processing.  
- **Controls:**  
  - 🌐 Translate  
  - 📝 Summarize  
  - 🔄 Restart  
- **Chat Interface:** Free‑form queries invoke RAG & response generation citeturn0file3.  
- **Benefit:** No‑code access for non‑technical users; real‑time feedback.

---

## Architecture
1. **Content Extraction** (`content_extract.py`): Parse PDFs into text, images, tables.  
2. **Embedding Generation** (`embeddings.py`): Vectorize text, images, tables.  
3. **Vector Storage** (`store_embeddings.py`): Persist embeddings in Qdrant.  
4. **Retrieval** (`retrieve.py`): Fetch relevant vectors & merge results.  
5. **Response Generation** (`retrieve.py`): Qwen2‑VL crafts final answers.  
6. **UI Layer** (`main.py`): Streamlit app for file upload, chat, and controls.

---

## Components

### A. PDF Content Extraction (`content_extract.py`)  
Extracts text, embedded images (resized 512×512), and tables (Camelot lattice) from PDFs; returns a unified dict:  
```py
content = {
  "text": "...",
  "images": ["page_1_image_1.png", ...],
  "tables": [{"table_id":1, "data":[{...}, ...]}, ...]
}
```  

### B. Embeddings Utility (`embeddings.py`)  
- **split_text:** Overlapping chunks via LangChain’s RecursiveCharacterTextSplitter  
- **image_generate_embeddings:** CLIP embeddings, returns vectors + raw pixel data + size  
- **format_table_for_embedding / generate_table_embeddings:** Converts tables to text & embeds them  

### C. Whisper Fine‑Tuning (`fine-tune-whisper.py`)  
- Loads and cleans audio/transcripts  
- Splits into ≤30 s segments  
- Prepares log‑Mel features & token labels  
- Trains with `Seq2SeqTrainer`, monitors WER, saves best model (`whisper-small-eng-v2`)  

### D. Audio Transcription & Translation (`transcribe.py`)  
- **split_audio:** 10 % overlap chunking  
- **transcribe:** Fine‑tuned Whisper inference  
- **save_transcription:** Writes `transcriptions.txt`  
- **translate_audio:** Outputs `Content_translated.txt` via Deep Translator  

### E. PDF Translation & Resizing (`translate.py`)  
- **resize_pdf:** Scales pages by factor (default 1.2)  
- **translate_pdf:** Overlays & replaces text blocks with translated HTML under OCG layers  

### F. Embedding Storage (`store_embeddings.py`)  
- **create_collection:** Manages Qdrant collections  
- **store_text:** LangChain Qdrant wrapper  
- **store_image_embeddings:** Upserts image vectors + metadata  
- **store_table_embeddings:** Uploads table vectors + payload  

### G. Retrieval & Response (`retrieve.py`)  
- **Prompt Templates:** Q&A (`prompt`), PDF & audio summarization  
- **retrieve_text / table_retrieve:** LangChain retrievers  
- **image_retrieval:** CLIP search in Qdrant  
- **reranking:** Merges text & table docs  
- **generate_response:** Feeds prompt + `<|image|>` tokens to Qwen2‑VL, extracts “Answer:” section  

### H. Streamlit App (`main.py`)  
- Initializes models, Qdrant client  
- **Uploader:** PDF → extract/embed/store; MP3 → transcribe/embed/store  
- **Controls:** Translate, Summarize, Restart  
- **Chat:** RAG Q&A with session state

---

## Requirements
```text
Python >=3.8
pymupdf, camelot-py, Pillow, pydub, librosa
transformers, torch, datasets, evaluate, scikit-learn
deep-translator, langchain-huggingface, langchain-community
qdrant-client, streamlit
```

---

## Installation & Setup
```bash
git clone https://github.com/YourUsername/Masters-Project.git
cd Masters-Project
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Download the medical audio dataset from Kaggle and place under `../audio_recordings`.

---

## Usage

### Fine‑Tune Whisper
```bash
python fine-tune-whisper.py
```
Generates `./whisper-small-eng-v2` with model & processor.

### Launch Streamlit UI
```bash
streamlit run main.py
```
- Upload PDF/MP3  
- Use 🌐 Translate, 📝 Summarize, 🔄 Restart  
- Enter queries in chat to interact with your files.

---

## Configuration
- **Qdrant:** In‑memory by default (`":memory:"`), change URI in `main.py` for persistence.  
- **Translation:** Uses `GoogleTranslator`, no API key needed.  
- **Models:** Auto‑download via Hugging Face Transformers.

---

## Contributing
Contributions welcome! Please open issues or PRs, follow existing style, and update tests/documentation.

---