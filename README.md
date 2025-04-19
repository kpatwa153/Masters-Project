# Masters-Project

### **1️⃣ Project Title & Description**  
🔹 **Title:** `Medical Chatbot: AI-Powered Transcription, Summarization, and Q&A System`  
🔹 **Description:**  
   - A multimodal AI chatbot designed for the **medical field** that processes **audio and PDFs** to provide:  
     - ✅ **Medical Transcription** (ASR for doctor-patient conversations, prescriptions)  
     - ✅ **Summarization** (Extract key insights from medical documents & audio)  
     - ✅ **Multilingual Translation**  
     - ✅ **Interactive Medical Q&A**

---

### **2️⃣ Project Architecture**  

![Architecture](image.png)

---

### **3️⃣ Features**  
✅ **Speech-to-Text Transcription**
✅ **Medical Text Summarization**  
✅ **Multilingual Translation**  
✅ **Interactive Q&A using LLM + RAG**   
✅ **Scalable & Modular AI Architecture**  

---

## 4️⃣ OpenAI Whisper Fine-Tuning

Fine-tuning OpenAI Whisper is crucial for adapting the model to the specific needs of medical transcription. General ASR models may struggle with domain-specific terminology, abbreviations, and nuanced speech patterns used in doctor-patient conversations. By fine-tuning Whisper on medical audio data, we improve its ability to:
- Accurately transcribe medical terms and prescriptions.
- Reduce errors caused by complex jargon and accents.
- Enhance usability for real-world clinical applications.

Dataset used for Fine-tuning: https://www.kaggle.com/datasets/najamahmed97/audio-recording-whisper

Fine-tuning OpenAI Whisper enhances its transcription accuracy for medical terminology. The process involves:

- **Data Preparation:**
  - Audio recordings segmented into 30-second chunks.
  - Transcripts cleaned and normalized.
  - Dataset formatted for Hugging Face compatibility.

- **Model Configuration:**
  - Base Model: `openai/whisper-small`
  - Feature Extractor: Log-Mel spectrograms extraction.
  - Tokenizer: Text tokenization for ASR processing.

- **Preprocessing Steps:**
  - Removing special characters and standardizing text.
  - Splitting long audio files using **pydub**.
  - Applying **WhisperFeatureExtractor** and **WhisperTokenizer**.

This fine-tuning ensures better adaptation to specialized medical conversations, improving transcription reliability and usability.

### 5️⃣ PDF Translation

The `translate.py` module ensures that **medical documents**—such as prescriptions, diagnostic reports, and patient instructions, can be **translated into multiple languages** while preserving their **original layout, structure, and visual formatting**.

This module enables the chatbot to serve **multilingual patients and practitioners**, expand **accessibility** across regions, and comply with **language-inclusive healthcare standards**.

---

#### 🎯 Purpose:

The goal of this component is to:

- **Bridge language barriers** in medical communication by translating formal medical PDFs into regional or international languages.
- Preserve **visual structure and formatting** to maintain medical document integrity and readability post-translation.
- Support downstream tasks like **summarization** and **semantic search**, by producing clean and structured translated documents.

---

#### ⚙️ Functional Overview

##### 🔧 `resize_pdf(input_pdf, output_pdf, scale_factor=1.2)`  
Increases the page size of the original PDF to ensure there's enough space for translated text, which is often longer than the source language (e.g., English → French). This step preserves **readability and avoids text overflow**.

##### 🌐 `translate_pdf(input_pdf_path, output_pdf_path, language, translator)`  
Translates extracted text blocks using an external translator like `GoogleTranslator`. Then it:
- Covers original text with white overlays.
- Replaces it with **styled, localized content** via HTML rendering.
- Adds an **Optional Content Group (OCG)** for clean layer-based rendering and toggling.

---

#### 🧰 Dependencies

- `pymupdf` → PDF parsing, editing, and rendering  
- `deep_translator` → Translation API wrapper (e.g., Google Translate)  
- `os` → File I/O operations



### 6️⃣ Audio Transcription & Translation

The `transcribe.py` module powers the **speech processing capabilities** of the Medical Chatbot System, enabling it to **understand, transcribe, and translate** long-form clinical audio such as:

- Doctor–patient consultations  
- Verbal prescriptions  
- Patient-reported symptoms  
- Healthcare instructions  

By leveraging a **locally fine-tuned Whisper model**, the system converts spoken medical content into structured multilingual text, facilitating **summarization**, **Q&A**, and **language-inclusive documentation**.

---

#### 🎯 Purpose Within the Project

This component enables the chatbot to:

- Support **medical transcription** workflows with **high-quality ASR** (Automatic Speech Recognition)
- Translate spoken content into multiple languages, enhancing **patient understanding**
- Power downstream tasks like **medical summarization**, and **semantic search** across languages

---

#### ⚙️ Functional Overview

##### 🔊 `split_audio(audio, chunk_samples)`  
Divides long clinical audio into overlapping chunks (10% overlap) to retain contextual continuity during transcription.

##### ✍️ `transcribe(audio_chunks)`  
Performs **speech-to-text conversion** using a Whisper model fine-tuned on medical conversations. Returns a list of English text segments.

##### 💾 `save_transcription(transcriptions)`  
Stores transcribed content into `transcriptions.txt`, with line numbers for organized review or auditing.

##### 🌍 `translate_audio(transcriptions, translator)`  
Translates transcribed English text to a **target language** using an external or local translator. Results are saved to `Content_translated.txt`.

---

#### 🧰 Dependencies

- `transformers` – Whisper model and processor  
- `torch` – Model inference  
- `os` – Path and file operations  
- Optional: `deep_translator` or custom translation API

---
