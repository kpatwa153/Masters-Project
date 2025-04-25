"""
Audio Transcription and Translation Utilities using Whisper

This module provides a complete pipeline for processing long-form audio:
splitting it into overlapping segments, transcribing with a fine-tuned Whisper model,
and translating the transcription into another language. The outputs can be saved
to text files for downstream tasks like summarization, captioning, or multilingual content creation.

Key Functionalities:
--------------------

1. **Audio Chunking**
   - `split_audio(audio, chunk_samples)`:
     Splits long audio into overlapping chunks (10% overlap) to preserve context and improve transcription continuity.

2. **Audio Transcription**
   - `transcribe(audio_chunks)`:
     Generates text transcriptions from a list of audio chunks using a locally fine-tuned Whisper model.
     Assumes mono audio at 16kHz sampling rate.

3. **Transcription Saving**
   - `save_transcription(transcriptions)`:
     Saves transcriptions to a file named `transcriptions.txt` with line numbers for readability.

4. **Translation of Transcripts**
   - `translate_audio(transcriptions, translator)`:
     Translates the list of transcribed English texts to a target language using a translator object (e.g., Deep Translator).
     Saves output to `Content_translated.txt`.

Expected Use Cases:
-------------------
- Speech-to-text applications
- Podcast/audio/video captioning
- Multilingual media workflows
- Real-time or batch translation of audio content

Dependencies:
-------------
- `transformers` (for Whisper model and processor)
- `torch` (for model inference)
- `os` (for model path handling)
- Optional: Deep Translator for multilingual support

Model Notes:
------------
- Uses a locally fine-tuned version of `"whisper-small-eng"` stored at `../whisper-small-eng`.
- The transcription assumes English audio input by default but can be adapted to multilingual Whisper.

"""

import os

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load fine-tuned model
model_path = os.path.abspath("../whisper-small-eng")
model = WhisperForConditionalGeneration.from_pretrained(model_path)
processor = WhisperProcessor.from_pretrained(model_path, use_fast=True)


# Ensure the model is in evaluation mode
model.eval()


def split_audio(audio, chunk_samples):
    """
    Splits a long audio signal into overlapping chunks for transcription.

    This function divides the input audio into smaller chunks with a 10% overlap
    between consecutive segments. This overlap helps preserve context and
    ensures smoother transitions in transcription results.

    Args:
        audio (np.ndarray): The raw audio waveform array (1D).
        chunk_samples (int): Number of samples per chunk.

    Returns:
        List[np.ndarray]: A list of audio chunks with overlap.
    """
    stride = int(0.1 * chunk_samples)  # 10% overlap to ensure smooth merging
    chunks = [
        audio[i : i + chunk_samples]
        for i in range(0, len(audio), chunk_samples - stride)
    ]
    return chunks


def transcribe(audio_chunks):
    """
    Transcribes a list of audio chunks using a fine-tuned OpenAI Whisper model.

    This function processes audio chunks (pre-split from longer audio) and generates
    transcriptions using a custom fine-tuned version of the Whisper model. It assumes
    the audio is mono and sampled at 16kHz.

    Args:
        audio_chunks (List[np.ndarray]): List of audio segments as numpy arrays.

    Returns:
        List[str]: List of transcribed strings, one per audio chunk.
    """
    transcriptions = list()
    for ind, chunk in enumerate(audio_chunks):
        input_features = processor(
            chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features
        with torch.no_grad():
            predicted_ids = model.generate(input_features, language="en")

        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        transcriptions.append(transcription)
    return transcriptions


def save_transcription(transcriptions):
    """
    Saves a list of audio transcriptions to a text file.

    This function writes each transcription to a file named 'transcriptions.txt',
    numbering each entry for readability. It is typically used after transcribing
    audio using a model like Whisper.

    Args:
        transcriptions (List[str]): List of transcribed text segments.

    Returns:
        str: Path to the saved transcription file.
    """
    output_file = "transcriptions.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        for ind, text in enumerate(transcriptions):
            file.write(f"{ind+1}: {text} \n\n")


def translate_audio(transcriptions, translator):
    """
    Translates a list of transcriptions and writes the output to a text file.

    This function takes in audio transcriptions (in English) and translates them
    to the target language using the provided translator object. The translated
    text is saved line by line in an output file.

    Args:
        transcriptions (List[str]): List of transcribed strings from audio.
        translator (Callable): A translator object with a `.translate(text)` method.

    Returns:
        str: Path to the saved translation output file.
    """
    output_file_translated = (
        "Content_translated.txt"  # File to save translations
    )
    with open(output_file_translated, "w", encoding="utf-8") as file:
        for ind, i in enumerate(transcriptions):
            translation = translator.translate(
                i
            )  # Translate each transcription
            file.write(
                f"{ind+1}: {translation} \n\n"
            )  # Write each translation to the file
