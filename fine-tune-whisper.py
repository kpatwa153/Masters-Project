# Importing the required libraries
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Audio, Dataset
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# Define the max duration per chunk (in seconds)
MAX_DURATION = 30  # Whisper handles max 30s of audio well

# defining the paths for the data
# Paths
dataset_path = "audio_recordings2"
audio_path = os.path.join(dataset_path, "Audio_ecordings")
transcript_path = os.path.join(dataset_path, "transcripts")
audio_folder = "audio_recordings/Audio_recordings"
transcript_folder = "audio_recordings/Clean_Transcripts"


# Combining the audio and transcript files
def load_audio_transcripts(audio_folder, transcript_folder):
    """
    Loads audio file paths and their corresponding transcript text from the given folders.

    Args:
        audio_folder (str): Path to the folder containing audio files.
        transcript_folder (str): Path to the folder containing transcript files.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - "audio" (str): Full path to the audio file.
            - "text" (str): Corresponding transcript text.
    """
    data = []

    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith(".mp3"):
            transcript_file = os.path.splitext(audio_file)[0] + ".txt"
            transcript_path = os.path.join(transcript_folder, transcript_file)

            if os.path.exists(transcript_path):
                with open(
                    transcript_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    transcript = f.read().strip()

                data.append(
                    {
                        "audio": os.path.join(audio_folder, audio_file),
                        "text": transcript,
                    }
                )

    return data


# Preprocessing the transcript data
def clean_text(text):
    """
    Cleans a given text string by performing the following operations:

    1. Replaces newline characters ('\n') with a space.
    2. Removes special characters, keeping only alphanumeric characters and spaces.
    3. Reduces multiple spaces to a single space and trims leading/trailing spaces.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # text = text.lower()  # Convert to lowercase
    text = re.sub(r"\n", " ", text)  # Replace newlines with space
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def split_and_process_audio(df, max_duration=MAX_DURATION):
    """
    Splits each audio file in a dataset into smaller chunks of specified duration
    while maintaining corresponding transcripts.

    Args:
        df (pandas.DataFrame): A DataFrame containing audio file paths and transcripts.
                               Expected columns: "audio" (file path), "text" (transcript).
        max_duration (int, optional): Maximum duration per audio chunk in seconds.
                                      Default is 30 seconds.

    Returns:
        tuple: Two lists:
            - new_audio_paths (list of str): File paths of the split audio chunks.
            - new_transcripts (list of str): Corresponding transcripts for each chunk.
    """
    new_audio_paths = []
    new_transcripts = []

    for idx, row in df.iterrows():
        audio_path = row["audio"]
        transcript = row["text"]

        # Load audio file
        audio = AudioSegment.from_file(audio_path)

        # Process audio in chunks
        for i in range(
            0, len(audio), max_duration * 1000
        ):  # Convert seconds to ms
            chunk = audio[i : i + max_duration * 1000]
            chunk_path = f"{audio_path}_part{i//1000}.mp3"
            chunk.export(chunk_path, format="mp3")  # Save the chunk

            # Assign transcript (currently keeping the same for all chunks)
            chunk_text = transcript

            new_audio_paths.append(chunk_path)
            new_transcripts.append(chunk_text)

    return new_audio_paths, new_transcripts


def prepare_dataset(examples):
    """
    Prepares a dataset by extracting audio features and encoding text labels.

    This function processes each example by:
    1. Extracting log-Mel spectrogram features from the input audio array.
    2. Encoding the target text into tokenized label IDs with padding and truncation.
    3. Removing the original "audio" and "text" fields after processing.

    Args:
        examples (dict): A dictionary containing:
            - "audio" (dict): An audio sample with an "array" key.
            - "text" (str): Corresponding transcript text.

    Returns:
        dict: The processed example with:
            - "input_features" (list): Extracted log-Mel spectrogram features.
            - "labels" (list): Tokenized label IDs.
    """
    audio = examples["audio"]
    examples["input_features"] = feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]
    del examples["audio"]

    sentences = examples["text"]

    # encode target text to label ids
    examples["labels"] = tokenizer(
        sentences, truncation=True, max_length=448, padding="max_length"
    ).input_ids
    del examples["text"]
    return examples


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    A data collator for speech-to-text sequence-to-sequence models that
    applies appropriate padding to both input features and labels.

    This collator:
    1. Pads input audio features to ensure uniform tensor sizes.
    2. Pads tokenized label sequences while replacing padding tokens with -100
       to correctly ignore them in loss computation.
    3. Removes the beginning-of-sequence (BOS) token if it was previously added.

    Args:
        processor (Any): A processor that includes a feature extractor for audio
                         inputs and a tokenizer for text labels.

    Methods:
        __call__(features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            Processes a batch of input features and labels, applies necessary padding,
            and returns them as PyTorch tensors.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            - "input_features" (torch.Tensor): Padded log-Mel spectrogram features.
            - "labels" (torch.Tensor): Padded tokenized labels with -100 for ignored positions.
    """

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # split inputs and labels since they have to be of different lengths and need different padding methods

        # first treat the audio inputs by simply returning torch tensors

        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences

        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]

        # pad the labels to max length

        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,

        # cut bos token here as it’s append later anyways

        if (
            (labels[:, 0] == self.processor.tokenizer.bos_token_id)
            .all()
            .cpu()
            .item()
        ):

            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    """
    Computes the Word Error Rate (WER) for model predictions.

    This function processes model predictions and reference labels by:
    1. Replacing -100 values in labels with the tokenizer's padding token ID.
    2. Decoding the predicted token sequences into text.
    3. Decoding the reference token sequences into text.
    4. Computing the Word Error Rate (WER) between predictions and references.

    Args:
        pred (transformers.EvalPrediction): A prediction object containing:
            - pred.predictions (np.ndarray): The model's predicted token IDs.
            - pred.label_ids (np.ndarray): The reference token IDs (ground truth labels).

    Returns:
        dict: A dictionary containing:
            - "wer" (float): The computed Word Error Rate (WER) as a percentage.
    """

    pred_ids = pred.predictions

    label_ids = pred.label_ids

    # replace -100 with the pad_token_id

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# Loading the audio and transcript data
data = load_audio_transcripts(audio_folder, transcript_folder)
df = pd.DataFrame(data)
df["text"] = df["text"].apply(clean_text)  # Cleaning the text data

new_audio_paths, new_transcripts = split_and_process_audio(
    df
)  # Splitting the audio files
# Creating a new DataFrame with segmented audio files
df_split = pd.DataFrame({"audio": new_audio_paths, "text": new_transcripts})

# Now splitting into train/test datasets
np.random.seed(42)
train_df, test_df = train_test_split(df_split, test_size=0.2)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Casting the audio column to Audio type (with sampling rate 16kHz)
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Loading the feature extractor, tokenizer, and processor models
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-base"
)  # Loading the feature extractor model
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="English", task="transcribe"
)  # Loading the tokenizer model
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="English", task="transcribe"
)  # Loading the processor model

# Preprocessing the training and testing datasets
train_dataset2 = train_dataset.map(prepare_dataset, num_proc=1)
test_dataset2 = test_dataset.map(prepare_dataset, num_proc=1)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor
)  # Creating the data collator

metric = evaluate.load("wer")  # Loading the WER metric

# Training the model
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-base"
)  # Loading the model

model.config.forced_decoder_ids = (
    None  # Disable forced decoding if causing issues
)
model.config.suppress_tokens = (
    []
)  # Disable token suppression if causing issues

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-eng-v2",  # output directory
    per_device_train_batch_size=4,  # batch size per device during training
    gradient_accumulation_steps=64,  # total number of steps before back propagation
    learning_rate=1e-5,  # learning rate
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    max_steps=100,  # total number of training steps
    gradient_checkpointing=True,  # enable gradient checkpointing to save memory
    fp16=True,  # enable mixed precision training
    evaluation_strategy="steps",  # evaluation strategy to use
    per_device_eval_batch_size=8,  # batch size for evaluation
    predict_with_generate=True,  # enable generation during evaluation
    generation_max_length=1024,  # maximum length of generated text
    save_steps=1000,  # save model every 1000 steps
    eval_steps=1000,  # evaluate model every 1000 steps
    logging_steps=25,  # log metrics every 25 steps
    report_to=["tensorboard"],  # enable tensorboard logging
    load_best_model_at_end=True,  # load best model at the end of training
    metric_for_best_model="wer",  # metric to use for best model
    greater_is_better=False,  # whether the best model is the one with the highest or lowest value of the metric
    push_to_hub=False,  # push the final model to the hub
    # num_train_epochs=3
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset2,
    eval_dataset=test_dataset2,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)  # Creating the trainer

trainer.train()  # Training the model

model.save_pretrained("./whisper-small-eng")  # Saving the model
processor.save_pretrained("./whisper-small-eng")  # Saving the processor
