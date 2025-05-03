"""
This module handles the preprocessing, training, and evaluation of a speech-to-text model using the Whisper architecture from Hugging Face.

The module includes the following steps:

1. **Loading and processing data**: Loads audio recordings and corresponding transcripts, cleans the text, and splits the audio files into smaller chunks to handle long recordings.

2. **Feature extraction and tokenization**: Extracts audio features (log-Mel spectrogram) and tokenizes the corresponding text for model training.

3. **Model preparation**: Loads the Whisper model and tokenizer, and sets up training configurations for fine-tuning.

4. **Training**: Uses a custom `Seq2SeqTrainer` with `WhisperForConditionalGeneration` for training the model on the preprocessed audio data.

5. **Evaluation**: Computes the Word Error Rate (WER) as the evaluation metric, which is used to assess the performance of the trained model.

Key functions:

- `load_audio_transcripts()`: Loads the audio and transcript files from specified directories.

- `clean_text()`: Preprocesses and cleans the text data (e.g., removing special characters and extra spaces).

- `split_and_process_audio()`: Splits long audio files into smaller chunks (with a maximum duration per chunk) while retaining the original transcripts.

- `prepare_dataset()`: Processes audio samples to extract features and tokenizes the corresponding text labels for training.

- `DataCollatorSpeechSeq2SeqWithPadding`: Custom data collator that applies padding to both input audio features and text labels, and ensures proper handling of padding tokens in the loss computation.

- `compute_metrics()`: Computes the Word Error Rate (WER) for model evaluation by comparing predicted and reference transcripts.

Dependencies:

- `transformers`: For loading pre-trained models and tokenizers.

- `datasets`: For dataset management and preprocessing.

- `evaluate`: For evaluating the model's performance using WER.

- `pydub`: For audio processing (splitting audio files into chunks).

- `torch`: For handling tensor operations and training with PyTorch.

- `sklearn`: For splitting the dataset into training and test sets.

Configuration:

- The Whisper model is fine-tuned with a max duration of 30 seconds per audio chunk.

- The dataset is split into training and testing sets using an 80/20 split.

- The model is trained with a learning rate of 1e-5, a batch size of 4, and 64 gradient accumulation steps.

- WER is used as the evaluation metric, and model checkpoints are saved based on WER improvement.

The model and processor are saved after training for later inference and use.

Usage:

1. Place your audio recordings in the specified `audio_folder` and transcripts in the `transcript_folder`.

2. Run the module to preprocess the data, split audio, tokenize text, and train the Whisper model.

3. The trained model will be saved in the `whisper-small-eng` directory for later use.
"""

# pip install jiwer to use `wer` evaluation metrics
# Importing the required libraries
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import pandas as pd
import torch
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
dataset_path = "../../audio_recordings"
audio_path = os.path.join(dataset_path, "Audio_Recordings")
transcript_path = os.path.join(dataset_path, "transcripts")
audio_folder = "../../audio_recordings/Audio_Recordings"
transcript_folder = "../../audio_recordings/Clean_Transcripts"


# Combining the audio and transcript files
def load_audio_transcripts(audio_folder, transcript_folder):
    """
    Load audio file paths and their corresponding transcript text.

    :param audio_folder: Path to the folder containing audio files.
    :type audio_folder: str
    :param transcript_folder: Path to the folder containing transcript files.
    :type transcript_folder: str

    :return: A list of dictionaries, each containing:
            - **"audio"** (*str*): Full path to the audio file.
            - **"text"** (*str*): Corresponding transcript text.
    :rtype: list[dict[str, str]]
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
    Clean a given text string by applying standard preprocessing steps.

    The function performs the following operations:

    1. Replaces newline characters with a space.
    2. Removes special characters, keeping only alphanumeric characters and spaces.
    3. Reduces multiple spaces to a single space and trims leading/trailing spaces.

    :param text: The input text to be cleaned.
    :type text: str

    :return: The cleaned text.
    :rtype: str
    """

    # text = text.lower()  # Convert to lowercase
    text = re.sub(r"\n", " ", text)  # Replace newlines with space
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def split_and_process_audio(df, max_duration=MAX_DURATION):
    """
    Split each audio file in the dataset into smaller chunks of a specified duration,
    while maintaining the correspondence with transcripts.

    :param df: A DataFrame containing audio file paths and transcripts. Expected columns:
            "audio" (file path), and "text" (transcript).
    :type df: pandas.DataFrame

    :param max_duration: Maximum duration per audio chunk in seconds. Defaults to 30.
    :type max_duration: int, optional

    :return: A tuple containing:
            - new_audio_paths (list of str): File paths of the split audio chunks.
            - new_transcripts (list of str): Corresponding transcripts for each chunk.
    :rtype: tuple[list[str], list[str]]
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

    :param examples: A dictionary containing:
                    - "audio" (dict): An audio sample with an "array" key.
                    - "text" (str): Corresponding transcript text.
    :type examples: dict

    :return: The processed example with:
            - "input_features" (list): Extracted log-Mel spectrogram features.
            - "labels" (list): Tokenized label IDs.
    :rtype: dict
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

    :param processor: A processor that includes a feature extractor for audio
                    inputs and a tokenizer for text labels.
    :type processor: Any

    :returns: A dictionary containing:
            - **"input_features"** (*torch.Tensor*): Padded log-Mel spectrogram features.
            - **"labels"** (*torch.Tensor*): Padded tokenized labels with -100 for ignored positions.
    :rtype: Dict[str, torch.Tensor]

    :methods:
        **__call__**(*features: List[Dict[str, Union[List[int], torch.Tensor]]]*) -> *Dict[str, torch.Tensor]*:
            Processes a batch of input features and labels, applies necessary padding,
            and returns them as PyTorch tensors.
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


if __name__ == "__main__":
    # Loading the audio and transcript data
    data = load_audio_transcripts(audio_folder, transcript_folder)
    df = pd.DataFrame(data)
    df["text"] = df["text"].apply(clean_text)  # Cleaning the text data
    print("text cleaned")

    # Splitting the audio files
    new_audio_paths, new_transcripts = split_and_process_audio(df)

    # Creating a new DataFrame with segmented audio files
    df_split = pd.DataFrame(
        {"audio": new_audio_paths, "text": new_transcripts}
    )
    print("audio files segmented")

    # Now splitting into train/test datasets
    np.random.seed(42)
    train_df, test_df = train_test_split(df_split, test_size=0.2)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Casting the audio column to Audio type (with sampling rate 16kHz)
    train_dataset = train_dataset.cast_column(
        "audio", Audio(sampling_rate=16000)
    )
    test_dataset = test_dataset.cast_column(
        "audio", Audio(sampling_rate=16000)
    )
    print("Train-Test Split Completed")

    # Loading the feature extractor, tokenizer, and processor models
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-base"
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="English", task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="English", task="transcribe"
    )

    # Preprocessing the training and testing datasets
    train_dataset2 = train_dataset.map(prepare_dataset, num_proc=1)
    test_dataset2 = test_dataset.map(prepare_dataset, num_proc=1)
    print("split-2")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")

    # Training the model
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-base"
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print("model loaded")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-eng",  # output directory
        per_device_train_batch_size=4,  # batch size per device during training
        gradient_accumulation_steps=64,  # total number of steps before back propagation
        learning_rate=1e-5,  # learning rate
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        max_steps=100,  # total number of training steps
        gradient_checkpointing=True,  # enable gradient checkpointing to save memory
        fp16=True,  # enable mixed precision training
        eval_strategy="steps",  # evaluation strategy to use
        save_strategy="steps",  # save strategy to use
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
    )

    print("Training the model......")
    trainer.train()
    print("Model Saved")
    model.save_pretrained("./whisper-small-eng")
    processor.save_pretrained("./whisper-small-eng")
