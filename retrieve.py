"""
Multimodal Medical RAG Pipeline for Text, Table, and Image-Based Retrieval and Response Generation

This module provides utility functions for building and operating a Retrieval-Augmented Generation (RAG) system
designed for medical applications. The system supports multimodal inputs, including text, tables, and images,
and integrates LangChain, Qdrant, CLIP, and Qwen2VL to retrieve and generate context-aware responses to user queries.

Key Components:

1. Prompt Templates:
    - `pdf_prompt`: Prompt template for answering queries using PDF content (text + images).
    - `audio_prompt`: Prompt template for answering queries using audio transcriptions.
    - `pdf_summarization`: Prompt for summarizing the content of a PDF document.
    - `audio_summarization`: Prompt for summarizing content from audio transcriptions.

2. Retrieval Utilities:
    - `retrieve_text`: Retrieves relevant unstructured text from a Qdrant vector store using a similarity threshold.
    - `image_retrieval`: Uses CLIP embeddings to search for and return relevant images based on a text query.
    - `table_retrieve`: Retrieves relevant tabular content from a Qdrant collection using embedded vectors.

3. Reranking:
    - `reranking`: Combines results from text and table retrievers using `MergerRetriever` to form a unified context.

4. Multimodal Response Generation:
    - `generate_response`: Formats the query and context into a prompt, processes optional images, and uses the Qwen2VL
      model to generate a coherent and medically accurate response.

Dependencies:
    - LangChain
    - Qdrant
    - PIL (Pillow)
    - Torch
    - CLIP (Hugging Face)
    - Qwen2VL (Qwen processor + model)
    - IPython.display (for visual display in notebooks)

Typical Use Case:
This module can be used in a medical assistant chatbot that processes PDFs and audio recordings to answer domain-specific
queries, leveraging retrieval-augmented generation for explainable and data-grounded responses.

"""

# Third-Party Library Imports
import IPython.display as display
import torch
from langchain.prompts import PromptTemplate
from langchain.retrievers import MergerRetriever

# LangChain and Vector Store Libraries
from langchain_community.vectorstores import Qdrant
from PIL import Image
from qwen_vl_utils import process_vision_info


def pdf_prompt():
    """
    Creates a prompt template for answering medical queries based on PDF content.

    This function constructs a structured prompt intended for use with a language model
    in a medical retrieval-augmented generation (RAG) system. The prompt guides the model
    to analyze a medical query using the provided textual and visual (image) context
    extracted from a PDF document.

    The generated prompt instructs the model to:
    - Analyze the query carefully.
    - Extract only medically relevant information from the context.
    - Use step-by-step logical reasoning based solely on provided data.
    - Avoid repetition and assumptions.
    - Format the response clearly in bullet points.
    - Refer to images only if they are clearly relevant.
    - Return 'No Relevant Information found' if the answer is not present.

    Returns:
        PromptTemplate: A LangChain PromptTemplate instance with placeholders for:
            - query (str): The user's medical question.
            - output (str): Extracted text and table content from the PDF.
            - num_images (int): Number of images available in the context.
            - image_tokens (str): Visual tokens representing embedded images.
    """
    prompt_template = PromptTemplate(
        input_variables=["query", "output", "num_images", "image_tokens"],
        template=(
            """You are a licensed medical professional. 
            Your task is to analyze the following medical query using the provided context and relevant images, if applicable.
            Follow these steps:
            1. Analyze the query: Understand what is being asked.
            2. Examine the context: Identify and extract only the medically relevant facts from the context.
            3. Reason step-by-step: Think logically and explain your reasoning based solely on the information provided.
            4. If the same information appears multiple times in different forms mention it only once in your answer.
            5. Write a final answer: Generate a clear, concise, well-strucured, and accurate medical response in bullet points.

            Important Instructions:
            - Do not add, assume, or invent any facts.
            - Only refer to images if they are clearly relevant to the query.
            - You cannot repeat the same information in the answer.
            - If the answer cannot be found in the context, respond with exactly: No Relevant Information found.
            Query:
            {query}

            Context:
            {output}

            {num_images} image(s) provided.
            Image tokens:
            {image_tokens}

            Step-by-Step Reasoning:

            Answer:"""
        ),
    )
    return prompt_template


def audio_prompt():
    """
    Creates a prompt template for answering medical queries based on audio transcription context.

    This function returns a structured LangChain PromptTemplate designed for a language model
    to generate a medically accurate response based on the transcribed content of an audio recording
    (e.g., doctor-patient conversation, medical notes).

    The prompt instructs the model to:
    - Analyze the medical query carefully.
    - Identify and extract only medically relevant information from the transcription.
    - Use logical, step-by-step reasoning based only on the provided content.
    - Avoid redundant information.
    - Produce a well-structured, concise medical response.
    - Return 'No Relevant Information found' if the answer is not present in the context.

    Returns:
        PromptTemplate: A LangChain PromptTemplate with placeholders for:
            - query (str): The user's medical question.
            - output (str): Transcribed text from an audio source.
    """
    prompt_template = PromptTemplate(
        input_variables=["query", "output"],
        template=(
            """You are a licensed medical professional. 
            Your task is to analyze the following medical query using the provided context.
            Follow these steps:
            1. Analyze the query: Understand what is being asked.
            2. Examine the context: Identify and extract only the medically relevant facts from the context.
            3. Reason step-by-step: Think logically and explain your reasoning based solely on the information provided.
            4. Remove Redundancy: If the same information appears multiple times in different forms mention it only once in your answer.
            5. Write a final answer: Summarize a clear, concise, well-strucured, and accurate medical response.

            Important Instructions:
            - Do not add, assume, or invent any facts.
            - If the answer cannot be found in the context, respond with exactly: No Relevant Information found.
            Query:
            {query}

            Context:
            {output}

            Step-by-Step Reasoning:

            Answer:"""
        ),
    )
    return prompt_template


def pdf_summarization():
    """
    Creates a prompt template for summarizing the content of a medical PDF document.

    This function returns a LangChain PromptTemplate designed to guide a language model
    in generating a faithful summary of a PDF's contents, including both text and image references.

    The prompt instructs the model to:
    - Read the provided query.
    - Summarize the document content accurately.
    - Refer to attached images only if they are relevant.
    - Avoid adding or fabricating any facts not present in the content.

    Returns:
        PromptTemplate: A LangChain PromptTemplate with placeholders for:
            - query (str): The user's request or query.
            - output (str): Extracted text and table content from the PDF.
            - num_images (int): Number of associated images.
            - image_tokens (str): Visual tokens representing the attached images.
    """
    pdf_summarize_template = PromptTemplate(
        input_variables=["query", "output", "num_images", "image_tokens"],
        template=(
            "Summarize the content of the PDF document"
            "Do not add or invent any information of fact of your own.\n\n"
            "Query:\n{query}\n\n"
            "content:\n{output}\n\n"
            "{num_images} image(s) attached:\n"
            "{image_tokens}\n\n"
            "Answer:"
        ),
    )
    return pdf_summarize_template


def audio_summarization():
    """
    Creates a prompt template for summarizing medical audio transcriptions.

    This function returns a LangChain PromptTemplate designed to guide a language model
    in generating a clear and cohesive summary of transcribed audio data based on a user-provided query.

    The prompt instructs the model to:
    - Read the query and the transcription content.
    - Write a single, well-structured summary for the entire content.
    - Eliminate redundancy and ensure logical organization.
    - Avoid adding or fabricating any information.

    Returns:
        PromptTemplate: A LangChain PromptTemplate with placeholders for:
            - query (str): The medical query related to the audio content.
            - output (str): The full audio transcription to be summarized.
    """
    audio_summarize_template = PromptTemplate(
        input_variables=["query", "output"],
        template=(
            """You are a professional medical summarizer. Summarize the content of an audio transcriptions based on the Query.\n\n
        Write **one** cohesive summary of the **entire document** \n\n
        removing any duplication, and organizing it logically.\n\n
        Query:\n{query}\n\n
        content:\n{output}\n\n
        Answer:"""
        ),
    )
    return audio_summarize_template


def retrieve_text(text_store):
    """
    Converts a text vector store into a retriever using similarity score threshold.

    This function wraps the given vector store with retrieval capabilities, enabling
    similarity-based search with a specified score threshold. Only documents with a
    similarity score above the threshold will be returned during retrieval.

    Args:
        text_store: The Qdrant text vector store containing embedded documents.

    Returns:
        BaseRetriever: A retriever object that supports similarity-based text retrieval
        with a minimum relevance score of 0.75.
    """
    return text_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.75},
    )


def image_retrieval(
    client,
    query,
    clip_processor,
    clip_model,
    collection_name,
    limit=1,
    with_payload=True,
    score_threshold=0.7,
):
    """
    Retrieves images based on a text query using CLIP model embeddings.

    This function takes a text query, generates its embedding using the CLIP model,
    and performs a similarity search in a specified Qdrant collection. It returns
    the top matching images based on the query embedding and displays them.

    Args:
        query (str): The text query used to search for relevant images.
        clip_processor (CLIPProcessor): The processor used to tokenize and preprocess input text for CLIP.
        clip_model (CLIPModel): The CLIP model used to generate text embeddings.
        collection_name (str): The name of the Qdrant collection containing the image embeddings.
        limit (int, optional): The maximum number of results to return (default is 3).
        with_payload (bool, optional): Whether to include metadata (e.g., filenames) with the results (default is True).
        score_threshold (float, optional): The minimum similarity score for returning results (default is 0.7).

    Returns:
        None: Displays the top matching images based on the query
    """
    input_text = clip_processor(text=[query], return_tensors="pt")
    with torch.no_grad():
        text_embedding = (
            clip_model.get_text_features(**input_text).squeeze().tolist()
        )
    results_with_scores = client.search(
        collection_name=collection_name,
        query_vector=text_embedding,
        limit=limit,
        with_payload=with_payload,  # Retrieve metadata (e.g., filenames)
        score_threshold=score_threshold,
    )
    results = [res.payload["filename"] for res in results_with_scores]
    images = list()
    if results:
        for image_path in results:
            images.append(image_path)
            img = Image.open(image_path)
            display.display(img)
    return images


def table_retrieve(client, collection_name, text_embedding_model):
    """
    Initializes a retriever for tabular data stored in a Qdrant collection using similarity score threshold.

    This function creates a retriever from a Qdrant vector store containing table embeddings. It enables
    similarity-based search restricted to the "table_text" payload and returns only documents with a
    similarity score above the specified threshold.

    Args:
        client: Qdrant client instance connected to the database.
        collection_name (str): Name of the Qdrant collection containing table embeddings.
        text_embedding_model: Embedding model used to generate table embeddings.

    Returns:
        BaseRetriever: A retriever configured to return table data with similarity score >= 0.75.
    """
    return Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=text_embedding_model,
        content_payload_key="table_text",
    ).as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.75},
    )


def reranking(query, text_retriever, table_retriever):
    """
    Merges and ranks documents retrieved from both text and table retrievers for a unified response.

    This function combines the results from a text retriever and a table retriever using a
    `MergerRetriever`. It then extracts and concatenates the page content from the retrieved
    documents into a single string, preserving only the relevant content for answer generation.

    Args:
        query (str): The input query to search for relevant information.
        text_retriever: Retriever instance for unstructured text data.
        table_retriever: Retriever instance for structured/tabular data.

    Returns:
        str: Concatenated content from all retrieved documents.
    """
    merge_retriever = MergerRetriever(
        retrievers=[text_retriever, table_retriever]
    )
    retrieved_docs = merge_retriever.invoke(query)
    return " ".join([doc.page_content for doc in retrieved_docs])


def generate_response(
    qwen_processor,
    qwen_model,
    DEVICE,
    prompt_template,
    query,
    output_text,
    image_path=None,
):
    """
    Generates a medical response using the Qwen2VL model with both text and optional image inputs.

    This function constructs a formatted prompt using the provided query and retrieved text/table data.
    If images are available, they are embedded into the prompt using special tokens and processed
    alongside the text for multimodal inference with the Qwen2VL model.

    Args:
        qwen_processor: Qwen2VL processor for preparing text and image inputs.
        qwen_model: Qwen2VL model used to generate the response.
        DEVICE: The device ('cpu' or 'cuda') on which the model should run.
        prompt_template (PromptTemplate): LangChain-style prompt template for formatting the input.
        query (str): The medical query from the user.
        output_text (str): Retrieved context (text or table content) relevant to the query.
        image_path (list[str], optional): List of file paths to images used in multimodal response generation.

    Returns:
        str: A clean and concise medical response generated by the model.
    """
    # Format the prompt using LangChain template
    images = []
    image_tokens = ""

    if image_path:
        for path in image_path:
            img = Image.open(path).convert("RGB")
            images.append(img)
        image_tokens = "\n".join(["<|image|>"] * len(images))
    else:
        images = None

    prompt = prompt_template.format(
        query=query,
        output=output_text,
        num_images=len(images) if images else 0,
        image_tokens=image_tokens,
    )
    conversation = [
        {"role": "user", "content": prompt},
    ]
    image_inputs, _ = process_vision_info(conversation)
    # Prepare the inputs for the model
    inputs = qwen_processor(
        text=prompt, images=image_inputs, return_tensors="pt"
    ).to(DEVICE)

    # Generate the output from the model
    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **inputs, top_k=1, temperature=0.5, max_new_tokens=512
        )

    # Decode the generated tokens to text
    response = qwen_processor.decode(
        generated_ids[0], skip_special_tokens=True
    )
    summary_start = response.find("Answer:") + len("Answer:")
    summary = response[summary_start:].strip()

    return summary
