"""
Multimodal Retrieval and Response Generation Utilities

This module provides utilities to retrieve and merge relevant content (text, tables, and images)
from Qdrant vector stores and generate responses using the Qwen2-VL model. It is designed to power
multimodal assistants that support natural language queries enriched with visual and tabular context.

Key Functionalities:
--------------------

1. **Text Retrieval**
   - `retrieve_text(text_store)`:  
     Converts a LangChain-based Qdrant text store into a retriever that uses similarity scoring with a defined threshold.

2. **Image Retrieval**
   - `image_retrieval(...)`:  
     Embeds a text query using CLIP, searches a Qdrant image collection, and returns image paths above a score threshold.
     Also displays matched images using IPython's image viewer.

3. **Table Retrieval**
   - `table_retrieve(...)`:  
     Converts a Qdrant collection storing table embeddings into a retriever, using the `"table_text"` payload for matching.

4. **Merged Retrieval**
   - `reranking(query, text_retriever, table_retriever)`:  
     Merges results from both text and table retrievers using LangChain's `MergerRetriever` and concatenates the content.

5. **Response Generation**
   - `generate_response(...)`:  
     Builds a multimodal prompt including retrieved content and visual input (if any), then passes it to a Qwen2-VL model.
     The result is a coherent, context-aware summary or answer.

Use Cases:
----------
- Multimodal question answering
- Retrieval-augmented generation (RAG)
- PDF understanding tools
- Visual + textual AI agents

Dependencies:
-------------
- LangChain (`Qdrant`, `MergerRetriever`)
- Qdrant client and models
- PIL (for image loading)
- Torch (for inference)
- Hugging Face CLIP & Qwen2-VL
- `qwen_vl_utils.process_vision_info` for aligning prompt tokens with image input

Example Workflow:
-----------------
1. Use `retrieve_text`, `image_retrieval`, and `table_retrieve` to get content.
2. Merge using `reranking`.
3. Pass results to `generate_response` with a user query and retrieved context.

"""


# Third-Party Library Imports
from PIL import Image
import torch
import IPython.display as display

# LangChain and Vector Store Libraries
from langchain_community.vectorstores import Qdrant
from langchain.retrievers import MergerRetriever
from qwen_vl_utils import process_vision_info
from langchain.prompts import PromptTemplate


def prompt():
    prompt_template = PromptTemplate(
        input_variables=["query", "output", "num_images","image_tokens"],
        template=(
            """You are a medical professional. Given the following medical query and its context, 
            craft a clear, well written and easy to understand Answer by extracting relevant information from the context.\n\n
            Do not add or invent any facts or information. If the answer cannot be found in the context, Answer Exactly: `No Relevant Information found`.\n\n
            Query:\n{query}\n\n
            Context:\n{output}\n\n
            {num_images} image(s) are also provided. Refer them, only if they are relevant\n\n
            "{image_tokens}\n\n"
            Answer:"""
            ),
    )
    return prompt_template

def pdf_summarization():
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
    audio_summarize_template = PromptTemplate(
    input_variables=["query", "output"],
    template=(
        """You are a professional medical summarizer. Summarize the content of an audio transcriptions\n\n
        Write **one** cohesive summary of the **entire document** \n\n
        removing any duplication, and organizing it logically.\n\n
        Query:\n{query}\n\n
        content:\n{output}\n\n
        Answer:"""
        ),
    )
    return audio_summarize_template

def retrieve_text(text_store):
    return text_store.as_retriever(search_type="mmr", search_kwargs={"k": 8, "lambda_mult":0.6},)

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
    return Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=text_embedding_model,
    content_payload_key="table_text",
    ).as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6,"lambda_mult":0.6},
    )


def reranking(query, text_retriever, table_retriever):
    merge_retriever = MergerRetriever(retrievers=[text_retriever, table_retriever])
    retrieved_docs = merge_retriever.invoke(query)
    return " ".join([doc.page_content for doc in retrieved_docs])


def generate_response(qwen_processor, qwen_model, DEVICE, prompt_template, query, output_text, image_path=None):
    """
    Generate a response using the Qwen2VL model with text and image input.

    Args:
        query (str): The user's query/question.
        output_text (str): The relevant context or data retrieved (text and table).
        image_path (str, optional): Path to the image for multimodal input.

    Returns:
        str: The generated response.
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
        text=prompt,
        images=image_inputs,
        return_tensors="pt"
    ).to(DEVICE)

    # Generate the output from the model
    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, top_k = 1, temperature = 0.5, max_new_tokens=512)
 
    # Decode the generated tokens to text
    response = qwen_processor.decode(
        generated_ids[0], skip_special_tokens=True
    )
    summary_start = response.find("Answer:") + len("Answer:")
    summary = response[summary_start:].strip()

    return summary