import torch

# LangChain and Vector Store Libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image


def split_text(text, chunk_size=512, chunk_overlap=50):
    """
    Splits the input text into smaller overlapping chunks for efficient processing.

    This function uses a recursive character-based text splitter to divide the input
    text into chunks of a specified size, ensuring some overlap between consecutive
    chunks to maintain context continuity.

    Args:
        text (str): The input text to be split.
        chunk_size (int, optional): The maximum size of each chunk (default: 512).
        chunk_overlap (int, optional): The number of overlapping characters
                                       between consecutive chunks (default: 50).

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


def image_generate_embeddings(image_paths, clip_processor, clip_model):
    """
    Generates embeddings for a list of images using a CLIP-based model.

    This function processes images through a CLIP model to generate numerical
    embeddings, which can be used for similarity search, classification, or
    other vision-language tasks.

    Args:
        image_paths (list of str): A list of file paths to the images.

    Returns:
        list of torch.Tensor: A list of image embeddings, where each embedding
                              is a tensor representation of the input image.
    """
    image_embeddings = []

    for img_path in image_paths:
        image = Image.open(img_path)  # Load image
        input_image = clip_processor(
            images=image, return_tensors="pt"
        )  # Preprocess
        px = list(image.getdata())
        size = image.size
        with torch.no_grad():
            image_features = (
                clip_model.get_image_features(**input_image)
                .detach()
                .numpy()
                .tolist()[0]
            )  # Generate embeddings
        image_embeddings.append(image_features)

    return image_embeddings, px, size


def format_table_for_embedding(table_data):
    """
    Converts table JSON data into a structured text format.

    Args:
        table_data (dict): A dictionary containing table data.

    Returns:
        str: A structured text representation of the table.
    """
    table_text = ""
    for row in table_data:
        row_text = ", ".join([f"{key}: {value}" for key, value in row.items()])
        table_text += row_text + "\n"
    return table_text.strip()  # Remove trailing newline


def generate_table_embeddings(content, text_embedding_model):
    """
    Generates embeddings for tables extracted from a PDF document.

    This function converts each table's data into a structured text format
    and generates embeddings using a text embedding model.

    Args:
        content (dict): A dictionary containing extracted tables, where each
                        table has a "table_id" and "data" field.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - "table_id" (int): The unique identifier of the table.
            - "embedding" (list): The embedding vector of the table content.
            - "text" (str): The formatted text representation of the table.
    """
    table_embeddings = []

    for table in content["tables"]:
        table_text = format_table_for_embedding(
            table["data"]
        )  # Convert to text
        embedding = text_embedding_model.embed_query(
            table_text
        )  # Generate embedding
        table_embeddings.append(
            {
                "table_id": table["table_id"],
                "embedding": embedding,
                "text": table_text,
            }
        )

    return table_embeddings
