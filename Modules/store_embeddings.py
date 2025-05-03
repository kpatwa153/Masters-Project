"""
Qdrant-Based Embedding Storage Utilities

This module provides functions for storing text, image, and table embeddings into
Qdrant collections. It supports both memory-based and persistent Qdrant usage,
and includes metadata storage for downstream search, retrieval, or visualization tasks.

Key Functionalities:
--------------------

1. **Collection Management**
   - `create_collection(client, collection_name, vector_size)`:
     Creates or resets a Qdrant collection with the specified vector size using cosine distance.

2. **Text Embedding Storage**
   - `store_text(text, embedding_model, collection_name)`:
     Stores embedded text documents into a Qdrant collection using LangChain's Qdrant wrapper.

3. **Image Embedding Storage**
   - `store_image_embeddings(...)`:
     Stores image embeddings in Qdrant along with metadata such as image size, file path, and raw pixel data.

4. **Table Embedding Storage**
   - `store_table_embeddings(...)`:
     Saves tabular data embeddings in Qdrant, storing both the embedding vectors and original formatted text as metadata.

Use Cases:
----------
- Multimodal semantic search (text, image, table)
- Retrieval-augmented generation (RAG)
- Embedding-based data exploration and visualization
- Metadata-rich vector storage for custom AI pipelines

Dependencies:
-------------
- Qdrant (via `qdrant-client` and `langchain.vectorstores.Qdrant`)
- LangChain (for text embedding integration)
- Python standard libraries (e.g., `list`, `tuple`, `print`)

Example Output:
---------------
- Console log of created collections
- Stored vectors with IDs and rich metadata
- Callable Qdrant vector store (for text)

"""


# LangChain and Vector Store Libraries
from langchain_community.vectorstores import Qdrant

# Qdrant Client and Models
from qdrant_client.models import Distance, PointStruct, VectorParams


def create_collection(client, collection_name, vector_size):
    """
    Creates a Qdrant collection for storing embeddings.

    This function checks if a collection with the specified name already exists
    in Qdrant. If it does, the existing collection is deleted before creating
    a new one with the specified vector size and cosine similarity as the
    distance metric.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to create.
        vector_size (int): The dimensionality of the embedding vectors.

    Returns:
        None
    """
    # Check if the collection exists and delete it if necessary
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    # Create the new collection with cosine similarity for image embeddings
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size, distance=Distance.COSINE
        ),
    )
    print(f"Collection '{collection_name}' created successfully!")


def store_text(text, embedding_model, collection_name):
    """
    Stores text embeddings into a Qdrant collection.

    Args:
        texts (list): A list of text documents to embed and store.
        embedding_model: A model or callable that returns embeddings for the texts.
        collection_name (str): Name of the Qdrant collection to store embeddings.

    Returns:
        Qdrant: A Qdrant vector store instance with the stored embeddings.
    """
    return Qdrant.from_texts(
        texts=text,
        embedding=embedding_model,
        collection_name=collection_name,
        location=":memory:",
    )


def store_image_embeddings(
    client,
    image_paths,
    image_embeddings,
    px,
    size,
    image_collection,
    image_vector_size,
):
    """
    Stores image embeddings and metadata in a Qdrant collection.

    This function creates a Qdrant collection (or overwrites it if it exists),
    then stores the provided image embeddings along with associated metadata
    such as filenames, image sizes, and pixel data.

    Args:
        client (QdrantClient): The Qdrant client instance.
        image_paths (list of str): List of file paths to the images.
        image_embeddings (list): List of embedding vectors for the images.
        px (list): Pixel data for each image (optional for search, useful for metadata).
        size (tuple): Size (width, height) of the images.
        image_collection (str): Name of the Qdrant collection to create.
        image_vector_size (int): Dimensionality of the image embedding vectors.

    Returns:
        None
    """
    create_collection(client, image_collection, image_vector_size)
    points = [
        PointStruct(
            id=idx,  # Unique ID for each image
            vector=image_embeddings[idx],  # Embedding vector
            payload={
                "filename": image_paths[idx],  # Store filename for reference
                "size": size,  # Store image size
                "pixels": px,  # Store pixel data
            },
        )
        for idx in range(len(image_paths))
    ]

    client.upsert(collection_name=image_collection, points=points)
    print(f"Stored {len(points)} images in Qdrant.")


def store_table_embeddings(
    client, TABLE_COLLECTION, TABLE_VECTOR_SIZE, table_embeddings
):
    """
    Stores table embeddings and metadata in a Qdrant collection.

    This function creates a Qdrant collection for table data (or replaces it if it already exists),
    then uploads the table embeddings along with their IDs and associated text content.

    Args:
        client (QdrantClient): The Qdrant client instance.
        TABLE_COLLECTION (str): Name of the Qdrant collection to store table embeddings.
        TABLE_VECTOR_SIZE (int): Dimensionality of the table embedding vectors.
        table_embeddings (list of dict): List of dictionaries containing:
            - "table_id" (int): Unique identifier for each table.
            - "embedding" (list or np.array): Embedding vector of the table.
            - "text" (str): Text representation of the table for metadata/reference.

    Returns:
        None
    """
    create_collection(client, TABLE_COLLECTION, TABLE_VECTOR_SIZE)
    client.upload_collection(
        collection_name=TABLE_COLLECTION,
        vectors=[table["embedding"] for table in table_embeddings],
        ids=[table["table_id"] for table in table_embeddings],
        payload=[{"table_text": table["text"]} for table in table_embeddings],
    )
    print("✅ Table embeddings stored in Qdrant successfully!")
