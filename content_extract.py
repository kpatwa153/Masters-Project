import io
import json
import os

import camelot.io as camelot
import fitz  # PyMuPDF
import IPython.display as display
from PIL import Image


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    This function reads a PDF document and extracts text from each page,
    concatenating the text into a single string with newline separators.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the entire PDF document.
    """
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"  # Extract text from each page

    return text


def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """
    Extracts images from a PDF file and saves them as PNG files.

    This function scans each page of the PDF for embedded images, extracts them,
    resizes them to 512x512 pixels, and saves them as PNG files in the specified
    output folder.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str, optional): Directory where extracted images will be
                                       saved. Defaults to "extracted_images".

    Returns:
        list: A list of file paths for the extracted images.
    """
    os.makedirs(
        output_folder, exist_ok=True
    )  # Create output folder if it doesn't exist
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        for img_index, img in enumerate(doc[page_num].get_images(full=True)):
            xref = img[0]  # Get image reference
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Open the extracted image in PIL and resize
            img_pil = Image.open(io.BytesIO(image_bytes))
            img_pil = img_pil.resize((512, 512), Image.Resampling.LANCZOS)

            # Save the resized image
            image_filename = os.path.join(
                output_folder, f"page_{page_num+1}_image_{img_index+1}.png"
            )
            img_pil.save(image_filename, format="PNG")

            image_paths.append(image_filename)

    return image_paths


def extract_table_content(pdf_path):
    """
    Extracts tables from a PDF file using Camelot and formats them as JSON.

    This function reads a PDF document and extracts tables from each page using
    Camelot's lattice-based extraction. It then processes the tables into a list
    of dictionaries, where each dictionary represents a table with its extracted data.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        Dict: A dictionary containing extracted tables and their data.
    """
    tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
    all_tables_data = []  # List to store processed tables

    for i, table in enumerate(tables):
        headers = table.df.iloc[0].tolist()  # Extract headers
        data = table.df.iloc[1:].values.tolist()  # Extract table data
        table_data = [
            dict(zip(headers, row)) for row in data
        ]  # Convert to list of dicts
        all_tables_data.append(
            {"table_id": i + 1, "data": table_data}
        )  # Store with table ID

    return all_tables_data


def pdf_content_extraction(pdf_path):
    """
    Extracts text, images, and tables from a PDF file.

    This function combines the extraction of textual content, embedded images,
    and tabular data from a given PDF document. It internally uses specialized
    helper functions to process and collect each content type, and returns them
    as a structured dictionary.

    Args:
        pdf_path (str): The file path to the input PDF document.

    Returns:
        dict: A dictionary with the following keys:
            - "text" (str): Extracted textual content.
            - "images" (list): List of file paths to the extracted and resized images.
            - "tables" (list): List of extracted tables, each represented as a dictionary.
    """
    content = dict()
    content["text"] = extract_text_from_pdf(pdf_path)
    content["images"] = extract_images_from_pdf(pdf_path)
    content["tables"] = extract_table_content(pdf_path)
    return content
