import os

import pymupdf
from deep_translator import GoogleTranslator


def resize_pdf(input_pdf, output_pdf: str, scale_factor: float = 1.2):
    """
    Resizes a given PDF by scaling its page dimensions by a specified factor.

    Parameters:
    -----------
    input_pdf : str
        Path to the PDF file.
    output_pdf : str
        Path to save the resized PDF.
    scale_factor : float, optional
        Factor by which to scale the page size. Default is 1.2.

    Behavior:
    ---------
    - Opens the input PDF and creates a new blank PDF document.
    - Iterates through each page, resizes it by the given scale factor, and copies the content.
    - Saves the resized PDF to the specified output path.
    - Deletes the PDF (`input_pdf`) after resizing.

    Returns:
    --------
    str
        Path to the resized PDF.
    """
    old_doc = input_pdf
    new_doc = pymupdf.open()

    for page in old_doc:
        original_rect = page.rect
        new_width = original_rect.width * scale_factor
        new_height = original_rect.height * scale_factor

        # Create a new blank page with increased size
        new_page = new_doc.new_page(-1, width=new_width, height=new_height)
        # Copy content from the old page and scale it
        new_page.show_pdf_page(
            new_page.rect, old_doc, page.number, clip=original_rect
        )

    # Save the resized document
    new_doc.save(output_pdf)
    new_doc.close()
    old_doc.close()
    # os.remove(input_pdf)  # Remove the original PDF
    return output_pdf


def translate_pdf(input_pdf: str, output_pdf: str):
    """
    Translates the text content of a PDF from English to Desired Language while preserving the layout.

    Parameters:
    -----------
    input_pdf : str
        Path to the input PDF (expected to be resized).
    output_pdf : str
        Path to save the translated PDF.

    Behavior:
    ---------
    - Opens the input PDF and creates a new document.
    - Copies the original content from each page while maintaining formatting.
    - Extracts text blocks and translates them from English to French.
    - Covers the original text with a white background and inserts the translated text.
    - Saves the translated PDF to the specified output path.
    - Deletes the input PDF (`input_pdf`) after translation.

    Returns:
    --------
    str
        Path to the translated PDF.
    """
    resized_doc = pymupdf.open(input_pdf)
    translated_doc = pymupdf.open()

    for page in resized_doc:
        new_page = translated_doc.new_page(
            width=page.rect.width, height=page.rect.height
        )
        new_page.show_pdf_page(
            new_page.rect, resized_doc, page.number
        )  # Copy content from original page

        # Define an Optional Content layer in the new document named "French"
        ocg_xref = translated_doc.add_ocg("French", on=True)

        # Extract text grouped in blocks
        blocks = page.get_text("blocks", flags=textflags)

        # Process each text block
        for block in blocks:
            bbox = block[:4]  # Bounding box
            english = block[4]  # Extracted English text
            english = english.replace("\n", " ").replace("\uf0b7", "•")

            # Translate text
            french = translator.translate(english)

            # Create styled translated text
            french_html = f"""
            <div style='font-size:16px; text-align:justify; line-height:1.5;'>
                {french}
            </div>
            """

            # Cover original text and insert translated text
            new_page.draw_rect(bbox, color=None, fill=WHITE, oc=ocg_xref)
            new_page.insert_htmlbox(bbox, french_html, oc=ocg_xref)

    # Save the translated document
    translated_doc.save(output_pdf)
    translated_doc.close()
    resized_doc.close()
    os.remove(input_pdf)  # Remove the resized PDF
    return output_pdf


# Define color "white"
WHITE = pymupdf.pdfcolor["white"]
# This flag ensures that text will be dehyphenated after extraction.
textflags = pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_PRESERVE_WHITESPACE
# Configure the translator (English to French)
translator = GoogleTranslator(source="en", target="fr")


path = "pdf.pdf"  # Path to the PDF file
try:
    doc = pymupdf.open("pdf.pdf")  # Load the PDF file
    resized = resize_pdf(doc, "pdf_resized.pdf")  # Resize the PDF
    print("PDF Resized")
    translate_pdf(resized, "pdf-french.pdf")  # Translate the PDF
    print("pdf translated")
except Exception:
    pass
