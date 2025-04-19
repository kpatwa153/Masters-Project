"""
PDF Translation and Resizing Utilities

This module provides tools to preprocess and translate the text content of PDF documents
while preserving their visual structure, layout, and style. It is especially useful for
creating translated versions of documents without altering formatting or visuals.

Key Functionalities:
--------------------

1. **PDF Resizing**
   - `resize_pdf(input_pdf, output_pdf, scale_factor=1.2)`:
     Scales all pages of a PDF by a given factor. This is useful for enlarging the page
     space to make room for translated text that may take up more space than the original.

2. **PDF Translation**
   - `translate_pdf(input_pdf, output_pdf, language, translator)`:
     Translates the textual content of a resized PDF using a translator (e.g., Deep Translator).
     It preserves the layout by:
       - overlaying white rectangles on the original text
       - inserting translated text as styled HTML blocks
       - using Optional Content Groups (OCGs) to separate visual layers

   This function supports translation into any language supported by the given translator
   and ensures clean rendering in the output PDF.

Key Dependencies:
-----------------
- `pymupdf` (for PDF manipulation)
- `deep_translator.GoogleTranslator` (for translation support)
- `os` (for file operations)

Text Extraction Settings:
-------------------------
- Text is dehyphenated and whitespace preserved for improved block-based processing.

Example Use Case:
-----------------
This module is ideal for:
- Translating brochures, reports, or forms to other languages.
- Preserving visual content while replacing the textual content.
- Creating multilingual versions of static PDF documents.

"""

import os

import pymupdf
from deep_translator import GoogleTranslator

# Define color "white"
WHITE = pymupdf.pdfcolor["white"]
# This flag ensures that text will be dehyphenated after extraction.
textflags = pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_PRESERVE_WHITESPACE


def resize_pdf(input_pdf, output_pdf: str, scale_factor: float = 1.2):
    """
    Resizes a given PDF by scaling its page dimensions by a specified factor.

    Parameters:
    -----------
    input_pdf : Document
        the PDF file to be translated.
        This should be a `pymupdf` Document object.
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


def translate_pdf(input_pdf: str, output_pdf: str, language: str, translator):
    """
    Translates the text content of a PDF from English to a specified language while preserving layout and structure.

    Parameters
    ----------
    input_pdf : str
        Path to the input PDF (expected to be preprocessed/resized).
    output_pdf : str
        Path where the translated PDF will be saved.
    language : str
        Target language for translation (used for naming the optional content group).
    translator : Callable
        A translation function or model with a `.translate(text)` method.

    Process
    -------
    - Opens the input PDF and creates a new blank document.
    - For each page:
        - Duplicates the layout and visuals of the original page.
        - Extracts text blocks from the original.
        - Translates each block from English to the target language.
        - Draws a white rectangle over the original text area.
        - Inserts the translated text using HTML formatting (to preserve style).
    - Adds translated text as an optional content group (OCG).
    - Deletes the original input PDF after saving the translated one.

    Returns
    -------
    str
        Path to the newly saved translated PDF.
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

        # Define an Optional Content layer in the new document
        ocg_xref = translated_doc.add_ocg(f"{language.capitalize()}", on=True)

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


# language = str(input("Enter the language you want to translate to: "))
# languages = GoogleTranslator().get_supported_languages(as_dict=True)
# language_code = languages.get(language.lower())
# # Configure the translator (English to selected language)
# translator = GoogleTranslator(source="auto", target=language_code)


# path = "pdf.pdf"  # Path to the PDF file
# try:
#     doc = pymupdf.open("pdf.pdf")  # Load the PDF file
#     resized = resize_pdf(doc, "pdf_resized.pdf")  # Resize the PDF
#     print("PDF Resized")
#     translate_pdf(
#         resized, f"pdf-{language}.pdf", language, translator
#     )  # Translate the PDF
#     print("pdf translated")
# except Exception:
#     pass
