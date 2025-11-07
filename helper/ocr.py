from langchain_core.documents import Document
from typing import List

def process_with_ocr(file_path: str) -> List[Document]:
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_path
    import os
    
    documents = []
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            pages = convert_from_path(file_path)
            
            for i, page in enumerate(pages):
                text = pytesseract.image_to_string(page)
                
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": i + 1,
                        "type": "scanned_pdf"
                    }
                )
                documents.append(doc)
                
        elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            image = Image.open(file_path)
            
            text = pytesseract.image_to_string(image)
            
            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "image"
                }
            )
            documents.append(doc)
            
    except Exception as e:
        print(f"OCR processing error: {str(e)}")
        doc = Document(
            page_content="",
            metadata={
                "source": file_path,
                "error": str(e),
                "type": "ocr_failed"
            }
        )
        documents.append(doc)
    
    return documents