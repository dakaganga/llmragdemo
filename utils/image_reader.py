from easyocr import Reader
from langchain_core.documents import Document
from typing import List
import cv2

langs = ['en']
reader = Reader(langs, gpu=False)

def get_image_text(image_path) -> List[Document]:
    image = cv2.imread(image_path)
    result = reader.readtext(image)
    complete_text = ""
    for (bbox, text, prob) in result:
        complete_text += text +" "
    metadata = {
            "source": image_path
        }
    return [Document(page_content=complete_text, metadata=metadata)]


if __name__ == "__main__":
  print(get_image_text("documents/doctor_note.jpg"))