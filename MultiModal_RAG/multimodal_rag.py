import fitz
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain.chat_models import init_chat_model
from langchain.schema.messages import HumanMessage
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

PDF_PATH = 'multimodal_sample.pdf'

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

LLM = init_chat_model("openai:gpt-4.1")

# load clip model
CLIP_MODEL = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True)
CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_data):
    if isinstance(image_data, str):
        image = Image.open(image_data).convert('RGB')
    else:
        image = image_data

    inputs = CLIP_PROCESSOR(images=image, return_tensors='pt')

    with torch.no_grad():
        features = CLIP_MODEL.get_image_features(**inputs)  # extract the feature vector of the image

        # normalize
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    
def embed_text(text):
    inputs = CLIP_PROCESSOR(
        text=text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=77   # max token length
    )

    with torch.no_grad():
        features = CLIP_MODEL.get_text_features(**inputs)

        #normalize
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    
def process_pdf(pdf_path, splitter):
    doc = fitz.open(pdf_path)

    all_docs = []
    all_embeddings = []
    image_data_store = {}

    for i, page in enumerate(doc):
        # process texts
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={'page':i, "type":"text"})
            text_chunks = splitter.split_documents([temp_doc])

            # embed chunk using clip
            for chunk in text_chunks:
                embeddings = embed_text(chunk.page_content)
                all_embeddings.append(embeddings)
                all_docs.append(chunk)

        # process images
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image['image']

                #convert to PIL image
                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                # create unique identifier
                image_id = f"page_{i}_img_{img_index}"

                # convert to base_64 for gpt compatibility
                buffered = io.BytesIO()
                pil_image.save(buffered, format='PNG')
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64

                # embed image
                embedding = embed_image(pil_image)
                all_embeddings.append(embedding)

                # not actual pixels, only the reference
                image_doc = Document(
                    page_content = f"[Image: {image_id}]",
                    metadata = {"page": i, "type":"image", "image_id":image_id}
                )
                all_docs.append(image_doc)

            except Exception as e:
                print("Error processing image {img_index} on page {i}: {e}")
                continue

    doc.close()

    return all_docs, all_embeddings, image_data_store

def create_vector_store(all_docs, all_embeddings):
    embeddings_array = np.array(all_embeddings)

    # custom FAISS index since we have precomputed embeddings
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],  #list of pairs of text and embedding
        embedding = None,   # since we are using precomputed embeddings
        metadatas=[doc.metadata for doc in all_docs])

    return vector_store

def retrieve_multimodal_docs(query, vector_store, k=5):
    query_embedding = embed_text(query)

    retrieved_docs = vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)

    return retrieved_docs

def create_multimodal_message(query, image_data_store, retrieved_docs):
    content = []

    content.append(
        {
            "type":"text",
            "text":f"Question: {query}\n\nContext:\n"
        }
    )

    #separate text and image from retrieved docs
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type")=='text']
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type")=='image']

    # add retrieved text docs' content
    if text_docs:   # if the retrieved docs has atleast one text doc, retrieve the content
        text_content = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_content}\n"
        })

    # add retrieved image docs
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url":f"data:image/png;base64,{image_data_store[image_id]}" #data URI format. way to embed data inside url instead of actual image
                }
            })

    # add instruction
    content.append({
        "type": "text",
        "text": f"Please answer the question based on the provided text and image"
    })

    return HumanMessage(content=content)

def get_llm_response(message, llm):
    response = llm.invoke([message])
    return response.content

def rag(query, vector_store, image_data_store):
    retrieved_docs = retrieve_multimodal_docs(query, vector_store, k=5)
    message = create_multimodal_message(query, image_data_store, retrieved_docs)
    response_content = get_llm_response(message, LLM)

    # printing infos
    print(f"\nRetrieved {len(retrieved_docs)} docs:")
    for doc in retrieved_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")    # page number

        if doc_type == 'text':  # text doc
            preview = doc.page_content[:100] + '...' if len(doc.page_content)<100 else doc.page_content
            print(f'    -   Text from page {page}: {preview}')
        elif doc_type == 'image':   # image doc
            print(f'    -   Image from page {page}')
        print('\n')

    return response_content

if __name__=='__main__':

    all_docs, all_embeddings, image_data_store = process_pdf(PDF_PATH, SPLITTER)

    vector_store = create_vector_store(all_docs, all_embeddings)
    
    queries = [
        'What does the chart on page 1 show about revenue trends?',
        'Summarize the main findings from the document',
        'What visual elements are present in the document?'
    ]

    for query in queries:
        print('-'*50)
        print(f'\nQuery: {query}')
        print('-'*50)
        response = rag(query, vector_store, image_data_store)
        print(f'Answer: {response}')
        print('='*70)
