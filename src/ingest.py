#!/usr/bin/env python3
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    JSONLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document
from langchain.vectorstores.utils import DistanceStrategy

# from langchain.embeddings import HuggingFaceEmbeddings
from embeddings import HuggingFaceEmbeddings

# Load environment variables
db_type = "context"

source_directory = "/mnt/sdd/nguyen.van.quan/Researchs/Qlora/data/company/final/json/"
persist_directory = f"/mnt/sdd/nguyen.van.quan/Researchs/Qlora/data/company/final/db/faiss_{db_type}_db"


# encode_kwargs = {'normalize_embeddings': True}
# model_kwargs = {'device': 'cuda'}
# embeddings_model_name  = "VoVanPhuc/bge-base-vi"
# embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
embeddings_model_name  = "vinai/vinai-translate-vi2en"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, device="cuda")




# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".json":(JSONLoader, {"jq_schema": f'.[].{db_type}'}),
    ".csv": (TextLoader, {"encoding": "utf8"}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            sorted(glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True))
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    print(filtered_files)
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():
    # if does_vectorstore_exist(persist_directory):
    #     # Update and store locally vectorstore
    #     print(f"Appending to existing vectorstore at {persist_directory}")
    #     db = Chroma(documents, embeddings, persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"})
    #     collection = db.get()
    #     ignore_files = [metadata['source'] for metadata in collection['metadatas']]
    #     documents = load_documents(source_directory, ignore_files)
    #     print(f"Creating embeddings. May take some minutes...")
    #     db.add_documents(documents)
    # else:
        # Create and store locally vectorstore
    print("Creating new vectorstore")
    documents = load_documents(source_directory)
    print(f"Creating embeddings. May take some minutes...")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(persist_directory)
    db = None

    print(f"Ingestion complete! You can now query your documents")


if __name__ == "__main__":
    main()
