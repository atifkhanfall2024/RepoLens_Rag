def IndexingPhase(folder_path: str):
    import os
    from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from dotenv import load_dotenv

    # Load API key from .env
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key not found in environment variables")

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist")

    all_docs = []

    # Load supported files
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            try:
                if ext in [".txt", ".py", ".js", ".ts", ".java", ".md", ".json"]:
                    loader = TextLoader(filepath, encoding="utf-8")
                elif ext == ".pdf":
                    loader = PyPDFLoader(filepath)
                elif ext == ".docx":
                    loader = UnstructuredWordDocumentLoader(filepath)
                else:
                    continue

                loaded_docs = loader.load()
                all_docs.extend(loaded_docs)
                print(f"Loaded {len(loaded_docs)} docs from {filepath}")
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")

    if not all_docs:
        raise ValueError("No supported documents found in folder.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    text_chunks = splitter.split_documents(all_docs)
    print(f"Created {len(text_chunks)} chunks")

    # Initialize embeddings with API key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=GOOGLE_API_KEY
    )

    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "RepoLens Indexing"

    # Create collection if it doesn't exist
    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name not in existing_collections:
        # Use embeddings to get vector size safely
        sample_vector = embeddings.embed_query(text_chunks[0].page_content)
        vector_size = len(sample_vector)

        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE)
        )
        print(f"Created collection '{collection_name}' with vector size {vector_size}")

    # Store documents
    store = QdrantVectorStore(
        collection_name=collection_name,
        embedding=embeddings,
        client=client
    )
    store.add_documents(text_chunks)
    print("✅ Data successfully stored into Qdrant")