from langchain_community.document_loaders import DirectoryLoader  , TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
load_dotenv()
#print("Api key " , os.getenv('GOOGLE'))
def IndexingPhase(folders:str):
    if not os.path.exists(folders):
        raise FileNotFoundError(f"Folder {folders} does not exist")
    loader = DirectoryLoader(
        folders,
        glob="**/*",
        loader_cls=TextLoader
    )

    docx = loader.load()
    print("Total files:", len(docx))


    # now making chunks 

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )

    text = chunks.split_documents(documents=docx)

    # now make embeddings of these chunks 

    embeddings = GoogleGenerativeAIEmbeddings(
         model="gemini-embedding-001"
    )

    vectors = QdrantVectorStore(
        url="http://localhost:6333/",
        collection_name="RepoLens Indexing",
        documents=text,
        embedding=embeddings
    )

print("Data Success store into Vector Database")






