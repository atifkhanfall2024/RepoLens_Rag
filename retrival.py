# retrival.py
def RetirvalPhase(query: str):
    """
    Retrieves relevant documents from Qdrant and returns
    a professional, context-aware answer using Gemini API.
    """
    try:
        import os
        from dotenv import load_dotenv
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from openai import OpenAI

        # Load environment variables
        load_dotenv()
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY is missing in .env")

        # Initialize embeddings (must match IndexingPhase)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=key
        )

        # Connect to Qdrant collection
        vectors = QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333",
            collection_name="RepoLens Indexing",
            embedding=embeddings
        )

        # Perform similarity search (increase k for better coverage)
        result = vectors.similarity_search(query, k=5)

        # Debug logs: show retrieved documents
        #print(f"[DEBUG] Retrieved documents: {len(result)}")
        

        # Build context string from retrieved documents
        context = "\n\n".join([
            f"File: {doc.metadata.get('source', 'unknown')}\nContent:\n{doc.page_content}"
            for doc in result
        ])

        # Professional system prompt
        system_prompt = f"""
You are a professional AI assistant and RAG system for analyzing GitHub repositories.
Answer clearly, concisely, and professionally using ONLY the context below.
Provide structured answers: paragraphs, bullet points, or code blocks if applicable.
. also donot give negative response tune positive and always answer accordingi to avalible data not extra outside data

Context:
{context}

If the answer is not in the context, respond: "I could not find the answer in the repository."
This system was designed by Muhammad Atif Khan.
"""

        # Call Gemini OpenAI API
        client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )

        # Return cleaned answer
        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        # Raise RuntimeError with full details
        raise RuntimeError(f"RAG retrieval failed: {e}")