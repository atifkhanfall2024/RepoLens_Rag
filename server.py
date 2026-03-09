from fastapi import FastAPI, UploadFile, File
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import os
from index import IndexingPhase  # your indexing file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def getData():
    return "Server is Running"

@app.post("/Rag/data")
async def receive_files(files: List[UploadFile] = File(...)):
    if not files:
        return {"success": False, "message": "No files received"}

    # Save files to temp folder
    temp_folder = os.path.join("temprepo")
    os.makedirs(temp_folder, exist_ok=True)

    for file in files:
        contents = await file.read()
        save_path = os.path.join(temp_folder, file.filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(contents)
        print(f"Received: {file.filename}, size: {len(contents)} bytes")

    # Call IndexingPhase with the folder containing all uploaded files
    try:
        IndexingPhase(temp_folder)
        return {"success": True, "message": "Data successfully stored into Qdrant"}
    except Exception as e:
        return {"success": False, "error": str(e)}