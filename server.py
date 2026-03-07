from fastapi import FastAPI
from pydantic import BaseModel
from index import IndexingPhase

app = FastAPI()


class FolderData(BaseModel):
    folders: str 

@app.get('/')
def getData():
    return "Server is Running"