from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


app = FastAPI()

#http://127.0.0.1:8000/

class Libro(BaseModel):
    titulo:str
    autor:str
    paginas:int
    editorial:Optional[str]

@app.get("/")
def index():
    textoASCII = "尺闩ﾁ七〤ㄖ ﾁ闩丂セ 闩尸讠"
    return {"message" : textoASCII}

@app.get("/libros/{id}")
def mostrar_libro(id:int):
    return {"data":id}

@app.post("/libros")
def insertar_libro(libro:Libro):
    return {"message": f"Libro {libro.titulo} insertado"}