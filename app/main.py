from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from app import model
from app.routes.predict import router as predict_router

def init_db():
    conn = sqlite3.connect("database/alerts.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            timestamp TEXT,
            label TEXT,
            severity INTEGER,
            snapshot_path TEXT
        )
    """)

    conn.commit()
    conn.close()

app = FastAPI()

init_db()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(predict_router)

@app.get("/")
def home():
    return {"message": "Fire Detection API is running 🚀"}