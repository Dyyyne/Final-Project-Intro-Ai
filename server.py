
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# allow your webpage to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiaryEntry(BaseModel):
    text: str

@app.post("/predict")
def predict_emotion(entry: DiaryEntry):
    text = entry.text.lower()

    happy_words = ["happy", "joy", "glad", "excited", "grateful", "content", "great","fun","good"]
    sad_words = ["sad", "down", "unhappy", "depressed", "crying", "hurt", "awful", "lonely"]
    angry_words = ["angry", "mad", "furious", "annoyed", "irritated", "upset"]
    anxious_words = ["anxious", "nervous", "worried", "scared", "afraid", "stressed"]

    if any(w in text for w in happy_words):
        return {"emotion": "happy 😊"}
    elif any(w in text for w in sad_words):
        return {"emotion": "sad 😢"}
    elif any(w in text for w in angry_words):
        return {"emotion": "angry 😠"}
    elif any(w in text for w in anxious_words):
        return {"emotion": "anxious 😬"}
    else:
        return {"emotion": "neutral 😐"}
