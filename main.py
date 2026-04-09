from fastapi import FastAPI
from pydantic import BaseModel
import spacy

nlp = {
    "en": spacy.load("en_core_web_sm"),
    "fr": spacy.load("fr_core_news_sm"),
}

app = FastAPI()

class Request(BaseModel):
    text: str
    lang: str = "en"  # "en" or "fr"

@app.post("/tokenize")
def tokenize(req: Request):
    doc = nlp[req.lang](req.text)
    return {
        "tokens": [
            {
                "text": t.text,
                "lemma": t.lemma_,
                "pos": t.pos_,
                "tag": t.tag_,
                "dep": t.dep_,
                "is_stop": t.is_stop,
                "is_punct": t.is_punct,
                "shape": t.shape_,
                "ent_type": t.ent_type_ or None,
            }
            for t in doc
        ],
        "entities": [{"text": e.text, "label": e.label_} for e in doc.ents],
    }
