"""
HALFMARATON - Estymacja czasu
--------------------------------
Pipeline:
1. chat_bot()     - prowadzi rozmowę z użytkownikiem, dopytuje o brakujące dane
2. extract_data() - po cichu wyciąga dane strukturalne z całej konwersacji (Pydantic)
3. ujednolic_rocznik() / wylicz_kategoria_wiekowa() - logika deterministyczna w Pythonie
4. przygotuj_dane_do_modelu() - buduje finalny słownik cech dla modelu .pkl
"""


import json
import streamlit as st
import pandas as pd

import os
from pycaret.regression import load_model, predict_model
from pydantic import BaseModel, Field
# from typing import Optional
# from dotenv import load_dotenv
# import openAI
# import datetime
from llm import chat_bot, extract_data





### -------- Constants
MODEL_NAME = "best_5km_model_2023_2024" #.pkl"
MODEL_FOLDER = "Models"
### -------- OPEN AI - Load Model name
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Brak OPENAI API-KEY - sprawdź zmienne środowiskowe")

client = OpenAI(api_key=openai_api_key)

### -------------------------------------------- ###

@st.cache_data
def get_model():
    model_path = os.path.join(MODEL_FOLDER, MODEL_NAME) # f"{MODEL_NAME}.pkl")
    return load_model(model_path)


model = get_model()


st.header("🏃‍♂️💨 HALFMARATON - Estymacja czasu ⏱️")

### -------- Data model
class RunInputData(BaseModel):
    imie: Optional[str] = Field(default=None, description="Imię użytkownika")
    plec: Optional[str] = Field(default=None, description="Płeć: K lub M")
    rocznik: Optional[int] = Field(default=None, description="Rok urodzenia, np. 1995")
    wiek: Optional[int] = Field(default=None, description="Wiek w latach, jeśli user podał wiek zamiast roku urodzenia")
    czas_5km: Optional[str] = Field(default=None, description="Czas na 5km w formacie MM:SS, np. '24:30'")

historia_wiadomosci = [
    {"role": "user", "content": "Cześć Ilona, jestem kobietą, urodziłam się w 1993 r."},
    {"role": "assistant", "content": "Super! A jaki masz czas na 5km?"},
    {"role": "user", "content": "24:30"}
]


### -------- BOT 1:  konwersacyjny
def chat_bot(historia_wiadomosci: list[dict]) -> str:
    dzisiejsza_data = datetime.date.today().strftime("%Y-%m-%d")
 
    response = client.chat.completions.create(
        model=MODEL_NAME_LLM,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Dzisiejsza data to {dzisiejsza_data}. "
                    "Jesteś asystentem, który zbiera od użytkownika dane potrzebne "
                    "do estymacji czasu półmaratonu: płeć, rok urodzenia (lub wiek), "
                    "czas na 5km. Dopytuj uprzejmie o brakujące informacje, jedna rzecz na raz. "
                    "Gdy masz komplet danych, podsumuj je i zapytaj, czy są poprawne."
                ),
            },
            *historia_wiadomosci,
        ],
    )
    return response.choices[0].message.content

### -------- BOT 2: Data Extraction
def extract_data(historia_wiadomosci) -> RunInputData:
    # sklejasz całą konwersację w jeden tekst albo przekazujesz messages bezpośrednio
    response = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Wyciągnij dane biegacza z poniższej konwersacji. "
                    "Jeśli użytkownik podał rok urodzenia — wpisz go w pole 'rocznik'. "
                    "Jeśli podał swój wiek (liczbę lat) — wpisz go w pole 'wiek', "
                    "NIE przeliczaj samodzielnie na rocznik. "
                    "Jeśli czegoś brak w rozmowie, zostaw jako null."
                )
            },
            *historia_wiadomosci
        ],
        response_format=RunInputData,
    )
    return response.choices[0].message.parsed


print(extract_data(historia_wiadomosci))