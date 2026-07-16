import os

import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

from llm import RunInputData, extract_data

### -------- Constants
MODEL_NAME = "best_5km_model_2023_2024"
MODEL_FOLDER = "Models"

### -------------------------------------------- ###


@st.cache_data
def get_model():
    model_path = os.path.join(MODEL_FOLDER, MODEL_NAME)
    return load_model(model_path)


model = get_model()


### -------- Logika deterministyczna (Python, nie LLM)


def sprawdz_brakujace_dane(dane: RunInputData) -> list[str]:
    braki = []
    if not dane.plec:
        braki.append("płeć")
    if not dane.rocznik and not dane.wiek:
        braki.append("rok urodzenia lub wiek")
    if not dane.czas_5km:
        braki.append("czas na 5km (format MM:SS)")
    return braki


def komunikat_o_brakach(dane: RunInputData, braki: list[str]) -> dict:
    zwrot = f"{dane.imie}, " if dane.imie else ""

    if not braki:
        content = f"{zwrot}mam komplet danych! Liczę estymację czasu półmaratonu... 🏃"
    else:
        content = f"{zwrot}potrzebuję jeszcze: " + ", ".join(braki) + ". Podaj proszę te informacje."
        content = content[0].upper() + content[1:]

    return {"role": "assistant", "content": content}


def ujednolic_rocznik(dane: RunInputData) -> int | None:
    import datetime

    rok_biezacy = datetime.date.today().year

    if dane.rocznik is not None:
        if 1920 <= dane.rocznik <= rok_biezacy - 5:
            return dane.rocznik
        return None

    if dane.wiek is not None:
        if 5 <= dane.wiek <= 100:
            return rok_biezacy - dane.wiek
        return None

    return None


def wylicz_kategoria_wiekowa(rocznik: int) -> str:
    # UWAGA: dopasuj te przedziały do wartości z danych treningowych modelu!
    # sprawdź: df_raw["Kategoria wiekowa"].unique()
    import datetime

    wiek = datetime.date.today().year - rocznik

    if wiek < 20:
        return "U20"
    elif wiek < 30:
        return "20-29"
    elif wiek < 40:
        return "30-39"
    elif wiek < 50:
        return "40-49"
    else:
        return "50+"


def przygotuj_dane_do_modelu(dane: RunInputData) -> dict | None:
    rocznik = ujednolic_rocznik(dane)
    if not all([dane.plec, rocznik, dane.czas_5km]):
        return None

    return {
        "Płeć": dane.plec,
        "Kategoria wiekowa": wylicz_kategoria_wiekowa(rocznik),
        "5 km Czas": dane.czas_5km,
    }


### -------- UI

st.header("🏃‍♂️💨 HALFMARATON - Estymacja czasu ⏱️")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Jak się nazywasz? Podaj swój wiek lub rok urodzenia, płeć oraz wynik na 5km w formacie MM:SS"
)

if prompt:
    user_message = {"role": "user", "content": prompt}
    with st.chat_message("user"):
        st.markdown(user_message["content"])
    st.session_state["messages"].append(user_message)

    dane = extract_data(st.session_state["messages"])
    braki = sprawdz_brakujace_dane(dane)
    odpowiedz_bota = komunikat_o_brakach(dane, braki)

    with st.chat_message("assistant"):
        st.markdown(odpowiedz_bota["content"])
    st.session_state["messages"].append(odpowiedz_bota)

    if not braki:
        dane_do_modelu = przygotuj_dane_do_modelu(dane)
        wynik = predict_model(model, data=pd.DataFrame([dane_do_modelu]))
        czas_estymowany = wynik["prediction_label"].iloc[0]

        st.markdown(f"### 🎯 Estymowany czas półmaratonu: **{czas_estymowany}**")