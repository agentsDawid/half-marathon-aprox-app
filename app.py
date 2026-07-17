"""
HALFMARATON - Estymacja czasu
--------------------------------
Pipeline:
1. chat_bot()     - prowadzi rozmowę z użytkownikiem, dopytuje o brakujące dane
2. extract_data() - po cichu wyciąga dane strukturalne z całej konwersacji (Pydantic)
3. wylicz_kategoria_wiekowa() - logika deterministyczna w Pythonie
4. prepare_model_data() - buduje finalny słownik cech dla modelu .pkl
"""


import streamlit as st
import pandas as pd
import os
import datetime

from pycaret.regression import load_model, predict_model
from llm import extract_data, check_missing_data, missing_status, RunInputData
from features import fill_year_category, unify_age, convert_time_to_seconds, convert_seconds_to_time, merge_data
from do_client import load_model

### -------- Constants
MODEL_NAME = "best_5km_model_2023_2024" #.pkl"
MODEL_FOLDER = "Models"
MISSING_DATA = "Brak informacji"
MESSAGES_LIMIT = 8
LLM_MESSAGES_LIMIT = 1
### -------------------------------------------- ###

@st.cache_data
def get_local_model():
    model_path = os.path.join(MODEL_FOLDER, MODEL_NAME) # f"{MODEL_NAME}.pkl")
    return load_model(model_path)

model = load_model(MODEL_NAME) # get_local_model()

    # ['Imię', 'Płeć', 'Kategoria wiekowa', 'Rocznik', '5 km Czas',
#        '5 km Miejsce Open', '5 km Tempo', 'Czas', 'rok zawodów']
def prepare_model_data(dane: RunInputData) -> dict | None:
    bornYear = unify_age(dane)
    if not all([dane.plec, bornYear, dane.czas_5km]):
        return None
    dane.czas_5km = convert_time_to_seconds(dane.czas_5km)
    age = datetime.date.today().year - bornYear
    return {
        "Imię": dane.imie if dane.imie else "Nieznane",
        "Płeć": dane.plec,
        "Kategoria wiekowa": fill_year_category(age, dane.plec),
        "Rocznik": bornYear,
        "5 km Czas": convert_time_to_seconds(dane.czas_5km),
    }

def return_estimated_time(userData) -> str | None:
    if userData is None:
        return None
    userData_model = prepare_model_data(userData)
    st.session_state['userdata_model'] = userData_model 
    if userData_model is None:
        return None
    result = predict_model(model, data=pd.DataFrame([userData_model]))
    predictedTime = result["prediction_label"].iloc[0]

    return convert_seconds_to_time(predictedTime)

### -------------------------------------------- ###
st.header("🏃‍♂️💨 HALFMARATON - Estymacja czasu ⏱️")

if "messages" not in st.session_state:
    st.session_state['messages'] = []
if "userData" not in st.session_state:
    st.session_state["userData"] = None
if "userdata_model" not in st.session_state:
    st.session_state['userdata_model'] = {}

prompt = st.chat_input('Hey. Jak się nazywasz? Podaj swój wiek lub datę urodzenia oraz wynik na 5km')


col_chat, col_data = st.columns([2, 1])
with col_chat:    
    for message in st.session_state['messages'][-MESSAGES_LIMIT:]:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            
    if prompt:
        user_message = {"role": "user", "content": prompt}
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["messages"].append(user_message)

        messages_to_llm = st.session_state["messages"][-LLM_MESSAGES_LIMIT:]
        newData = extract_data(messages_to_llm)
        userData = merge_data(st.session_state["userData"], newData)
        st.session_state["userData"] = userData
        missing = check_missing_data(userData)
        chatbot_answer = missing_status(userData, missing)

        with st.chat_message("assistant"):
            st.markdown(chatbot_answer)
        st.session_state["messages"].append({"role": "assistant", "content": chatbot_answer})

        if not missing:
            predictedTime = return_estimated_time(userData)
            if predictedTime is None:
                st.warning(f"Estymowany czas półmaratonu nieudany, popraw dane")
            else:
                userName = userData.imie
                if userName is None:
                    userName = ""
                else:
                    userName = f'{userName} '
                st.success(f"Super, {userName}twój estymowany czas to: {predictedTime}")
      
with col_data:
    st.subheader("📋 Zebrane dane")
    userData = st.session_state["userData"]
 
    if userData is None:
        st.info(MISSING_DATA)
    else:
        missing = check_missing_data(userData)
 
        st.markdown(f"**Imię:** {userData.imie if userData.imie else '—'}")
        st.markdown(f"**Płeć:** {userData.plec if userData.plec else MISSING_DATA}")
        st.markdown(
            f"**Rok urodzenia / wiek:** "
            f"{userData.rocznik if userData.rocznik else (userData.wiek if userData.wiek else MISSING_DATA)}"
            f"{' r.' if userData.rocznik else (' lat' if userData.wiek else '')}"
        )
        age_cathegory = st.session_state['userdata_model'].get('Kategoria wiekowa', MISSING_DATA)
        st.markdown(f"**Kategoria wiekowa:** {age_cathegory}")
        st.markdown(f"**Czas na 5km:** {convert_seconds_to_time(userData.czas_5km) if userData.czas_5km else MISSING_DATA}")
 
        st.divider()
        st.subheader("🎯🏃‍♂️ Estymacja")
 
        if missing:
            st.warning(MISSING_DATA)
        else:
            predictedTime = return_estimated_time(userData)
            if predictedTime is None:
                st.warning(f"Estymowany czas półmaratonu nieudany")
            else:
                st.success(f"Estymowany czas półmaratonu: **{predictedTime}**")
