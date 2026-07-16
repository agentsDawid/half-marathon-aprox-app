from dotenv import load_dotenv
from typing import Optional
from openai import OpenAI
import os
from pydantic import BaseModel, Field
from langfuse_client import langfuse

# import instructor
# from langfuse.decorators import observe
# from langfuse.openai import OpenAI


### -------- Constants
OPENAI_MODEL = "gpt-4o-mini"

### -------- OPEN AI - Load Model name
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Brak OPENAI API-KEY - sprawdź zmienne środowiskowe")

client = OpenAI(api_key=openai_api_key)
# client = instructor.from_openai(OpenAI(api_key=openai_api_key)) # z użyciem instructor

### -------- Data model
class RunInputData(BaseModel):
    imie: Optional[str] = Field(default=None, description="Imię użytkownika")
    plec: Optional[str] = Field(default=None, description="Płeć: K lub M")
    rocznik: Optional[int] = Field(default=None, description="Rok urodzenia, np. 1995")
    wiek: Optional[int] = Field(default=None, description="Wiek w latach, jeśli user podał wiek zamiast roku urodzenia")
    czas_5km: Optional[str] = Field(default=None, description="Czas na 5km w formacie MM:SS, np. '24:30'")



### -------- BOT: Data Extraction
def extract_data(historia_wiadomosci) -> RunInputData:
    messages=[
        {
            "role": "system",
            "content": (
                "Wyciągnij dane biegacza z poniższej konwersacji. "
                "Jeśli użytkownik podał rok urodzenia — wpisz go w pole 'rocznik'."
                "Jeśli podał swój wiek (liczbę lat) — wpisz go w pole 'wiek', "
                "NIE przeliczaj samodzielnie na rocznik. "
                "Jeśli czegoś brak w rozmowie, zostaw jako null."
            )
        },
        *historia_wiadomosci
    ]
    
    trace = langfuse.trace(
        name="half-marathon-predicton",
        input=messages,
        # metadata={"response_format":{"type": "json_object"}}
    )

    span = trace.span(
        name="runner-data-History-message",
        input={"Message_History": historia_wiadomosci}
        )

    generation = span.generation(
        name="data-extraction",
        model=OPENAI_MODEL,
        input=messages,
    )
    
    response = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=messages,
        response_format=RunInputData,
    )
    
    # response = client.chat.completions.create( # z użyciem instructor
    #     model=OPENAI_MODEL,
    #     response_model=RunInputData,
    #     temperature=0,
    #     messages=messages,
    # )
    
    generation.end(
        output=response.model_dump(),
    )

    span.end(output=response)
    langfuse.flush()
    
    return response.choices[0].message.parsed
    # return response # z użyciem instructor

def check_missing_data(dane: RunInputData) -> list[str]:
    """Zwraca listę nazw brakujących pól (po polsku, do wyświetlenia userowi)."""
    missing = []
    if not dane.plec:
        missing.append("płeć")
    if not dane.rocznik and not dane.wiek:
        missing.append("rok urodzenia lub wiek")
    if not dane.czas_5km:
        missing.append("czas na 5km (format MM:SS)")
    return missing

# def missing_status(dane: RunInputData, missing: list[str]) -> dict:
#     zwrot = f"{dane.imie}, " if dane.imie else ""  # personalizacja, jeśli znamy imię

#     if not missing:
#         content = f"{zwrot}mam komplet danych! Liczę estymację czasu półmaratonu... 🏃"
#     else:
#         content = f"{zwrot}potrzebuję jeszcze: " + ", ".join(missing) + ". Podaj proszę te informacje."
#         content = content[0].upper() + content[1:]  # duża litera na początku zdania

#     return {"role": "assistant", "content": content}

def missing_status(dane: RunInputData, missing: list[str]) -> str:
    zwrot = f"{dane.imie}, " if dane.imie else ""

    if not missing:
        content = f"{zwrot}mam komplet danych! Liczę estymację czasu półmaratonu... 🏃"
    else:
        content = f"{zwrot}potrzebuję jeszcze: " + ", ".join(missing) + ". Podaj proszę te informacje."
        content = content[0].upper() + content[1:]

    return content

def missing_status_llm(dane: RunInputData, missing: list[str]) -> dict:
    if not missing:
        content = f"{dane.imie + ', ' if dane.imie else ''}mam komplet danych! Liczę estymację... 🏃"
        return {"role": "assistant", "content": content}

    prompt = (
        f"Imię użytkownika: {dane.imie if dane.imie else 'nieznane, pomiń zwracanie się po imieniu'}. "
        f"Brakujące dane do dopytania: {', '.join(missing)}. "
        "Napisz krótkie, uprzejme jedno zdanie proszące o te dane. "
        "Jeśli imię jest znane, zwróć się po imieniu naturalnie."
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Jesteś asystentem zbierającym dane do estymacji czasu biegu."},
            {"role": "user", "content": prompt},
        ],
    )
    return {"role": "assistant", "content": response.choices[0].message.content}