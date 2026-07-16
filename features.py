import datetime
import math
import pandas as pd
from llm import RunInputData

def unify_age(data: RunInputData) -> int | None:
    

    actual_data = datetime.date.today().year

    
    if data.rocznik is not None:
        if 1920 <= data.rocznik <= actual_data - 5:
            return data.rocznik
        return None

    if data.wiek is not None:
        if 5 <= data.wiek <= 100:
            return actual_data - data.wiek
        return None
    
    return None

def fill_year_category(age: int, gender: str) -> str | None:
    tGender = gender.upper().strip()
    if tGender not in ("K", "M"):
        raise ValueError(f"Płeć musi być 'K' lub 'M', otrzymano: '{gender}'")

    if age > 0:
        r = math.floor(age/10)*10
        
        if r < 20:
            r = 20
        elif r > 100:
            r = 100
        return f'{tGender}{r}'
    return None

def convert_time_to_seconds(time):
    if pd.isnull(time) or time in ['DNS', 'DNF']:
        return None
    if  not isinstance(time, str):
        return time
    try:
        time = time.split(':')
        if len(time) < 3:
            time.insert(0, '0')
        return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
    except (ValueError, IndexError):
        return None

def convert_seconds_to_time(seconds) -> str | None:
    if seconds is None:
        return None
    
    # Jeśli to już jest string (np. "24:15"), po prostu go zwróć
    if isinstance(seconds, str):
        return seconds
    
    # Jeśli dostaliśmy liczbę w formacie tekstowym (np. "1455"), 
    # spróbujmy ją bezpiecznie przekonwertować na float/int
    try:
        seconds = float(seconds)
    except (ValueError, TypeError):
        return str(seconds) # Jeśli konwersja się nie uda, zwracamy jako tekst
    
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    elif m > 0:
        return f"{m}:{s:02d}"
    else:
        return f"{s}s"

def merge_data(old: RunInputData | None, new: RunInputData) -> RunInputData:
    if old is None:
        return new

    merged = old.model_copy()
    for field in RunInputData.model_fields:
        new_value = getattr(new, field)
        if new_value is not None:
            setattr(merged, field, new_value)
            
            if field == "rocznik":
                merged.wiek = None
            elif field == "wiek":
                merged.rocznik = None
    return merged