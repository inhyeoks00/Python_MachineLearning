import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
import requests

def fetch_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 404:
        print(f"Error: City '{city}' not found. Please check the city name.")
        return None
    elif response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def process_weather_data(raw_data, city):
    weather_data = []
    for entry in raw_data['list']:
        temp = entry['main']['temp'] + 273.15 
        humidity = entry['main']['humidity']
        weather_id = entry['weather'][0]['id']
        rain = 1 if 200 <= weather_id <= 531 else 0
        weather_data.append([city, temp, humidity, rain])
    
    df = pd.DataFrame(weather_data, columns=['City', 'Temperature', 'Humidity', 'Rain'])
    return df

def train_and_evaluate_model(data):
    X = data[['Temperature', 'Humidity']]
    y = data['Rain']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
    
    return model

if __name__ == "__main__":
    API_KEY = "ef6fdcf157ae61448bd466e8bc6e8ab0" 
    CITIES = ["New Orleans,US", "London,GB", "Sydney,AU", "Tokyo,JP", "Seoul,KR", "São Paulo,BR"]  
    
    all_data = pd.DataFrame()
    
    for city in CITIES:
        raw_data = fetch_weather_data(API_KEY, city)
        if raw_data:
            processed_data = process_weather_data(raw_data, city)
            all_data = pd.concat([all_data, processed_data], ignore_index=True)  
    
    print("Each city's first 2 rows:")
    print(all_data.groupby('City').head(2))  

  
    model = train_and_evaluate_model(all_data)
