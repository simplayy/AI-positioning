import pandas as pd
import numpy as np
import joblib
import json
import firebase_admin
import datetime
import subprocess

from firebase_admin import credentials, firestore
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from database_manager import DatabaseManager  # Importazione della classe DatabaseManager
from sklearn.neighbors import KNeighborsClassifier
import ast

base_dir = "/app/"
#base_dir = "/home/pi/"

# Configurazione Firebase
cred = credentials.Certificate("config_fb.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Funzione per ottenere l'indirizzo MAC di eth0 senza usare pacchetti aggiuntivi
def get_mac_address(interface='eth0'):
    try:
        result = subprocess.run(['ip', 'link', 'show', interface], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stdout.split('\n'):
            if 'link/ether' in line:
                mac_address = line.split()[1]
                return mac_address
        return None
    except Exception as e:
        print(f"Error getting MAC address: {e}")
        return None
    
    
# Funzione per ottenere il documento più recente
def get_latest_calibration_document():
    path = f'organization/JQuaztkKOdPqLVBUAHzs/network/{mac_address}/calibration'
    docs = db.collection(path).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
    latest_doc = next(docs, None)
    return latest_doc.to_dict() if latest_doc else None

def calculate_bs_stats(df):
    # Remove 'fuoricasa' rows and reshape the DataFrame
    df = df[df['room'] != 0]
    melted_df = df.melt(id_vars=['mac_device', 'room', 'index'], var_name='mac_bs', value_name='rssi_value')
    
    # Assuming rssi_value has been previously transformed to '120 - original rssi'
    # Reverse the transformation to get back to the original rssi values
    melted_df['rssi_value'] = 120 - melted_df['rssi_value']
    
    # Calculate statistics for each room and mac_bs combination using the original RSSI values
    stats_df = melted_df[(melted_df['rssi_value'] != 120)].groupby(['room', 'mac_bs']).agg(
        mean=('rssi_value', 'mean'),
        max=('rssi_value', 'max'),
        min=('rssi_value', 'min'),
        count=('rssi_value', 'count')
    ).reset_index()

    return stats_df

def save_bs_stats_to_db(db_manager, stats_df, calibration_id):
    bs_stats_data = [{
        'calibration_id': calibration_id,
        'room_id': row['room'],
        'mac_bs': row['mac_bs'],
        'mean': row['mean'],
        'max': row['max'],
        'min': row['min'],
        'count': row['count'],
        'timestamp': str(round(datetime.datetime.now().timestamp()))
    } for index, row in stats_df.iterrows()]
    
    db_manager.insert_json_data('bs_stats', bs_stats_data)

def save_room_stats_to_db(db_manager, room_times, calibration_id):
    print(room_times)
    room_stats_data = [{
        'calibration_id': calibration_id,
        'room': room,
        'time': seconds,
        'timestamp': str(round(datetime.datetime.now().timestamp()))
    } for room, seconds in room_times.items()]
    print(room_stats_data)
    db_manager.insert_json_data('room_stats', room_stats_data)
    
def load_and_preprocess_data(filepath):
    # Legge i dati dal file CSV
    data = pd.read_csv(filepath)
    
    # Converti 'rssi_values' da stringa a lista effettiva di numeri
    data['rssi_values'] = data['rssi_values'].apply(ast.literal_eval)
    
    # Raggruppa i dati per mac_device, room, e mac_bs
    grouped_data = {}
    for index, row in data.iterrows():
        key = (row['mac_device'], row['room'], row['mac_bs'])
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].extend(row['rssi_values'])
    
    # Trasforma il dizionario in un DataFrame lungo con multiindice
    rows = []
    for (mac_device, room, mac_bs), rssi_values in grouped_data.items():
        for i, rssi in enumerate(rssi_values):
            rows.append({
                'mac_device': mac_device,
                'room': room,
                'mac_bs': mac_bs,
                'index': i,
                'rssi_value': 120 - rssi if rssi != 0 else 0
            })
    
    long_data = pd.DataFrame(rows)
    
    # Pivot il DataFrame per ottenere mac_bs come colonne e index come indice di riga
    final_data = long_data.pivot_table(
        index=['mac_device', 'room', 'index'],
        columns='mac_bs',
        values='rssi_value',
        aggfunc='first'
    ).reset_index().fillna(0)
    
    # Riempi gli spazi vuoti con 0
    bs_columns = data['mac_bs'].unique()
    for col in bs_columns:
        if col not in final_data:
            final_data[col] = 0.0
    
    return final_data

def replace_nan_logic(dataframe):
    df = dataframe.copy()
    rssi_columns = [col for col in df.columns if col not in ['mac_device', 'room', 'index']]
    for col in rssi_columns:
        for room in df['room'].unique():
            mask = (df['room'] == room)
            if df.loc[mask, col].notna().any():
                df.loc[mask, col] = df.loc[mask, col].fillna(0)
            else:
                df.loc[mask, col] = df.loc[mask, col].fillna(0)
    return df

def ensure_numeric(dataframe, db_manager):
    # Generate MAC mapping
    mac_mapping = {mac: idx for idx, mac in enumerate(dataframe['mac_device'].unique(), start=1)}

    # Save the MAC mapping to a JSON file
    with open(base_dir+'shared_dir/mac_mapping.json', 'w') as file:
        json.dump(mac_mapping, file)

    # Apply mappings to the DataFrame
    dataframe['mac_device'] = dataframe['mac_device'].map(mac_mapping)

    # Convert all data in the DataFrame to numeric format and handle errors
    numeric_df = dataframe.apply(pd.to_numeric, errors='coerce').fillna(0)

    return numeric_df, mac_mapping

def preprocess_input(input_data, mac_mapping, bs_columns):
    input_df = pd.DataFrame([input_data])
    input_df['mac_device'] = input_df['mac_device'].map(mac_mapping)
    for col in bs_columns:
        if col not in input_df.columns:
            input_df[col] = 0
        input_df[col] = input_df[col].fillna(0)
    numeric_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return numeric_df

db_file_path = base_dir+'shared_dir/positioning.db'
db_manager = DatabaseManager(db_file_path)

# Percorso del file CSV
file_path = base_dir+'shared_dir/calibration_results.csv'
prepared_data = load_and_preprocess_data(file_path)
clean_data = prepared_data

clean_data.to_csv('cleaned_data.csv', index=False)

numeric_data, mac_mapping = ensure_numeric(clean_data, db_manager)
clean_data.to_csv('cleaned_data_number.csv', index=False)

# Dividere i dati in caratteristiche (X) e target (y)
X = numeric_data.drop(columns=['room', 'mac_device', 'index'])
y = numeric_data['room']

# Salva le colonne di addestramento
training_columns = X.columns.tolist()
with open(base_dir+'shared_dir/training_columns.json', 'w') as f:
    json.dump(training_columns, f)

# Dividere i dati in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardizzare le caratteristiche
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creare e addestrare il modello con parametri modificati
model = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=4, random_state=42)

model.fit(X_train, y_train)

# Predire sui dati di test
y_pred = model.predict(X_test)

# Valutare il modello
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Salva il modello e lo scaler
model_path = base_dir+'shared_dir/rf_model.pkl'
scaler_path = base_dir+'shared_dir/scaler.pkl'
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Il modello è stato salvato in: {model_path}")
print(f"Lo scaler è stato salvato in: {scaler_path}")

# Recupera i dati più recenti da Firebase
latest_data = get_latest_calibration_document()
if latest_data:
    # Inserisci i dettagli della calibrazione in SQLite
    calibration_info = {
        "name": str(latest_data.get("nome", "Unknown")),
        "timestamp": str(latest_data.get("timestamp", "")),
        "description": str(latest_data.get("descrizione", "")),
        "device": str(latest_data.get("device", "")),
        "accuracy": str(f"{accuracy * 100:.2f}")   # Placeholder per l'accuratezza
    }
    calibration_id = db_manager.insert_json_data("calibration_details", [calibration_info])

    # Assume data_df è il DataFrame elaborato contenente i dati di calibrazione
    stats_df = calculate_bs_stats(clean_data)
    
    # Mostra un esempio delle statistiche
    save_bs_stats_to_db(db_manager, stats_df, calibration_id)

    save_room_stats_to_db(db_manager, latest_data.get("time", {}), calibration_id)

# Funzione di predizione con la logica "fuoricasa"
def predict_room(input_data):
    preprocessed_input = preprocess_input(input_data, mac_mapping, training_columns)

    # Aggiungi le colonne mancanti per corrispondere al training
    for col in training_columns:
        if col not in preprocessed_input.columns:
            preprocessed_input[col] = 0

    # Ordina le colonne per corrispondere al modello di addestramento
    preprocessed_input = preprocessed_input[training_columns]

    # Verifica se tutti i valori sono zero
    if (preprocessed_input == 0).all().all():
        predicted_room_name = 'fuoricasa'
        confidence_score = 100
    else:
        # Standardizzare la rilevazione
        scaled_input = scaler.transform(preprocessed_input)

        # Predire la stanza e le probabilità
        predicted_proba = model.predict_proba(scaled_input)[0]
        predicted_room = model.predict(scaled_input)[0]

        # Calcolare la percentuale di sicurezza
        confidence_score = np.max(predicted_proba) * 100

    print(f"La rilevazione appartiene alla stanza: {predicted_room}")
    print(f"Percentuale di sicurezza: {confidence_score:.2f}%")
    return predicted_room, confidence_score

db_manager.close_connection()
