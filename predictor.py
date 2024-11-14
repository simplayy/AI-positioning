import json
import os
import time
import pandas as pd
import numpy as np
import joblib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from database_manager import DatabaseManager  # Importazione della classe DatabaseManager
import datetime

base_dir = "/app/"
# base_dir = "/home/pi/"

# Carica modello, scaler e mapping
def load_resources(model_path, scaler_path, mac_mapping_path, training_columns_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(mac_mapping_path, 'r') as f:
        mac_mapping = json.load(f)
    with open(training_columns_path, 'r') as f:
        training_columns = json.load(f)
    return model, scaler, mac_mapping, training_columns

# Preprocessa i dati di input
def preprocess_input(input_data, mac_mapping, bs_columns):
    # Creazione del DataFrame
    input_df = pd.DataFrame([input_data])
    original_mac = input_df['mac_device'].iloc[0]
    print(f"Mac tracked device: {original_mac}")    
    
    # Assicurarsi che ogni colonna richiesta esista, altrimenti assegnarle un valore di 0
    for col in bs_columns:
        input_df[col] = input_df.get(col, pd.Series([0]))
    
    # Convertire tutti i valori in numerici, gestire errori con 'coerce' e riempire NaN con 110
    numeric_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return numeric_df

def remove_existing_file(file_path):
    if os.path.exists(file_path):
        print(f"Removing existing file {file_path}")
        os.remove(file_path)

# Gestore degli eventi per la creazione di file
class FileCreatedHandler(FileSystemEventHandler):
    def __init__(self, model, scaler, mac_mapping, training_columns, db_manager):
        self.model = model
        self.scaler = scaler
        self.mac_mapping = mac_mapping
        self.training_columns = training_columns
        self.db_manager = db_manager

    def on_created(self, event):
        if event.src_path.endswith('bangle_position.json'):
            print(f"Detected creation of {event.src_path}")
            time.sleep(1)  # Wait for the file to be completely written
            self.predict_and_cleanup(event.src_path)

    def predict_and_cleanup(self, file_path):
        print(f"Processing file {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)

        predictions = []
        current_timestamp = str(round(datetime.datetime.now().timestamp()))
        for mac_device, rssi_data in data.items():
            rssi_data['mac_device'] = mac_device
            preprocessed_input = preprocess_input(rssi_data, self.mac_mapping, self.training_columns)
            
            # Applicare la funzione su tutte le colonne di interesse usando applymap
            for col in self.training_columns:
                preprocessed_input[col] = preprocessed_input[col].map(lambda x: 120 - x if x != 0 else 0)

            # Check if all values in preprocessed_input are zero
            if (preprocessed_input[self.training_columns] == 0).all().all():
                predicted_room = 53550
                confidence_score = 100  # Optionally set confidence to a default value when skipping prediction
            else:
                predicted_proba = self.model.predict_proba(preprocessed_input[self.training_columns])[0]
                predicted_room = self.model.predict(preprocessed_input[self.training_columns])[0]
                confidence_score = np.max(predicted_proba) * 100

            print(f"Detected room ID: {predicted_room} with confidence: {confidence_score:.2f}%")
            # Print in the required format
            print(f'{{"mac":"{mac_device}", "position":"{predicted_room}", "accuracy":"{confidence_score:.2f}"}}')
            
            prediction = {
                'mac_device': mac_device,
                'predicted_room': str(round(predicted_room)),
                'confidence': f"{confidence_score:.2f}",
                'timestamp': current_timestamp  # Aggiunta del timestamp
            }
            predictions.append(prediction)
        self.db_manager.create_connection()
        self.db_manager.insert_json_data('predictions', predictions)
        
        os.remove(file_path)




if __name__ == "__main__":
    db_path = base_dir+'shared_dir/positioning.db'
    model_path = base_dir+'shared_dir/rf_model.pkl'
    scaler_path = base_dir+'shared_dir/scaler.pkl'
    mac_mapping_path = base_dir+'shared_dir/mac_mapping.json'
    training_columns_path = base_dir+'shared_dir/training_columns.json'
    
    # Elimina il file esistente all'avvio del programma
    remove_existing_file(base_dir+"shared_dir/bangle_position.json")
    # remove_existing_file(base_dir+"shared_dir/bangle_position_temp.json")

    model, scaler, mac_mapping, training_columns = load_resources(
        model_path, scaler_path, mac_mapping_path, training_columns_path
    )

    db_manager = DatabaseManager(db_path)
    event_handler = FileCreatedHandler(model, scaler, mac_mapping, training_columns, db_manager)
    observer = Observer()
    observer.schedule(event_handler, path=base_dir+'shared_dir', recursive=False)
    observer.start()
    print("Observer started.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
        db_manager.close_connection()
        print("Observer and database connection stopped.")
