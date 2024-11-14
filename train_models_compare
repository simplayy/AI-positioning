import pandas as pd
import numpy as np
import joblib
import json
import firebase_admin
import datetime
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from firebase_admin import credentials, firestore
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from database_manager import DatabaseManager  # Importazione della classe DatabaseManager
import ast

# Ignora i warning per una visualizzazione più pulita
warnings.filterwarnings('ignore')

# Impostazioni per la visualizzazione dei grafici
plt.style.use('ggplot')

# base_dir = "/app/"
base_dir = "/home/pi/"

# Creazione della directory per salvare i grafici
graphs_dir = os.path.join(base_dir, 'shared_dir', 'graphs')
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

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

mac_address = get_mac_address()

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizzare le caratteristiche
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **Creare e addestrare diversi modelli di machine learning**

models = {
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=4, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

model_results = {}
for model_name, model in models.items():
    # Addestramento del modello
    model.fit(X_train_scaled, y_train)
    # Predizioni
    y_pred = model.predict(X_test_scaled)
    # Valutazione
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True)
    model_results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'y_pred': y_pred
    }
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# **Confronto delle performance dei modelli**

# Creazione di un DataFrame per le metriche
metrics_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Accuracy': [result['accuracy'] for result in model_results.values()],
    'Precision': [result['precision'] for result in model_results.values()],
    'Recall': [result['recall'] for result in model_results.values()],
    'F1 Score': [result['f1_score'] for result in model_results.values()]
})

# Visualizzazione e salvataggio dei grafici delle metriche

# Grafico delle accuratezze
plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='Accuracy', data=metrics_df)
plt.title('Confronto delle Accuratezze dei Modelli')
plt.ylabel('Accuratezza')
accuracy_comparison_path = os.path.join(graphs_dir, 'model_accuracy_comparison.png')
plt.savefig(accuracy_comparison_path)
plt.close()
print(f"Il confronto delle accuratezze è stato salvato in: {accuracy_comparison_path}")

# Grafico delle precisioni
plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='Precision', data=metrics_df)
plt.title('Confronto delle Precisioni dei Modelli')
plt.ylabel('Precisione')
precision_comparison_path = os.path.join(graphs_dir, 'model_precision_comparison.png')
plt.savefig(precision_comparison_path)
plt.close()
print(f"Il confronto delle precisioni è stato salvato in: {precision_comparison_path}")

# Grafico dei richiami
plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='Recall', data=metrics_df)
plt.title('Confronto dei Richiami dei Modelli')
plt.ylabel('Richiamo')
recall_comparison_path = os.path.join(graphs_dir, 'model_recall_comparison.png')
plt.savefig(recall_comparison_path)
plt.close()
print(f"Il confronto dei richiami è stato salvato in: {recall_comparison_path}")

# Grafico degli F1 Score
plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='F1 Score', data=metrics_df)
plt.title('Confronto degli F1 Score dei Modelli')
plt.ylabel('F1 Score')
f1_comparison_path = os.path.join(graphs_dir, 'model_f1_comparison.png')
plt.savefig(f1_comparison_path)
plt.close()
print(f"Il confronto degli F1 Score è stato salvato in: {f1_comparison_path}")

# **Interpretazione dei grafici e dei dati**

# Salvare il modello migliore (in base all'F1 Score)
best_model_name = metrics_df.loc[metrics_df['F1 Score'].idxmax(), 'Model']
best_model = model_results[best_model_name]['model']
best_accuracy = model_results[best_model_name]['accuracy']
print(f"\nIl modello migliore è: {best_model_name} con un F1 Score del {metrics_df['F1 Score'].max() * 100:.2f}%")

# Salvare il modello e lo scaler
model_path = base_dir+f'shared_dir/{best_model_name.lower().replace(" ", "_")}_model.pkl'
scaler_path = base_dir+'shared_dir/scaler.pkl'
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Il modello migliore è stato salvato in: {model_path}")
print(f"Lo scaler è stato salvato in: {scaler_path}")

# **Matrice di confusione per il modello migliore**

best_y_pred = model_results[best_model_name]['y_pred']
cm = confusion_matrix(y_test, best_y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Matrice di Confusione - {best_model_name}')
best_cm_path = os.path.join(graphs_dir, f'confusion_matrix_{best_model_name.lower().replace(" ", "_")}.png')
plt.savefig(best_cm_path)
plt.close()
print(f"La matrice di confusione del modello migliore è stata salvata in: {best_cm_path}")

# **Importanza delle caratteristiche (solo per modelli che la supportano)**

if hasattr(best_model, 'feature_importances_'):
    feature_importances = best_model.feature_importances_
    features = X.columns

    fi_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    fi_df = fi_df.sort_values('Importance', ascending=False)

    plt.figure(figsize=(12,6))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title(f'Importanza delle Caratteristiche - {best_model_name}')
    fi_path = os.path.join(graphs_dir, f'feature_importances_{best_model_name.lower().replace(" ", "_")}.png')
    plt.savefig(fi_path)
    plt.close()
    print(f"L'importanza delle caratteristiche è stata salvata in: {fi_path}")

# **Visualizzazione delle distribuzioni RSSI per ogni stanza**

melted_data = pd.melt(numeric_data, id_vars=['room'], value_vars=X.columns, var_name='Beacon', value_name='RSSI')
plt.figure(figsize=(12,8))
sns.boxplot(x='room', y='RSSI', data=melted_data)
plt.title('Distribuzione RSSI per Stanza')
plt.xlabel('Stanza')
plt.ylabel('RSSI')
rssi_distribution_path = os.path.join(graphs_dir, 'rssi_distribution.png')
plt.savefig(rssi_distribution_path)
plt.close()
print(f"La distribuzione RSSI per stanza è stata salvata in: {rssi_distribution_path}")

# Recupera i dati più recenti da Firebase
latest_data = get_latest_calibration_document()
if latest_data:
    # Inserisci i dettagli della calibrazione in SQLite
    calibration_info = {
        "name": str(latest_data.get("nome", "Unknown")),
        "timestamp": str(latest_data.get("timestamp", "")),
        "description": str(latest_data.get("descrizione", "")),
        "device": str(latest_data.get("device", "")),
        "accuracy": str(f"{best_accuracy * 100:.2f}")   # Accuratezza del modello migliore
    }
    calibration_id = db_manager.insert_json_data("calibration_details", [calibration_info])

    # Calcola le statistiche dei beacon station (bs)
    stats_df = calculate_bs_stats(clean_data)

    # Mostra un esempio delle statistiche
    print("Beacon Station Statistics:")
    print(stats_df.head())

    save_bs_stats_to_db(db_manager, stats_df, calibration_id)

    save_room_stats_to_db(db_manager, latest_data.get("time", {}), calibration_id)

    # **Visualizzazione delle statistiche delle Beacon Station**

    # Grafico delle medie RSSI per ogni Beacon Station
    plt.figure(figsize=(12,8))
    sns.barplot(x='mac_bs', y='mean', hue='room', data=stats_df)
    plt.title('Media RSSI per Beacon Station e Stanza')
    plt.xlabel('Beacon Station')
    plt.ylabel('Media RSSI')
    plt.legend(title='Stanza')
    bs_mean_rssi_path = os.path.join(graphs_dir, 'bs_mean_rssi.png')
    plt.savefig(bs_mean_rssi_path)
    plt.close()
    print(f"La media RSSI per Beacon Station è stata salvata in: {bs_mean_rssi_path}")

    # Grafico del conteggio delle rilevazioni per Beacon Station
    plt.figure(figsize=(12,8))
    sns.barplot(x='mac_bs', y='count', hue='room', data=stats_df)
    plt.title('Conteggio Rilevazioni per Beacon Station e Stanza')
    plt.xlabel('Beacon Station')
    plt.ylabel('Conteggio Rilevazioni')
    plt.legend(title='Stanza')
    bs_count_path = os.path.join(graphs_dir, 'bs_count.png')
    plt.savefig(bs_count_path)
    plt.close()
    print(f"Il conteggio delle rilevazioni per Beacon Station è stato salvato in: {bs_count_path}")

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
        predicted_proba = best_model.predict_proba(scaled_input)[0]
        predicted_room = best_model.predict(scaled_input)[0]

        # Calcolare la percentuale di sicurezza
        confidence_score = np.max(predicted_proba) * 100

    print(f"La rilevazione appartiene alla stanza: {predicted_room}")
    print(f"Percentuale di sicurezza: {confidence_score:.2f}%")
    return predicted_room, confidence_score

db_manager.close_connection()
