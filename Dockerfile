# Utilizzare un'immagine base ufficiale di Python
FROM python:3.9-slim

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia i file dello script e altri file necessari nella directory di lavoro del container
COPY . /app

# Installa le dipendenze Python
RUN pip install pandas numpy watchdog joblib scikit-learn firebase-admin

# Espone la porta, se necessario (ad esempio, se il tuo predictor.py fornisce un API server)
# EXPOSE 8000
ENV PYTHONUNBUFFERED=1

# Comando per avviare lo script quando il container viene eseguito
CMD ["python3", "auto_train.py"]
