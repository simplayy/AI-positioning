import sqlite3
from sqlite3 import Error
import json

class DatabaseManager:
    def __init__(self, db_file):
        """ Inizializza il gestore del database con il file del database. """
        self.db_file = db_file
        self.conn = self.create_connection()
        
    

    def create_connection(self):
        """ Crea una connessione al database SQLite. """
        try:
            conn = sqlite3.connect(self.db_file, check_same_thread = False)
            print("Connessione riuscita. Versione SQLite:", sqlite3.version)
            return conn
        except Error as e:
            print(e)
            return None

    def create_table(self, table_name, json_data):
        """Crea una tabella dinamicamente basata sui campi del JSON, se non esiste, inclusivo di un campo timestamp."""
        fields = json_data[0].keys()
        field_definitions = ', '.join([f"{field} TEXT" for field in fields])
        sql_create_table = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                                id INTEGER PRIMARY KEY,
                                {field_definitions}
                            );"""
        try:
            c = self.conn.cursor()
            c.execute(sql_create_table)
        except Error as e:
            print(e)


    def insert_json_data(self, table_name, json_data):
        """ Inserisce dati JSON nella tabella. Assicura prima che la tabella esista. """
        self.create_table(table_name, json_data)
        fields = json_data[0].keys()
        placeholders = ', '.join(['?' for _ in fields])
        sql = f"INSERT INTO {table_name}({', '.join(fields)}) VALUES({placeholders})"
        cur = self.conn.cursor()
        for data in json_data:
            cur.execute(sql, tuple(data[field] for field in fields))
        self.conn.commit()
        return cur.lastrowid


    def close_connection(self):
        """ Chiude la connessione al database. """
        if self.conn:
            self.conn.close()

    def read_from_table(self, table_name, filter_clause=None):
        """
        Legge i dati dalla tabella specificata.
        Può applicare una clausola di filtro opzionale per restringere i risultati.
        
        Args:
            table_name (str): Il nome della tabella da cui leggere i dati.
            filter_clause (str, optional): Una clausola WHERE SQL per filtrare i risultati, es. "age > 21".
            
        Returns:
            list[dict]: Una lista di dizionari dove ogni dizionario rappresenta una riga della tabella.
        """
        cur = self.conn.cursor()
        sql = f"SELECT * FROM {table_name}"
        if filter_clause:
            sql += f" WHERE {filter_clause}"
        cur.execute(sql)
        columns = [column[0] for column in cur.description]
        results = [dict(zip(columns, row)) for row in cur.fetchall()]
        return results
