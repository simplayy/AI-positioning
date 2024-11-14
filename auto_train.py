import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

base_dir = "/app/"
# base_dir = "/home/pi/"

MODEL_FILE_PATH = os.path.join(base_dir, 'shared_dir', 'rf_model.pkl')

class TrainingHandler(FileSystemEventHandler):
    def __init__(self, filepath, predictor_script_path, training_script_path):
        self.filepath = os.path.abspath(filepath)
        self.predictor_script_path = predictor_script_path
        self.training_script_path = training_script_path
        self.predictor_process = None
        self.is_training = False
        self.last_modified_time = None

    def on_modified(self, event):
        if event.src_path == self.filepath:
            self.last_modified_time = time.time()

    def check_for_updates(self):
        """Check if the file was modified during training and trigger retraining if necessary."""
        if self.last_modified_time and time.time() - self.last_modified_time > 1 and not self.is_training:
            self.start_training_cycle()

    def start_training_cycle(self):
        self.stop_predictor()
        self.train_model()
        self.start_predictor()
        self.last_modified_time = None

    def train_model(self):
        """Function to perform the training process."""
        self.is_training = True
        print("Starting model training...")
        subprocess.run(['python3', self.training_script_path], check=True)
        self.is_training = False

    def start_predictor(self):
        """Function to start the predictor.py script."""
        if self.predictor_process is None:
            self.predictor_process = subprocess.Popen(['python3', self.predictor_script_path])

    def stop_predictor(self):
        """Function to terminate the predictor.py script if it is running."""
        if self.predictor_process:
            self.predictor_process.terminate()
            self.predictor_process.wait()
            self.predictor_process = None

# File and script paths
FILE_TO_WATCH = os.path.join(base_dir, 'shared_dir', 'calibration_results.csv')
PREDICTOR_SCRIPT_PATH = 'predictor.py'
TRAINING_SCRIPT_PATH = 'main.py'

# Setup file system monitoring
event_handler = TrainingHandler(FILE_TO_WATCH, PREDICTOR_SCRIPT_PATH, TRAINING_SCRIPT_PATH)
observer = Observer()
observer.schedule(event_handler, path=os.path.dirname(FILE_TO_WATCH), recursive=False)

# Start monitoring
observer.start()

# Verify model existence and start predictor at launch
if os.path.exists(MODEL_FILE_PATH):
    event_handler.start_predictor()

try:
    while True:
        event_handler.check_for_updates()
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
    observer.join()
