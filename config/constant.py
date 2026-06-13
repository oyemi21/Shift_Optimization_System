import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database_path = os.path.abspath(os.path.join(BASE_DIR, 'ShiftData.db'))
feature_artifact = os.path.abspath(os.path.join(BASE_DIR, 'artifact', 'features', 'feature_engineerin.csv'))
SCHEMA_PATH = os.path.abspath(os.path.join(BASE_DIR, 'config', 'schema.yaml'))


LOG_DIR = "logs"
LOG_FILE = "app.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

log_folder_path = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(log_folder_path, exist_ok=True)

log_file_path = os.path.join(log_folder_path, LOG_FILE) 

target_column = "shift_efficiency_score"
