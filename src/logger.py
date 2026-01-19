import csv
import os
from datetime import datetime

class SessionLogger:
    def __init__(self, filename="drowsiness_session.csv"):
        self.filename = filename
        self.file_exists = os.path.isfile(self.filename)

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not self.file_exists:
                writer.writerow(["timestamp", "ear", "drowsy"])

    def log(self, ear, drowsy):
        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                round(ear, 3),
                int(drowsy)
            ])
