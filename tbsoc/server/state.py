from tbsoc.server.data_manager import DataManager
import os
import threading

class LogBuffer:
    def __init__(self, max_lines=1000):
        self.buffer = []
        self.max_lines = max_lines
        self.lock = threading.Lock()

    def write(self, message):
        with self.lock:
            # Handle multi-line messages
            lines = message.split('\n')
            for line in lines:
                if line: # Filter empty lines if desired, or keep format
                    self.buffer.append(line)
            
            # Trim buffer
            if len(self.buffer) > self.max_lines:
                self.buffer = self.buffer[-self.max_lines:]

    def flush(self):
        pass

    def get_logs(self):
        with self.lock:
            return list(self.buffer)

class GlobalState:
    def __init__(self):
        self.window = None
        self.current_directory = os.getcwd()
        self.data_manager = DataManager()
        self.log_buffer = LogBuffer()

# Singleton instance
state = GlobalState()
