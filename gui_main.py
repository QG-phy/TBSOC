import os
import sys

# Ensure tbsoc is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from tbsoc.server.main import start_desktop_app

if __name__ == '__main__':
    start_desktop_app()
