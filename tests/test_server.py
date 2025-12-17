from fastapi.testclient import TestClient
from tbsoc.server.main import app
import os

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_list_files():
    response = client.get("/api/files")
    assert response.status_code == 200
    files = response.json()
    assert isinstance(files, list)

def test_serve_frontend():
    # Verify strict static file serving for root
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!doctype html>" in response.text.lower()
