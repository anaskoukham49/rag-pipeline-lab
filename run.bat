@echo off
echo Lancement TP3 RAG - FastAPI + Frontend
echo Backend: http://127.0.0.1:8000  Docs: http://127.0.0.1:8000/docs  App: http://127.0.0.1:8000/app
echo Frontend fichier: frontend\index.html  (ou via /app)
".venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
