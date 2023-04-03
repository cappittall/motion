@echo off
call myenv\Scripts\activate
uvicorn motions:app --host 0.0.0.0 --port 8000 --reload
pause