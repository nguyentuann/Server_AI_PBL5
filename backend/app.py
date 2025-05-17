import os
import uvicorn
import threading

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from rabbitMQ import start_server

app = FastAPI()

@app.get("/")
def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Server</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            h1 {
                font-size: 30px;
                color: #222;
            }
        </style>
    </head>
    <body>
        <h1>WELCOME TO SERVER AI</h1>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Khởi chạy server xử lý AI trong luồng song song
def run_background_ai():
    print("🚀 Starting AI server in the background...")
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()

run_background_ai()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
