from fastapi import FastAPI
import threading
from rabbitMQ import start_server
import uvicorn
import os

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Server is running..."}

# Khởi chạy server xử lý AI trong luồng song song
def run_background_ai():
    print("🚀 Starting AI server in the background...")
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()

run_background_ai()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
