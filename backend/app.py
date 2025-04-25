from fastapi import FastAPI
import uvicorn
from backend.ws_handler import websocket_endpoint

app = FastAPI()

# Đăng ký WebSocket endpoint
app.websocket("/ws")(websocket_endpoint)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
