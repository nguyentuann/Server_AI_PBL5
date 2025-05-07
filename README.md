# Server_AI_PBL5

A machine learning-based system for real-time squat form detection, analysis, and feedback.

## Project Overview

This project uses AI to analyze squat form from skeletal keypoints, detect errors in posture, count repetitions, and provide real-time feedback. The system includes:

- AI models for pose detection and error classification
- WebSocket and RabbitMQ communication interfaces
- Frontend interfaces for viewing streams and feedback

## System Architecture

The project consists of three main components:

1. **AI Core**: Detects pose errors and counts repetitions
2. **Backend Server**: Handles communication and data processing
3. **Frontend**: User interface for viewing streams and receiving feedback

## Technologies Used

- **AI & ML**: TensorFlow, MediaPipe
- **Backend**: FastAPI, RabbitMQ, WebSockets
- **Frontend**: HTML, JavaScript, WebRTC
- **Data Processing**: NumPy, Pandas

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
