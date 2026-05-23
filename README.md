# Indian Road Intelligence System

Real-time traffic video analysis and Automatic Number Plate Recognition (ANPR) system featuring a custom-trained **YOLOv8 license plate detector** and **EasyOCR** pipeline connected to a beautiful **React (Vite) dashboard**.

This repository has been fully configured and optimized to run **natively on Windows** with **GPU acceleration** powered by your NVIDIA GeForce RTX 3050 Laptop GPU.

---

## 🚀 How to Run the Project (Windows Native)

Follow these simple steps to start both the backend API and frontend dashboard natively on Windows.

### Step 1: Start the Python Backend (FastAPI)
1. Open a terminal (Git Bash, Command Prompt, or PowerShell) in the project root directory.
2. Run the server directly using the virtual environment's Python interpreter (this is the most reliable way on Windows and avoids any global path/activation errors):
   ```cmd
   .venv/Scripts/python -m uvicorn server:app --host 127.0.0.1 --port 8000
   ```
   *The backend will load the custom license plate model (`models/license_plate_detector.pt`) and initialize the EasyOCR engine with RTX 3050 GPU acceleration. It will be listening on `http://127.0.0.1:8000`.*

### Step 2: Start the React Frontend (Vite)
1. Open a **second** command line terminal in the project directory.
2. Navigate to the frontend directory:
   ```cmd
   cd frontend
   ```
3. Launch the web dashboard development server:
   ```cmd
   npm run dev
   ```
   *The React application will compile and open at **[http://localhost:3000](http://localhost:3000)**.*

---

## 🛠️ Windows Optimization & Setup Details

### ⚡ GPU Hardware Acceleration
The local environment is configured with a high-performance **CUDA 12.4 PyTorch** wheel (`torch-2.6.0+cu124`).
* To check or verify your GPU binding status at any time, run:
  ```cmd
  .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
  ```
  *(Should print `True` for your NVIDIA GeForce RTX 3050 Laptop GPU).*

### 📁 Local Bounding Box / Dataset Path
The custom license plate dataset configuration (`plate_dataset/data.yaml`) is reconfigured to target absolute Windows paths with forward-slash formatting:
```yaml
path: C:/Users/lohit/.vscode/Code/OWN/traffic_analysis/plate_dataset
```

### 🐚 Script Shell Customization (`.npmrc`)
To prevent Windows script execution policy errors (`EPERM` / `spawn /bin/bash`) during npm installs and Vite dev startups, a local `frontend/.npmrc` file is configured with:
```ini
script-shell=cmd.exe
```
This forces all lifecycle hooks to run natively through Windows command prompt rather than expecting a Linux bash terminal.

---

## 📁 Project Structure

```
traffic_analysis/
├── server.py                 # FastAPI backend (bridges WebSocket ↔ OCR Pipeline)
├── main.py                   # CLI entry point (standalone, no frontend)
├── tracker.py                # Multi-object tracker
├── annotator.py              # Visual annotator utilities
├── config.py                 # Pipeline thresholds & visual configs
├── csv_logger.py             # CSV output generator
├── detector.py               # Custom YOLO vehicle + plate detector wrappers
├── ocr_engine.py             # Multi-threaded asynchronous EasyOCR engine
├── requirements.txt          # Core Python dependencies
├── models/
│   ├── yolov8n.pt            # Auto-downloaded base vehicle tracking model
│   └── license_plate_detector.pt # Custom YOLOv8n license plate detector (90.6% mAP50!)
├── plate_dataset/            # Custom training dataset (images and data.yaml)
└── frontend/
    ├── package.json          # React Vite configurations
    ├── .npmrc                # Local Windows npm override
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── index.css         # Styling system
        ├── pages/
        │   ├── Home.jsx      # Video upload & config drag-and-drop
        │   ├── LiveAnalysis.jsx # Real-time tracking and raw OCR logs
        │   └── Analytics.jsx # Charts & historical logs
        └── components/       # Custom React widgets
```

---

## 📡 API Endpoints & Interfaces

### WebSocket Interface
The frontend communicates with the backend via WebSocket stream:
* **Endpoint:** `ws://127.0.0.1:8000/video-feed`
* **Message Payload:**
  ```json
  {
    "frame": "<base64-jpeg>",
    "fps": 24.3,
    "counts": { "car": 5, "truck": 2, "motorcycle": 4, "total": 11 },
    "plates": [{ "plate": "KA01AB1234", "timestamp": 1716100000000 }],
    "violations": [{ "type": "No Helmet", "plate": "KA01AB1234" }],
    "source": "VIDEO"
  }
  ```

### HTTP Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API service health check |
| POST | `/upload` | Upload and prepare video/image files |
| POST | `/connect` | Establish webcam or RTSP feed bridge |
| GET | `/analytics` | Fetch logged vehicles and recognized plates |

---

## 💻 CLI Usage (No Frontend)

You can also run the traffic analysis engine directly as a command-line script:

```bash
# Analyze a local video file
python main.py --source uploads/L2.mp4

# Analyze a live IP camera or RTSP stream
python main.py --source rtsp://user:pass@192.168.1.1:554/stream
```
