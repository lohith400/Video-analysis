# Indian Road Intelligence System

Real-time traffic video analysis — YOLOv8 + EasyOCR + React dashboard.

---

## Project Structure

```
traffic_analysis/
├── server.py              ← FastAPI backend (bridges frontend ↔ pipeline)
├── main.py                ← CLI entry point (standalone, no frontend)
├── tracker.py
├── annotator.py
├── config.py
├── csv_logger.py
├── detector.py
├── download_models.py
├── ocr_engine.py
├── requirements.txt
├── models/
│   ├── yolov8n.pt                  ← auto-downloaded
│   └── license_plate_detector.pt  ← you must supply this
└── frontend/
    ├── index.html
    ├── package.json
    ├── tailwind.config.js
    ├── vite.config.js
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        ├── pages/
        │   ├── Home.jsx
        │   ├── LiveAnalysis.jsx
        │   └── Analytics.jsx
        └── components/
            ├── Navbar.jsx
            ├── VideoFeed.jsx
            ├── VehicleCounts.jsx
            ├── PlateTable.jsx
            ├── ViolationList.jsx
            ├── AlertBanner.jsx
            ├── MetricCard.jsx
            └── Charts.jsx
```

---

## Setup

### 1. Python backend

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart

# Download YOLOv8n weights
python download_models.py

# Place your plate detector model
# Copy license_plate_detector.pt → models/license_plate_detector.pt
```

### 2. Start the FastAPI server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be at `http://localhost:8000`.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## Usage

| Page | URL | Description |
|------|-----|-------------|
| Upload | `/` | Upload video/image or connect RTSP |
| Live | `/live` | Real-time WebSocket feed + stats |
| Analytics | `/analytics` | Charts + downloadable CSV history |

### WebSocket message format
```json
{
  "frame": "<base64-jpeg>",
  "fps": 24.3,
  "counts": { "car": 5, "truck": 2, "total": 12 },
  "plates": [{ "plate": "KA01AB1234", "timestamp": 1716100000000 }],
  "violations": [{ "type": "No Helmet", "plate": "KA01AB1234" }],
  "source": "VIDEO"
}
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload video/image file |
| POST | `/connect` | Connect RTSP/webcam |
| GET | `/analytics` | Return CSV rows as JSON |
| WS | `/video-feed` | WebSocket stream |

---

## CLI (no frontend)

```bash
python main.py --source path/to/video.mp4
python main.py --source rtsp://user:pass@192.168.1.1:554/stream
```
