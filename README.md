# Indian Road Intelligence System

Real-time traffic video analysis, Automatic Number Plate Recognition (ANPR), and safety compliance monitoring system. Features custom-trained **YOLOv8** detectors connected to a beautiful **React (Vite) dashboard**.

This repository is optimized to run **natively on Windows** with **GPU acceleration** powered by your NVIDIA GeForce RTX 3050 Laptop GPU.

---

## 📂 Project Directory Structure

The project root is cleanly isolated into distinct backend and frontend sub-modules:

```
traffic_analysis/               <-- Project Root
├── .gitignore
├── README.md
├── .venv/                      <-- Python Virtual Environment (Windows Native)
├── frontend/                   <-- Frontend Dashboard (Vite + React)
│   ├── src/
│   ├── package.json
│   └── vite.config.js
└── backend/                    <-- Backend Intelligence (FastAPI + YOLO + OCR)
    ├── server.py               <-- API Gateway and WS Stream Engine
    ├── prepare_data.py         <-- Dataset Extraction & splits engineer
    ├── train_models.py         <-- YOLOv8 custom model training & MLOps
    ├── config.py               <-- Shared detection parameters & limits
    ├── helmet_checker.py       <-- Two-wheeler & helmet detection engine
    ├── pedestrian_detector.py  <-- Pedestrian gender/child classifier
    ├── ocr_engine.py           <-- Multi-threaded EasyOCR ANPR engine
    ├── models/                 <-- YOLO weights (.pt) & Yaml tracker configs
    ├── datasets/               <-- Structured dataset training splits
    ├── videos/                 <-- Sample test videos
    └── runs/                   <-- YOLO checkpoints and logs
```

---

## 🚀 How to Run the Project (Windows Native)

Follow these steps to run both the FastAPI API and the React web dashboard.

### Step 1: Start the Python Backend (FastAPI)
1. Open a terminal (Git Bash, Command Prompt, or PowerShell) in the project root.
2. Navigate to the `backend` directory:
   ```cmd
   cd backend
   ```
3. Launch the API server using your virtual environment:
   ```cmd
   ..\.venv\Scripts\python server.py
   ```
   *The backend will initialize EasyOCR with CUDA acceleration on your RTX 3050 GPU and start listening on `http://localhost:8000`.*

### Step 2: Start the React Frontend (Vite)
1. Open a **second** terminal in the project root.
2. Navigate to the `frontend` directory:
   ```cmd
   cd frontend
   ```
3. Start the Vite development server:
   ```cmd
   npm run dev
   ```
   *The dashboard will compile and open instantly at **[http://localhost:3000](http://localhost:3000)**.*

---

## 🏍️ Two-Wheeler Safety & Compliance (New Feature!)

The system tracks all two-wheelers (motorcycles and scooters) and monitors helmet compliance for **both** the Rider and the Pillion:
* **Two-Wheeler Safety Log:** Real-time table rendering the Rider and Pillion safety badges (e.g. green for **Helmet**, red for **No Helmet**, gray for **No Pillion**).
* **ANPR Mapping:** Automatically links detected license plate codes with active safety violations for law enforcement logging.
* **Persistent Summary:** Cumulative compliance statistics are formatted and shown in the session finalized report when batch processing completes.

---

## 🚶 Pedestrian & Child Safety (Module 2)

Integrated pedestrian classifier tracks foot traffic cross-sections:
* **Gender Demographics:** Automatically classifies pedestrians as **Male** or **Female**.
* **Child Detection:** Detects child presence by calculating pedestrian height-to-bounding-box relationships relative to neighboring pedestrians.
* **Visual Dashboards:** Active counts are displayed on the main UI and cataloged inside the session summary.

---

## 📈 ML Training & Automation Pipeline

For training new custom detection models, utilize the automated pipeline scripts inside the `backend/` folder:

### 1. Data Engineering (`backend/prepare_data.py`)
Extract frames from videos and format dataset splits:
```bash
cd backend

# Extract frames for annotation
..\.venv\Scripts\python prepare_data.py --step extract_helmet
..\.venv\Scripts\python prepare_data.py --step extract_gender

# After manually annotating, split datasets 80/10/10 and build data.yaml
..\.venv\Scripts\python prepare_data.py --step setup_helmet
..\.venv\Scripts\python prepare_data.py --step setup_gender
```

### 2. Machine Learning Operations (`backend/train_models.py`)
Train and deploy YOLOv8 models on the GPU:
```bash
cd backend

# Run GPU-accelerated training
..\.venv\Scripts\python train_models.py --step train_helmet
..\.venv\Scripts\python train_models.py --step train_gender

# Evaluate validation metrics and print report
..\.venv\Scripts\python train_models.py --step evaluate_helmet
..\.venv\Scripts\python train_models.py --step evaluate_gender

# Deploy best models directly to models/
..\.venv\Scripts\python train_models.py --step deploy
```

---

## ⚡ GPU Hardware Acceleration
Check your hardware acceleration status by running this command from the project root:
```cmd
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```
*(Should print `True` for your NVIDIA GeForce RTX 3050 Laptop GPU).*
