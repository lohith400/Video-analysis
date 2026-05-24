# Indian Road Intelligence System

Real-time traffic video analysis, Automatic Number Plate Recognition (ANPR), and safety compliance monitoring system. Features custom-trained **YOLOv8** detectors connected to a beautiful **React (Vite) dashboard**.

This repository is optimized to run **natively on Windows** with **GPU acceleration** powered by your NVIDIA GeForce RTX 3050 Laptop GPU.

---

## 🚀 How to Run the Project (Windows Native)

Follow these steps to run the FastAPI API and the React web dashboard.

### Step 1: Start the Python Backend (FastAPI)
1. Open a terminal (Git Bash, Command Prompt, or PowerShell) in the project root.
2. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
3. Launch the API server using your virtual environment:
   * **In PowerShell or CMD**:
     ```cmd
     ..\.venv\Scripts\python server.py
     ```
   * **In Git Bash**:
     ```bash
     ../.venv/Scripts/python server.py
     ```
   *The backend will initialize EasyOCR with CUDA acceleration on your RTX 3050 GPU and start listening on `http://localhost:8000`.*

### Step 2: Start the React Frontend (Vite)
1. Open a **second** terminal in the project root.
2. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The dashboard will compile and open instantly at **[http://localhost:3000](http://localhost:3000)**.*

---

## 🌳 Complete Visual Project Tree & File Directory

Here is the complete nested directory structure of the repository showing **each and every file and its exact purpose**:

```
traffic_analysis/                              # Project Root Directory
├── .gitignore                                 # Git exclusions (ignores .venv, node_modules, etc.)
├── README.md                                  # You are here - project roadmap & start guide
├── .venv/                                     # Python Virtual Environment (Windows Native modules)
│
├── frontend/                                  # React (Vite) Frontend Sub-Project Directory
│   ├── .npmrc                                 # Configures npm to use cmd.exe on Windows to prevent Bash execution block
│   ├── index.html                             # Single-page application root entry HTML document
│   ├── package.json                           # NPM dependencies, scripts (dev, build, lint) & project metadata
│   ├── postcss.config.js                      # PostCSS optimizer configuration utility
│   ├── tailwind.config.js                     # Tailwind theme limits, colors, fonts, and animation transitions config
│   ├── vite.config.js                         # Vite dev server and proxy configuration
│   └── src/                                   # Frontend Source Directory
│       ├── main.jsx                           # Application bootstrap mounting script
│       ├── App.jsx                            # Shared routing and page layout coordinator
│       ├── index.css                          # Custom global styling, daylight scrollbars & glassmorphism classes
│       │
│       ├── components/                        # UI Functional Widgets
│       │   ├── AlertBanner.jsx                # Flashes red warning pulses instantly on severe helmet violations
│       │   ├── Charts.jsx                     # Renders traffic count and safety analytics graphs
│       │   ├── Footer.jsx                     # Simple, elegant copyright footer component
│       │   ├── MetricCard.jsx                 # Displays live stats (FPS, duration, processed frames)
│       │   ├── Navbar.jsx                     # Dynamic blur-effect floating top navigation bar
│       │   ├── PlateTable.jsx                 # ANPR log grid showing scrolling plates & confidence levels
│       │   ├── TwoWheelerSafetyTable.jsx      # Rider & Pillion safety log rendering green/red/gray compliance badges
│       │   ├── VehicleCounts.jsx              # Category boxes displaying vehicle totals with custom emojis
│       │   ├── VideoFeed.jsx                  # Video stream player with canvassing and status overlays
│       │   └── ViolationList.jsx              # Sidebar listing categorized compliance violations
│       │
│       └── pages/                             # Core Screen Dashboards
│           ├── Home.jsx                       # Landing page hosting video upload dropzones and line configuration
│           ├── LiveAnalysis.jsx               # Active command center orchestrating WebSocket streams and data tables
│           └── Analytics.jsx                  # Intelligence terminal parsing graphs from historical CSV logs
│
└── backend/                                   # FastAPI + YOLOv8 + EasyOCR Backend Sub-Project Directory
    ├── requirements.txt                       # Backend pip dependency package listings
    ├── traffic_log.csv                        # Main database logging vehicle crossings, ANPR reads & timestamps
    ├── train_error.log                        # Technical compiler logs tracking GPU hardware errors
    ├── main.txt                               # Legacy textual roadmap outlining development history
    ├── server.py                              # FastAPI entry gateway that streams video frames & coordinates WS payloads
    ├── config.py                              # Pipeline parameters, checkpoints, and thresholds configuration
    ├── helmet_checker.py                      # Rider/Pillion helmet detector & compliance logging engine
    ├── pedestrian_detector.py                 # Crosswalk pedestrian classifier mapping Gender/Child demographics
    ├── ocr_engine.py                          # Async multi-threaded OCR engine reading plate characters
    ├── tracker.py                             # Localized ByteTrack multi-object persistent tracker
    ├── traffic_counter.py                     # Centroid crossings line manager that logs counts
    ├── detector.py                            # Wrapper for batch box inferences
    ├── annotator.py                           # Frame drawing manager drawing boundary boxes, lists & line coordinates
    ├── prepare_data.py                        # Frames extractor & automated dataset directories splitter
    ├── train_models.py                        # MLOps pipeline automating GPU YOLO trainings & weights deployments
    ├── auto_annotate.py                       # Pre-annotation tool using custom weights to bootstrap labeling
    ├── extract_frames.py                      # Frame sampler utility slicing media files into image directories
    ├── sanitize_dataset.py                    # Annotation files sanitation script
    ├── download_models.py                     # Download utility pulling base YOLO model parameters
    ├── verify_pipeline.py                     # Diagnostic validation script checking CUDA and hardware bindings
    ├── verify_pipeline_debug.py               # Diagnostic debugging checklist trace logger
    ├── test_model.py                          # Stand-alone detection validation tool
    ├── test_traffic_counter.py                # Diagnostic tracking line crossed auditor
    ├── evaluate_model.py                      # Basic metrics validation checks
    ├── train_plate_model.py                   # Legacy license plate training engine
    ├── yolo26n.pt                             # Custom-compiled vision network parameters
    ├── yolov8n.pt                             # Default pretrained vehicle network parameters
    │
    ├── models/                                # Production Model Weights Directory
    │   ├── custom_bytetrack.yaml              # Multi-object ByteTrack tracking config file
    │   ├── gender_detector.pt                 # Custom pedestrian model classifying demographics
    │   ├── helmet_detector.pt                 # Custom safety model classifying helmet compliance
    │   └── license_plate_detector.pt          # Custom high-precision license plate detector
    │
    ├── dataset/                               # Dataset engineering repository (legacy)
    ├── gender_dataset/                        # Pedestrian dataset storage folder
    ├── gender_dataset_train/                  # GPU Pedestrian training split outputs
    ├── helmet_dataset/                        # Helmet dataset storage folder
    ├── helmet_dataset_train/                  # GPU Helmet training split outputs
    ├── plate_dataset/                         # Custom License plate dataset storage folder
    ├── runs/                                  # YOLO checkpoints, logs & metrics reports
    ├── scratch/                               # Temporary testing play sandbox scripts
    ├── uploads/                               # Gateway temporary video storage folder
    └── videos/                                # Sample traffic test streams
```

---

## ⚡ GPU Hardware Acceleration
Check your hardware acceleration status by running this command from the project root:
```cmd
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
```
*(Should print `True` for your NVIDIA GeForce RTX 3050 Laptop GPU).*
