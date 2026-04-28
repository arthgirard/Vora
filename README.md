# Vora

Vora is an AI-powered ocular health assistant designed to monitor eye blink frequency in real-time. By utilizing computer vision, the application tracks user habits to help prevent digital eye strain and optimize long-term visual well-being.

## Features

* **Real-Time Blink Monitoring**: High-precision detection using MediaPipe Face Landmarker to track eye aspect ratios.
* **Live Session Analytics**: Real-time graphing of blink frequency per minute using Matplotlib.
* **Historical Benchmarking**: Comparison of current session data against historical averages stored in a local database.
* **Fatigue Alerts**: Identification of low-frequency blinking intervals that may indicate ocular fatigue.
* **Customizable Sensitivity**: Adjustable alert thresholds between 10 and 15 blinks per minute to suit individual needs.
* **Modern Interface**: A responsive GUI built with Flet, featuring seamless light and dark mode support.

## Technical Stack

* **Frontend**: Flet (Python-based framework).
* **Computer Vision**: MediaPipe.
* **Data Visualization**: Matplotlib and Pandas.
* **Database**: SQLite for persistent session logging.
* **Language**: Python 3.x.

## Project Structure

* `backend/`: Core logic for blink detection, database management, and tracking.
* `frontend/`: GUI implementation and MediaPipe task configurations.
* `assets/`: UI resources including branding logos and audio files.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arthgirard/vora.git
   cd vora
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `face_landmarker.task` is present in the `frontend/` directory.

## Usage

Run the application from the project root:

```bash
python frontend/gui.py
```

* **Démarrage**: Click "Lancer" to begin monitoring via webcam.
* **Analyse**: View real-time graphs and fatigue alerts based on session data.
* **Paramètres**: Toggle Dark Mode or adjust blink sensitivity thresholds.
