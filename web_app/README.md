# Mental Health Screening Web Portal

## Setup Instructions

### Quick Start
1. **Run the setup script:**
   ```bash
   setup_web_portal.bat
   ```

### Manual Setup
If the setup script doesn't work, follow these steps:

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare models:**
   ```bash
   mkdir models
   # Copy your ensemble model files here:
   # - ensemble_1.h5
   # - ensemble_2.h5
   # - ensemble_3.h5
   ```

4. **Start the web portal:**
   ```bash
   python app.py
   ```

### Model Files Required
The web portal expects these model files in the `models/` directory:
- `ensemble_1.h5` - Model 1 (weight: 1.0x)
- `ensemble_2.h5` - Model 2 (weight: 1.0x)  
- `ensemble_3.h5` - Model 3 (weight: 1.5x)

### Access the Portal
Once running, access the web portal at:
- **Local**: http://localhost:8000
- **Network**: http://YOUR_IP:8000

### Features
- **AI-Driven Severity Screening** using ensemble voting
- **Real-time Analysis** with webcam integration
- **Professional Dashboard** with Chart.js visualizations
- **Dynamic Thresholding** for High classification
- **Confidence-Weighted Voting** for improved accuracy

### API Endpoints
- `GET /` - Main web interface
- `POST /predict` - Make predictions with (300, 17) features
- `GET /health` - System health check

### Troubleshooting

#### TensorFlow Issues
If you encounter TensorFlow errors, try:
```bash
pip install tensorflow-cpu==2.16.1
```

#### Model Loading Issues
- Ensure model files are in `models/` directory
- Check file names match exactly: `ensemble_1.h5`, etc.
- Verify models are saved in Keras format (.h5)

#### Port Issues
- If port 8000 is occupied, modify `app.py`:
  ```python
  uvicorn.run(app, host="0.0.0.0", port=8001)  # Change port
  ```

### Medical Disclaimer
This AI-driven screening tool is for research and educational purposes only. It is not a substitute for professional medical diagnosis, treatment, or advice. Always consult qualified healthcare professionals for mental health concerns.
