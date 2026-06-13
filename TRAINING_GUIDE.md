# Training & Execution Guide: Face Liveness Model on AMD ROCm Server

This guide explains how to acquire the dataset, set up the environment, and train the binary face liveness model on your remote GPU-enabled ROCm server.

---

## 1. Dataset Selection & Acquisition

To train an effective Face Anti-Spoofing (liveness detection) model, you need a dataset consisting of:
1. **Real/Genuine Faces**: Videos or images of actual people looking at the camera.
2. **Spoof/Attack Attempts**: Videos or images of print attacks (printed photo of face), replay attacks (video playbacks on screen), or deepfakes.

### Recommended Public Datasets
- **NUAA Imposter Database**: Contains real faces and printed photos of the same people. Excellent for simple, fast training.
  - [NUAA Website](http://parnec.nuaa.edu.cn/xtan/data/ImposterDB.html)
- **FAS-SD (Face Anti-Spoofing System Database)**: A general database containing screen replays and print attacks.
- **CASIA-FASD / Replay-Attack**: Standard academic benchmarks for face anti-spoofing.

### Expected Folder Structure
The training script and notebook expect the dataset to be organized in the following standard `ImageFolder` structure:
```text
dataset/
├── train/
│   ├── real/
│   │   ├── face_001.jpg
│   │   └── ...
│   └── spoof/
│       ├── spoof_001.jpg
│       └── ...
└── val/
    ├── real/
    │   ├── face_val_001.jpg
    │   └── ...
    └── spoof/
        ├── spoof_val_001.jpg
        └── ...
```

---

## 2. Environment Setup (Remote ROCm Server)

On your remote GPU/ROCm machine, set up the dependencies:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with ROCm acceleration (from the official AMD package source)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# 3. Install other requirements
pip install opencv-python-headless pillow jinja2 jupyterlab
```

---

## 3. Running a Synthetic Test Run (Zero-Data Verification)

Before downloading gigabytes of real face data, you can generate a small synthetic dataset of colored noise to verify that the training script runs end-to-end on the GPU.

1. **Generate Synthetic Data**:
   Create a test script `src/training/generate_mock_data.py` (see implementation) and run it:
   ```bash
   python src/training/generate_mock_data.py --output_dir ./mock_dataset
   ```
2. **Verify Training**:
   Run the training script with the mock dataset:
   ```bash
   python src/training/train_liveness.py --data_dir ./mock_dataset --epochs 2 --batch_size 8 --save_path notebooks/liveness_model.pt
   ```

---

## 4. Full Model Training Flow

Once your real dataset is loaded into the ROCm server, you can trigger training in two ways:

### Method A: Jupyter Lab Notebook (Interactive)
1. Open your Jupyter Lab workspace:
   `https://notebooks.amd.com/jupyter-hack-team-.../lab`
2. Open the notebook `notebooks/train_liveness_model.ipynb`.
3. Set your dataset paths.
4. Execute cells sequentially. It will:
   - Configure data loaders with random horizontal flips and rotations.
   - Load pre-trained Small MobileNetV3 and adapt the final output layer for binary classification.
   - Run optimization loops saving the best weights to `liveness_model.pt`.

### Method B: Terminal Script Execution
Run the training script directly from the shell:
```bash
python src/training/train_liveness.py \
    --data_dir /path/to/your/acquired/dataset \
    --epochs 25 \
    --batch_size 32 \
    --lr 0.001 \
    --save_path notebooks/liveness_model.pt
```

---

## 5. Exposing the Streamlit Dashboard from the ROCm Server

If you want to run the full GPU pipeline on your ROCm server and access it locally:

1. **Start the Dashboard**:
   ```bash
   # Make sure MOCK_ML is disabled so actual PyTorch/vLLM are used
   export MOCK_ML=false
   streamlit run src/app.py --server.port 8501
   ```
2. **Access via Port Forwarding**:
   On your local machine, forward the port over SSH:
   ```bash
   ssh -L 8501:localhost:8501 user@your-rocm-server-ip
   ```
3. Open `http://localhost:8501` in your local browser to access the dashboard.
