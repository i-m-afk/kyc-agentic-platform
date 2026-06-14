# Advanced KYC Security: Forgery & Deepfake Detection Guide

This document details the checks, tells, programmatic algorithms, and open-source models required to build a resilient, commercial-grade identity verification pipeline.

---

## 1. Human Document Verification Checks

In manual KYC audits, compliance officers inspect physical identity documents for specific material and layout anomalies:

### A. Optical & Print Security Features
*   **Guilloche Patterns**: High-resolution, interwoven geometric curves that form the document background. Commercial offset printers render these as continuous vector lines. Home inkjet/laser printers lack the resolution, printing them as dotted, pixelated, or solid-color blocks.
*   **Microprinting**: Extremely small text (often less than 0.3mm tall) used in borders or signature lines. Under magnification, it is clean and legible. Counterfeits print this as blurry, solid, or broken lines.
*   **Optically Variable Ink (OVI) & Holograms**: Security seals and logos that shift in color (e.g., from gold to green) or display varying details when tilted. Static photocopies or screen displays fail to reproduce these angle-dependent color shifts.
*   **Rainbow Printing**: Subtle gradients where colors transition smoothly without halftone dots. Scans or counterfeits display distinct printing dots (halftones) under magnification.

### B. Structural & Data Consistency
*   **Font and Kerning Alignment**: Counterfeits often use automated text insertion, resulting in inconsistent kerning, varying character sizes, or mismatched typefaces.
*   **Lamination & Surface Integrity**: Scratches, cuts, or air bubbles around the portrait photo indicate a physical "photo substitution" attack.
*   **Logic Matches**: The ID card number syntax (e.g., specific prefixes for states/provinces) must match the issuing authority's database rules and the applicant's Date of Birth (DOB).

---

## 2. Deepfake Video Verification Tells (Human Observation)

When reviewing liveness videos or real-time calls, human reviewers look for visual glitches where generative models fail:

*   **Occlusion Boundaries**: The core vulnerability behind the "Three-Finger Test." Real-time AI face overlays (deepfakes) map a face onto a 3D mesh. When a physical object (like a hand, finger, or piece of paper) passes between the camera and the face, the mesh fails to calculate the boundary mask. The face overlay will glitch, appear *on top* of the fingers, or clip erratically.
*   **Profile Angles (Pose Deviations)**: Real-time face-swap models rely on key facial landmarks (eyes, eyebrows, nose bridge, mouth corners). When a user turns their head 90 degrees sideways, these landmarks disappear from view. The deepfake model loses tracking, causing the face texture to warp, lag, or look flat.
*   **Lighting and Specular Mismatch**: The artificial face overlay often has its own static lighting orientation. If a user moves their head relative to a lamp, or waves a hand across the light source, the shadows cast on the face will look unnatural or static.
*   **Physiological Inconsistencies**: Absence of micro-blinks, lack of synchronized pupil dilation under changing screen brightness, or frozen neck muscles while the face is actively speaking.

---

## 3. Programmatic & Computer Vision Detection Techniques

To automate deepfake detection, we implement mathematical algorithms that detect details invisible to the human eye:

### A. Frequency Domain Analysis (Fast Fourier Transform - FFT)
Generative models (GANs, Diffusion Models) use upsampling layers to generate images. This leaves subtle, periodic grid-like artifacts in the high-frequency spectrum of the pixels.
*   **Method**: Compute the 2D Discrete Fourier Transform (2D-DFT) of the face image:
    $$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
*   **Detection**: Plotting the power spectrum reveals periodic high-frequency spikes (grid patterns) in AI-generated images that are completely absent in real photos.

### B. Remote Photoplethysmography (rPPG - Blood Flow Tracking)
Human skin changes color microscopically with each heartbeat. By tracking the green color channel of facial pixels over time, we can extract the user's pulse wave.
*   **Method**: Track average skin pixel values in regions of interest (cheeks, forehead) across video frames, apply bandpass filtering (0.75 - 3.0 Hz), and compute the heart rate.
*   **Detection**: Real people show a clear, periodic cardiac pulse wave. Deepfakes show a completely flat, chaotic, or non-harmonic signal.

### C. Optical Flow Consistency
Checks for spatial and temporal alignment between frames.
*   **Method**: Calculate motion vectors between consecutive video frames using Dense Optical Flow (e.g., Farneback algorithm in OpenCV).
*   **Detection**: If there is a deepfake face overlay, the motion vectors of the face pixels will display micro-jitters, lag, or misalignment compared to the motion vectors of the neck and background.

---

## 4. Open-Source Models & Frameworks

To implement these checks programmatically, developers utilize these open-source models:

*   **Face X-Ray (Blended Boundary Detection)**:
    *   *Concept*: Focuses on detecting the boundary line created when blending a fake face into a real background image.
    *   *Advantage*: Model-agnostic; it does not need to be trained on the specific AI model used to generate the deepfake.
*   **MesoNet (Meso-4 / MesoInception-4)**:
    *   *Concept*: A compact CNN designed to identify microscopic compression artifacts and eye/mouth rendering inconsistencies.
    *   *Advantage*: Extremely lightweight, low latency, and highly optimized for real-time video stream scanning.
*   **Capsule Networks (Capsule-Forensics)**:
    *   *Concept*: Uses capsules rather than standard CNN neurons to capture the spatial relationship and relative orientation of facial features (e.g., checking if the eyes, nose, and mouth are warped or misaligned).
*   **Reality Defender / OpenDFD Frameworks**:
    *   Pre-built detection benchmarks trained on standard datasets like **FaceForensics++**, **DFDC (DeepFake Detection Challenge)**, and **Celeb-DF**.
