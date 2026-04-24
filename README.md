**Confusion-Guided Knowledge Distillation for Remaining Useful Life (RUL) Prediction**

---

## Project Overview

This project focuses on predicting the **Remaining Useful Life (RUL)** of machinery using deep learning, enhanced by **knowledge distillation (KD)**.

### Core Idea

The system uses a **teacher–student framework**:

* A **teacher model** (larger and more accurate) is trained on the dataset
* A **student model** (lighter and faster) learns from both:

  * Ground truth labels
  * The teacher’s outputs

The key contribution is **confusion-guided distillation**, where the student model also learns from the teacher’s uncertainty patterns, improving its ability to generalize.

### Objective

* Maintain high prediction accuracy
* Reduce model complexity
* Enable efficient deployment in real-world systems

---

## How It Works

1. **Data Preparation**

   * Load and preprocess time-series sensor data
   * Normalize features and generate RUL labels

2. **Teacher Training**

   * Train a high-capacity model on the dataset

3. **Student Training with Distillation**

   * Train a smaller model using:

     * True labels
     * Teacher predictions
     * Confusion-based guidance

4. **Evaluation**

   * Measure performance using metrics like RMSE

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd confusion-guided-kd-rul
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

* Windows:

```bash
venv\Scripts\activate
```

* macOS/Linux:

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Prepare the Dataset

* Download the dataset (e.g., NASA CMAPSS)
* Place it inside a `data/` directory in the project root

Example:

```
data/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

---

### 5. Train the Model

```bash
python train.py
```

---

### 6. Evaluate the Model

```bash
python test.py
```

---

If you want a version that exactly mirrors your actual file names and scripts (instead of this generalized structure), I can refine it precisely to your repo.
