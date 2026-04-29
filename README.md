# Fine-Tune Gemma 3 270M with Keras & JAX for Medical Q&A

This project fine-tunes the **Gemma 3 270M** language model using **LoRA**, **Keras**, and a **JAX** backend on a [Medical Q&A dataset](https://www.kaggle.com/datasets/gpreda/medquad/) from Kaggle. It is designed to run on a **Google Cloud Vertex AI Workbench** instance with GPU support.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Setting Up a GCP Workbench Instance](#setting-up-a-gcp-workbench-instance)
4. [Configuring Kaggle API Credentials](#configuring-kaggle-api-credentials)
5. [Installing the Kaggle CLI](#installing-the-kaggle-cli)
6. [Downloading the Dataset](#downloading-the-dataset)
7. [Running the Notebook](#running-the-notebook)

---

## Project Overview

The notebook `finetune-gemma-3-270m-using-lora-for-medical-q-a-gcp-run.ipynb`:

- Loads the **Gemma 3 270M** model via `keras_nlp`
- Applies **LoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Uses the **JAX** backend for accelerated training on GPU/TPU
- Trains on a **Medical Q&A** CSV dataset (`medquad.csv`) from Kaggle
- Demonstrates inference before and after fine-tuning

---

## Prerequisites

- A Google Cloud Platform (GCP) account with billing enabled
- Project permissions to create Vertex AI Workbench instances
- A [Kaggle](https://www.kaggle.com) account

---

## Setting Up a GCP Workbench Instance

1. Go to the [Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench) page in the GCP Console.
2. Click **Create New** and choose **Instance**.
3. Configure the instance:
   - Select a **GPU machine type** (e.g., `n1-standard-8` with an NVIDIA T4 or A100 GPU) for efficient training.
   - Choose a framework image that includes **JupyterLab** (e.g., *PyTorch/JAX* or *TensorFlow Enterprise*).
4. Click **Create** and wait for the instance to start.
5. Once running, click **Open JupyterLab** to access the environment.
6. Upload the notebook file `finetune-gemma-3-270m-using-lora-for-medical-q-a-gcp-run.ipynb` via the JupyterLab file browser, or clone this repository directly in a terminal:

   ```bash
   git clone https://github.com/<your-username>/fine-tune-gemma3-keras-jax-medical-q-a.git
   ```

---

## Configuring Kaggle API Credentials

To download datasets from Kaggle programmatically, you need a **Kaggle API token**.

### Step 1 — Generate a Kaggle API Token

1. Log in to [kaggle.com](https://www.kaggle.com).
2. Click on your profile avatar (top-right) → **Settings**.
3. Scroll to the **API** section and click **Create New Token**.
4. A file named `kaggle.json` will be downloaded to your computer. It contains your credentials in this format:

   ```json
   {
     "username": "your_kaggle_username",
     "key": "your_kaggle_api_key"
   }
   ```

### Step 2 — Copy the Token to the GCP Instance

You need to place `kaggle.json` in the `~/.kaggle/` directory on the Workbench instance.

**Option A — Copy via JupyterLab Upload + Terminal**

1. In JupyterLab, use the file browser to upload `kaggle.json` to your home directory.
2. Open a **Terminal** tab and run:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Option B — Paste content directly in the terminal**

1. Open a **Terminal** tab in JupyterLab.
2. Create the `.kaggle` directory and the file:

   ```bash
   mkdir -p ~/.kaggle
   cat > ~/.kaggle/kaggle.json << 'EOF'
   {"username":"your_kaggle_username","key":"your_kaggle_api_key"}
   EOF
   chmod 600 ~/.kaggle/kaggle.json
   ```

   Replace `your_kaggle_username` and `your_kaggle_api_key` with the actual values from your downloaded `kaggle.json`.

> **Security note:** The `chmod 600` command restricts the file so only your user can read it, which is required by the Kaggle CLI.

---

## Installing the Kaggle CLI

In a JupyterLab terminal on the Workbench instance, install the Kaggle CLI using pip:

```bash
pip install --upgrade kaggle
```

Verify the installation:

```bash
kaggle --version
```

---

## Downloading the Dataset

The notebook uses the **MedQuAD** dataset hosted on Kaggle at:  
👉 [`https://www.kaggle.com/datasets/gpreda/medquad`](https://www.kaggle.com/datasets/gpreda/medquad)

To download it locally on the GCP instance, open a terminal and run:

```bash
# Navigate to the project directory (or wherever you want the data)
cd ~/fine-tune-gemma3-keras-jax-medical-q-a

# Download and unzip the dataset
kaggle datasets download -d gpreda/medquad --unzip
```

This will place `medquad.csv` in the current directory. The notebook expects this file to be present in the **same directory** from which the notebook is opened (i.e., the working directory at runtime).

---

## Running the Notebook

1. In JupyterLab, open `finetune-gemma-3-270m-using-lora-for-medical-q-a-gcp-run.ipynb`.
2. **Run the first two cells** to install/update the required packages:
   - `keras-nlp`, `keras`
   - `jax[cuda12]`
   - `kagglehub`
3. **Restart the kernel** when prompted (there is a prominent reminder cell in the notebook).
4. Continue running all remaining cells from top to bottom.

> **Tip:** The JAX backend is pre-configured in the notebook via environment variables. Make sure not to modify the `KERAS_BACKEND` or `XLA_PYTHON_CLIENT_MEM_FRACTION` settings unless you intentionally want to switch backends.
