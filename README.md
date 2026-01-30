# HCI_Hackathon
# AI OncoTree Classification Assistant

## Overview
This project is a lightweight **Streamlit web application** that uses a **locally running LLM (via Ollama)** to assist with:

1. Predicting the **OncoTree tissue**
2. Predicting the **OncoTree name**
3. Mapping the result to an **OncoTree code**

The app is designed to be **local-only** (no external APIs) and includes **human-in-the-loop validation**, allowing users to override model predictions when needed.

All required data files are already compiled and located correctly in the repository.  

---

## Requirements
You will need:

- **Python 3.9 or newer**
- **Ollama** installed and running locally
- At least one **local Ollama model**
- Python packages:
  - `streamlit`
  - `ollama`

---

## Step 1 — Download the repository
Clone the GitHub repository:

```bash
git clone [<GITHUB_REPO_URL>](https://github.com/GabrielaFort/HCI_Hackathon.git)
cd HCI_Hackathon/src
```

## Step 2 - Install python dependancies
pip install streamlit ollama

## Step 3 - Install and set up Ollama
1) Install Ollama (follow the official Ollama instructions for your OS)
2) Start Ollama
3) Pull at least one local model, for example: ```ollama pull granite4:latest```
4) Verify that models are available: ```ollama list```
* The app will automatically detect available local models and show them in the sidebar.

## Step 4 - Run the streamlit app
1) Ensure you are in the src directory (where the app.py script is)
2) Run ```streamlit run app.py``` from the command line
3) This should automatically opon a local browser running the app

