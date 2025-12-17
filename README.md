# 🌿 Agri-Crop Pro

Agri-Crop Pro is an AI-powered agricultural assistant designed to help farmers and agronomists detect plant diseases and receive treatment recommendations instantly.

Using a combination of **Computer Vision** and **Retrieval-Augmented Generation (RAG)**, the application analyzes images of crops, identifies pathologies, and retrieves relevant information from agricultural manuals to provide accurate diagnoses and actionable advice.

## ✨ Features

- **Plant Identification**: Automatically detects the plant species from an uploaded image.
- **Disease Detection**: Identifies signs of disease, pests, or nutrient deficiencies.
- **Smart Diagnosis**: Uses RAG to consult a knowledge base of agricultural manuals for verified information.
- **Treatment Plans**: Provides chemical and organic treatment options along with prevention tips.
- **Interactive Chat**: Ask follow-up questions to the AI Agronomist.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **AI Models**: [Ollama](https://ollama.com/) (LLaVA / Llama 3.2 Vision)
- **Vector DB**: ChromaDB
- **Embeddings**: FastEmbed & Sentence Transformers
- **Orchestration**: LangChain

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.8+** installed.
2.  **Ollama** installed and running. You can download it at [ollama.com](https://ollama.com).
3.  Pull the required vision model (default is `llava`):
    ```bash
    ollama pull llava
    ```

### Installation

1.  Clone the repository (if applicable) or navigate to the project directory.
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Prepare the Knowledge Base** (First run only):
    If you have agricultural manuals or data in the `datasets` or `data` folder, run the ingestion script to populate the vector database:
    ```bash
    python ingest.py
    ```

2.  **Start the Application**:
    ```bash
    streamlit run app.py
    ```

3.  **Analyze**:
    - Open your browser to the provided URL (usually `http://localhost:8501`).
    - Upload an image of a plant/leaf.
    - Wait for the **Neural Vision Processing** and **RAG Search** to complete.
    - View the diagnosis and chat with the assistant!

## 📂 Project Structure

- `app.py`: Main Streamlit application.
- `ingest.py`: Script to ingest data into ChromaDB.
- `vector_db/`: Persisted vector database (Git ignored).
- `datasets/` & `data/`: Raw data storage (Git ignored).
- `requirements.txt`: Python package dependencies.

## ⚠️ Note

- The application defaults to using `llava`. You can change the `VISION_MODEL` variable in `app.py` if you prefer `llama3.2-vision` or other supported models.
- Ensure Ollama is running in the background (`ollama serve`) for the app to function.
