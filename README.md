# EnergyLLM-RAG

A Retrieval-Augmented Generation (RAG) pipeline using MatSciBERT, Pinecone, and Gemini 2.0 Flash for battery materials research.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/MarreddyMohitSasankReddy/EnergyLLM-RAG.git
   cd EnergyLLM-RAG
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up secrets:**
   - Create `.streamlit/secrets.toml`
   - Add your API keys:
     ```toml
     [secrets]
     GOOGLE_API_KEY = "your-google-api-key"
     PINECONE_API_KEY = "your-pinecone-api-key"
     PINECONE_ENV = "us-east-1"
     ```

4. **Run the application:**
   ```sh
   streamlit run main.py
   ```

## Features

- MatSciBERT-based query embedding
- Pinecone vector search for document retrieval
- Gemini 2.0 Flash for response generation
- Hallucination detection with cosine similarity

## File Structure

```
EnergyLLM-RAG/
│── .streamlit/
│   └── secrets.toml
│── main.py
│── requirements.txt
│── README.md
│── .gitignore
```

## License
This project is licensed under the MIT License.

