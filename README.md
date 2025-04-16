# RAG-powered Chatbot Application for Large PDF Documents

This is a Streamlit-based Generative AI Chatbot Application using Retrieval-Augmented Generation (RAG) architecture that can ingest, summarize, and interactively answer questions from large PDF documents (230+ pages).

## Features

- **Large PDF Support**: Handles PDFs with 230+ pages through efficient chunking
- **Instant PDF Summary**: Generates a concise summary with 5-10 key points upon document upload
- **Auto-Generated Suggestive Questions**: Provides 5-10 relevant questions to help users get started
- **Interactive Chatbot**: Allows users to ask questions about the document in a chat interface
- **Vector Database Integration**: Uses FAISS to store and retrieve document embeddings efficiently
- **Advanced Embeddings & LLM**: Uses Hugging Face embeddings and Groq's Llama 3 model
- **User-Friendly UI**: Clean Streamlit interface with separate summary, suggestions, and chat sections
- **Chat History Download**: Allows users to download their conversation history

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd rag-pdf-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your environment variables
Create a `.env` file in the root directory with the following content:
```
GROQ_API_KEY=your_groq_api_key_here
```

You can get a Groq API key by signing up at https://console.groq.com/

### 5. Run the application
```bash
streamlit run app.py
```

## Usage

1. Upload a PDF document using the uploader in the sidebar
2. Wait for the document to be processed (this may take some time for large documents)
3. View the document summary and suggested questions in the "Document Summary" tab
4. Switch to the "Chat" tab to ask your own questions about the document
5. Download your chat history using the button in the sidebar

## Architecture

- **PDF Processing**: The application uses PyPDFLoader to extract text from PDFs
- **Text Splitting**: Documents are split into manageable chunks using RecursiveCharacterTextSplitter
- **Embeddings**: Uses Hugging Face's sentence-transformers/all-MiniLM-L6-v2 model for embeddings
- **Vector Storage**: FAISS is used as the vector database for efficient similarity search
- **Language Model**: Integrates with Groq's Llama 3 (70B) for high-quality responses
- **User Interface**: Built with Streamlit for an intuitive and responsive experience

## Requirements

- Python 3.8+
- Internet connection (for API calls to Groq)
- Groq API key
- Sufficient RAM (at least 8GB recommended for large PDFs)

## Limitations

- Processing very large PDFs (500+ pages) may take significant time
- API rate limits may apply depending on your Groq account tier
- Summaries are generated based on the first few chunks of the document

## Troubleshooting

- If you encounter memory issues, try processing smaller PDFs or reduce the chunk size
- Make sure your API key is correctly set in the .env file
- Check your internet connection if API calls are failing
