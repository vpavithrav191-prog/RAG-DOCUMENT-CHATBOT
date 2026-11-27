🤖 RAG Document Chatbot

An AI-powered document question-answering chatbot built using Retrieval-Augmented Generation (RAG). It allows users to upload documents and ask questions based on their content.

🌐 Live Demo

"Open RAG Document Chatbot" (https://vpavithrav191-prog-rag-document-chatbot-app-ugseef.streamlit.app/)

📌 Project Overview

The RAG Document Chatbot helps users understand and retrieve information from uploaded documents.

Instead of manually searching through lengthy documents, users can upload a file and ask questions in natural language. The application retrieves relevant information and uses a Large Language Model (LLM) to generate answers.

✨ Features

- Upload documents and ask questions.
- Supports PDF, DOCX, TXT, Markdown and CSV files.
- Extracts and splits document content into chunks.
- Generates embeddings for document content.
- Uses FAISS for vector storage and similarity search.
- Uses LangChain for RAG workflow integration.
- Uses OpenRouter to access LLMs.
- Provides answers based on retrieved document content.
- Simple and interactive Streamlit interface.

🛠️ Technologies Used

Technology| Purpose
Python| Programming language
Streamlit| User interface
LangChain| RAG workflow
FAISS| Vector database
OpenRouter| LLM API access
Embeddings| Text representation
PyPDF| PDF processing
Docx2txt| Word document processing

⚙️ How It Works

1. User uploads a document.
2. Text is extracted from the document.
3. Extracted text is divided into smaller chunks.
4. Embeddings are generated for the chunks.
5. FAISS stores the vectors.
6. User asks a question.
7. Relevant chunks are retrieved using similarity search.
8. The LLM generates an answer using the retrieved context.

🏗️ RAG Architecture

Document Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Store → Similarity Search → LLM → Answer

💻 Run Locally

1. Clone the repository

git clone https://github.com/vpavithrav191-prog/RAG-DOCUMENT-CHATBOT.git

2. Open the project folder

cd RAG-DOCUMENT-CHATBOT

3. Install dependencies

pip install -r requirements.txt

4. Configure your OpenRouter API key

Create a ".env" file in the project folder:

OPENROUTER_API_KEY=your_api_key_here

Never upload your ".env" file or API key to GitHub.

5. Run the application

streamlit run app.py

☁️ Deployment

The application is deployed using Streamlit Community Cloud.

Live application:
https://vpavithrav191-prog-rag-document-chatbot-app-ugseef.streamlit.app/

🎯 Project Objective

To develop an intelligent document-based question-answering system that helps users retrieve information from documents quickly and efficiently using Generative AI and Retrieval-Augmented Generation.

🔮 Future Enhancements

- Support for additional document formats.
- Improved answer accuracy and retrieval.
- Document summarization.
- Conversation history.
- Multilingual question answering.

👩‍💻 Developed By

V Pavithra

GitHub: https://github.com/vpavithrav191-prog

---

⭐ If you find this project useful, feel free to explore the repository!
