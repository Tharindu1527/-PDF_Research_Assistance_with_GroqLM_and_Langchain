# PDF Research Assistant with GroqLM and LangChain

A sophisticated chatbot designed to analyze and answer questions about PDF documents using GroqLM, LangChain, and various cutting-edge NLP components. This assistant leverages vector databases for efficient document retrieval and maintains conversation memory for contextual responses.

## Live Demo

Try out the live demo on Hugging Face Spaces:
[PDF Research Assistant](https://huggingface.co/spaces/Tharindu1527/PDF_Research_Assistance_with_GroqLM_and_Langchain)

## Features

- PDF document processing and analysis
- Conversational interface using Gradio
- Vector database storage for efficient document retrieval
- Memory capabilities for maintaining context
- Support for multiple document formats
- Advanced text splitting for optimal processing
- Deployed on Hugging Face Spaces for easy access

## Technology Stack

- **Large Language Model**: llama-3.3-70b-specdec
- **LLM Interface**: GroqLM
- **Framework**: LangChain
- **Embeddings**: Hugging Face sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: Vector database for document storage
- **Text Splitter**: RecursiveCharacterTextSplitter
- **UI Framework**: Gradio
- **Retrieval System**: ConversationRetrievalChain
- **Deployment**: Hugging Face Spaces

## Installation

```bash
# Clone the repository
git clone https://github.com/Tharindu1527/-PDF_Research_Assistance_with_GroqLM_and_Langchain.git

# Navigate to the project directory
cd -PDF_Research_Assistance_with_GroqLM_and_Langchain

# Install required packages
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root and add your API keys:
```
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

2. Ensure you have the necessary dependencies installed:
```
langchain
groq
gradio
sentence-transformers
python-dotenv
```

## Usage

### Online Usage
1. Visit the [Hugging Face Space](https://huggingface.co/spaces/Tharindu1527/PDF_Research_Assistance_with_GroqLM_and_Langchain)
2. Upload your PDF document(s)
3. Start asking questions about the content of your documents

### Local Development
1. Start the application:
```bash
python PDFBot.py
```

2. Open your web browser and navigate to the Gradio interface (typically `http://localhost:7860`)

3. Upload your PDF document(s)

4. Start asking questions about the content of your documents

## Features in Detail

### Document Processing
- Uses RecursiveCharacterTextSplitter for intelligent document chunking
- Processes PDFs while maintaining document structure
- Supports multiple document uploads

### Vector Database Integration
- Efficient storage and retrieval of document embeddings
- Fast similarity search capabilities
- Persistent storage for document vectors

### Conversation Memory
- Maintains context across multiple queries
- Improves response relevance
- Supports follow-up questions

### LLM Integration
- Leverages llama-3.3-70b-specdec model through GroqLM
- Provides natural language understanding
- Generates contextual and accurate responses

## Model Details

### Language Model
- **Model**: llama-3.3-70b-specdec
- **Interface**: GroqLM
- **Capabilities**: Advanced natural language understanding and generation

### Embedding Model
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Source**: Hugging Face
- **Features**: Efficient text embedding generation

## Deployment

The application is deployed on Hugging Face Spaces, providing:
- Easy access through a web interface
- No local setup required for users
- Automatic scaling and availability
- Integration with Hugging Face's infrastructure

Access the deployed version here: [https://huggingface.co/spaces/Tharindu1527/PDF_Research_Assistance_with_GroqLM_and_Langchain](https://huggingface.co/spaces/Tharindu1527/PDF_Research_Assistance_with_GroqLM_and_Langchain)


## Acknowledgments

- GroqLM for providing the LLM interface
- LangChain for the framework
- Hugging Face for the embedding model and hosting
- The open-source community for various tools and libraries

## Contact

Project Link: [https://github.com/Tharindu1527/-PDF_Research_Assistance_with_GroqLM_and_Langchain](https://github.com/Tharindu1527/-PDF_Research_Assistance_with_GroqLM_and_Langchain)

Deployment Link: [https://huggingface.co/spaces/Tharindu1527/PDF_Research_Assistance_with_GroqLM_and_Langchain](https://huggingface.co/spaces/Tharindu1527/PDF_Research_Assistance_with_GroqLM_and_Langchain)
