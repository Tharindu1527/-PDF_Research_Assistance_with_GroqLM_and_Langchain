import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain

class PDFChatAssistant:
    # 01.Deifne Huggingface Embedding , GroqLLM and Vector Database
    
    def __init__(self):
        # Initialize huggingface embeddings
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv(HUGGINGFACEHUB_API_TOKEN)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize the Groq Language Model
        """
        To use Groq’s language model, set your API key and initialize the ChatGroq 
        instance.
        """
        os.environ["GROQ_API_KEY"] = os.getenv(GROQ_API_KEY)
        self.llm = ChatGroq(model="llama-3.3-70b-specdec", temperature=0.5)
        
        # Create vectordatabase to add text_embeddings
        self.vector_store = Chroma(
            collection_name="PDF_info",
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db"
        )
        
        self.conversation_history = []
        self.current_pdf_name = None

    # 02.Extract the text from PDF
        
    def extract_text_from_pdf(self,pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    # 03.Split the text in to chunks

    def split_text_into_chunks(
            self, text: str, chunk_size: int = 2000, chunk_overlap: int = 200
        ) -> List[str]:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return text_splitter.split_text(text)

    # 04.Process of Uploaded PDF

    def process_uploaded_pdf(self, pdf_file) -> str:
        if pdf_file is None:
            return "No PDF file Uploaded"
        
        try:
            # 1.save temp file
            temp_path = "temp.pdf"
            if isinstance(pdf_file, dict):
                pdf_content = pdf_file["file"]
                self.current_pdf_name = pdf_file.get("name", "uploaded.pdf")
            else: # if raw bytes are provided
                pdf_content = pdf_file
                self.current_pdf_name = "Uploaded PDF"
            
            # 2.write the pdf content
            with open(temp_path, "wb") as f:
                f.write(pdf_content)

            # 3.extract and process text
            pdf_text = self.extract_text_from_pdf(temp_path)
            text_chunks = self.split_text_into_chunks(pdf_text)

            # 4.Update vector db
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                collection_name="PDF_info",
                embedding_function=self.embedding_model,
                persist_directory="./chroma_db"
            )
            self.vector_store.add_texts(text_chunks)

            # 5.Clear conversation history
            self.conversation_history =[]

            # 6.clear conversation history
            os.remove(temp_path)

            return f"Successfully processed PDF: {self.current_pdf_name}"
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
        

    # 04.Create the Conversational Retrieval Chain (We already created Vectordatabase)

    def get_response(self, user_query: str) -> str:
        if not self.current_pdf_name:
            return "Please upload a PDF First."
        
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm = self.llm,
            retriever = self.vector_store.as_retriever(search_kwargs={"k":3}),
            return_source_documents = True
        )
        response = retrieval_chain({
            "question" : user_query,
            "chat_history": self.conversation_history
        })

        self.conversation_history.append((user_query, response["answer"]))
        return response["answer"]


    # 05.Build the User Interface with Gradio
import gradio as gr
def create_enhanced_interface():
    assistant = PDFChatAssistant()
    
    with gr.Blocks(css="""
        .container { 
                   max-width: 900px; 
                   margin: auto; 
                   padding: 20px; }
        .header { 
                   text-align: center;
                   margin-bottom: 30px; }
        .chat-container {
                   height: 600px; 
                   overflow-y: auto; 
                   border-radius: 10px; 
                   background-color: #f7f7f7; 
                   padding: 20px; 
                   margin-bottom: 20px; }
        .input-container { 
                   display: flex; 
                   gap: 10px; }
        .footer { 
                   text-align: center;
                    margin-top: 20px; font-size: 
                   0.8em; color: #666; }
    """) as demo:
        with gr.Column(elem_classes="container"):
            with gr.Column(elem_classes="header"):
                gr.Markdown("""
                # 📚 PDF Research Assistant
                ### Your AI-powered research companion for document analysis
                """)
                
                # PDF upload component
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="binary",
                )
                upload_status = gr.Markdown("*Upload a PDF to begin*")

            with gr.Column(elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    container=True,
                    bubble_full_width=False,
                )
                
            with gr.Column():
                with gr.Row(elem_classes="input-container"):
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="Ask me anything about the PDF...",
                        container=False,
                        scale=9
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        "Tell me about uploaded PDF",
                        "What are the main contributions of this paper?",
                        "Summarize the methodology section.",
                        "What are the key findings?",
                        "Explain the limitations of this research.",
                    ],
                    inputs=user_input,
                    label="Example Questions"
                )

            with gr.Column(elem_classes="footer"):
                gr.Markdown("""
                Built with Gradio, LangChain, and Groq LLM | 
                [Source Code](https://github.com/yourusername/pdf-research-assistant)
                """)

        # State management
        state = gr.State([])

        # PDF upload handler
        def handle_pdf_upload(pdf_file):
            result = assistant.process_uploaded_pdf(pdf_file)
            return result, []  # Reset chat history when new PDF is uploaded
        
        pdf_upload.change(
            handle_pdf_upload,
            inputs=[pdf_upload],
            outputs=[upload_status, chatbot]
        )

        # Chat handlers
        def chat_interface(user_input, history):
            if not user_input.strip():
                return history, history
            
            try:
                response = assistant.get_response(user_input)
                history.append((user_input, response))
                return history, history
            except Exception as e:
                error_message = f"Error: {str(e)}"
                history.append((user_input, error_message))
                return history, history

        submit_btn.click(
            chat_interface,
            inputs=[user_input, state],
            outputs=[chatbot, state]
        ).then(
            lambda: gr.Textbox(value="", interactive=True),
            None,
            [user_input]
        )

        user_input.submit(
            chat_interface,
            inputs=[user_input, state],
            outputs=[chatbot, state]
        ).then(
            lambda: gr.Textbox(value="", interactive=True),
            None,
            [user_input]
        )

    return demo

if __name__ == "__main__":
    demo = create_enhanced_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
