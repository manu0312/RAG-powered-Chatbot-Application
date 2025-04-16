import os
import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import base64

# Load environment variables
load_dotenv()

# Directly set API key if not found in environment
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"  

# Set page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Function to create LLM instance
def create_llm():
    return ChatGroq(
        temperature=0.2,
        model_name="llama3-70b-8192",
        groq_api_key=os.environ["GROQ_API_KEY"]
    )

# Function to process the uploaded PDF
def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    # Load PDF
    with st.spinner("Processing PDF... This might take a while for large documents."):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Clean up the temporary file
        os.unlink(pdf_path)
        
        return vectorstore, chunks

# Function to generate a summary of the document
def generate_summary(chunks):
    llm = create_llm()
    
    combined_text = " ".join([chunk.page_content for chunk in chunks[:15]])  # Using first chunks for summary
    
    summary_prompt = PromptTemplate.from_template(
        """You are an expert document analyst. 
        Based on the following document excerpts, generate a concise summary with 5-10 key points.
        
        DOCUMENT EXCERPTS:
        {text}
        
        SUMMARY (5-10 bullet points):"""
    )
    
    summary = llm.invoke(summary_prompt.format(text=combined_text))
    return summary.content

# Function to generate suggested questions
def generate_suggested_questions(chunks):
    llm = create_llm()
    
    combined_text = " ".join([chunk.page_content for chunk in chunks[:15]])  # Using first chunks for questions
    
    question_prompt = PromptTemplate.from_template(
        """You are an expert document analyst. 
        Based on the following document excerpts, generate 5-7 insightful questions that a user might want to ask about this document.
        The questions should be varied and cover different aspects of the content.
        Format the questions as a numbered list.
        
        DOCUMENT EXCERPTS:
        {text}
        
        SUGGESTED QUESTIONS (5-7 questions):"""
    )
    
    questions = llm.invoke(question_prompt.format(text=combined_text))
    
    # Parse the questions from the response
    questions_list = [q.strip() for q in questions.content.split("\n") if q.strip()]
    # Remove any numbers or bullet points at the beginning of each question
    questions_list = [q.split('. ', 1)[-1] if '. ' in q else q for q in questions_list]
    questions_list = [q.lstrip('1234567890.- ') for q in questions_list]
    
    return questions_list

# Function to create the conversational chain
def get_conversation_chain(vectorstore):
    llm = create_llm()
    
    # Define a custom prompt template
    qa_template = """
    You are an intelligent assistant that helps users understand documents. Answer the question based on the context provided.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    CONTEXT:
    {context}
    
    CHAT HISTORY:
    {chat_history}
    
    QUESTION:
    {question}
    
    YOUR ANSWER:"""
    
    QA_PROMPT = PromptTemplate(
        template=qa_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    
    return conversation_chain

def handle_user_question(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload a PDF document first.")
        return
    
    with st.spinner("Thinking..."):
        response = st.session_state.conversation(
            {"question": user_question, "chat_history": st.session_state.chat_history}
        )
        
        answer = response["answer"]
        sources = response["source_documents"]
        
        # Format the answer with source references
        if sources:
            source_text = "\n\n**Sources:**\n"
            for i, source in enumerate(sources[:3]):  # Limit to top 3 sources
                page = source.metadata.get("page", "Unknown page")
                excerpt = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                source_text += f"\n**Page {page}:** {excerpt}\n"
            
            answer_with_sources = answer + source_text
        else:
            answer_with_sources = answer
        
        # Add to chat history
        st.session_state.chat_history.append((user_question, answer_with_sources))
        
        # Force a rerun to display the new message immediately
        st.rerun()
        
        return response

# Function to download chat history
def get_download_link(chat_history, filename="chat_history.txt"):
    chat_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
    b64 = base64.b64encode(chat_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Chat History</a>'
    return href

# Main function
def main():
    st.title("📚 RAG-powered PDF Chatbot")
    
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Check if a new file was uploaded
            if st.session_state.processed_file != uploaded_file.name:
                with st.spinner("Processing new document..."):
                    vectorstore, chunks = process_pdf(uploaded_file)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    # Generate document summary
                    with st.spinner("Generating document summary..."):
                        st.session_state.document_summary = generate_summary(chunks)
                    
                    # Generate suggested questions
                    with st.spinner("Generating suggested questions..."):
                        st.session_state.suggested_questions = generate_suggested_questions(chunks)
                    
                    st.session_state.processed_file = uploaded_file.name
                st.success(f"Successfully processed: {uploaded_file.name}")
                
                # Reset chat history when a new document is uploaded
                st.session_state.chat_history = []
        
        if st.session_state.chat_history:
            st.download_button(
                label="Download Chat History",
                data="\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history]),
                file_name="chat_history.txt",
                mime="text/plain"
            )
    
    # Main content area
    if uploaded_file is not None:
        # Create tabs for Summary and Chat
        tab1, tab2 = st.tabs(["Document Summary", "Chat"])
        
        with tab1:
            st.header("Document Summary")
            if st.session_state.document_summary:
                st.write(st.session_state.document_summary)
            else:
                st.info("Summary is being generated...")
                
            st.header("Suggested Questions")
            if st.session_state.suggested_questions:
                for i, question in enumerate(st.session_state.suggested_questions):
                    if st.button(f"{question}", key=f"q_{i}"):
                        with tab2:  # Switch to chat tab when a suggested question is clicked
                            response = handle_user_question(question)
            else:
                st.info("Suggested questions are being generated...")
                
        with tab2:
            st.header("Chat with your PDF")
            
            # Display chat history
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.chat_message("user").write(question)
                st.chat_message("assistant").write(answer)
            
            # User input for new questions
            user_question = st.chat_input("Ask a question about your document:")
            if user_question:
                st.chat_message("user").write(user_question)
                response = handle_user_question(user_question)
    else:
        st.info("Please upload a PDF document to get started.")
        st.markdown("""
        **This application allows you to:**
        - Upload large PDFs (230+ pages)
        - Get an automatic summary of the document
        - See suggested questions based on the content
        - Ask questions about the document using a chat interface
        - Download the chat history for future reference
        """)

if __name__ == "__main__":
    main()