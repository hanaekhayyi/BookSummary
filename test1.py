from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
import gradio as gr
import os
import re
from PyPDF2 import PdfReader

# Extract book metadata from filename and PDF
def extract_book_metadata(file_path):
    try:
        filename = os.path.basename(file_path)
        match = re.match(r'(.+?)_-_(.+?)\.pdf', filename)
        
        title = ""
        author = ""
        if match:
            title = match.group(1).replace('_', ' ').title()
            author = match.group(2).replace('_', ' ').title()
        else:
            title = filename.split('.')[0].replace('_', ' ').title()
        
        reader = PdfReader(file_path)
        info = reader.metadata
        if info:
            if not title and info.get('/Title'):
                title = info.get('/Title')
            if not author and info.get('/Author'):
                author = info.get('/Author')
        
        num_pages = len(reader.pages)
        
        return {
            "title": title,
            "author": author,
            "pages": num_pages,
            "file_path": file_path
        }
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {
            "title": os.path.basename(file_path),
            "author": "Unknown",
            "pages": 0,
            "file_path": file_path
        }

# Load book from PDF file
def load_book(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Split text into smaller chunks for processing
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Create vector embeddings for text chunks
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# Set up the language model
def setup_llm():
    # Clé API codée en dur (non recommandé pour la production)
    mistral_api_key = "0rxT82ORNloJ3GIcnod40upRvjOFmEDU"
    print(f"Using MISTRAL_API_KEY: {mistral_api_key[:4]}**** (hidden for security)")
    
    model_name = "mistral-small-latest"
    try:
        llm = ChatMistralAI(
            model=model_name,
            temperature=0.7,
            max_tokens=300,
            top_p=0.9,
            mistral_api_key=mistral_api_key
        )
        print(f"Mistral API model '{model_name}' configured successfully")
        return llm
    except Exception as e:
        print(f"Failed to initialize Mistral API: {str(e)}")
        raise

# Create the RAG chain
def create_rag_chain(vector_store, llm, book_metadata):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    template = """
    You are an assistant specialized in the book "{title}" written by {author}.
    
    Use the following context from the book to answer the user's question concisely and accurately.
    If the answer is not in the provided context, say "I don't have enough information from the book to answer this question."
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        partial_variables={"title": book_metadata["title"], "author": book_metadata["author"]}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Generate chatbot responses
def chatbot_response(question, qa_chain, chat_history):
    if not question:
        return "Please ask a question about the book."
    
    try:
        result = qa_chain({"query": question})
        answer = result["result"]
        
        sources = result.get("source_documents", [])
        if sources:
            source_pages = set()
            for doc in sources:
                if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                    source_pages.add(doc.metadata['page'])
            
            if source_pages:
                page_refs = ", ".join([str(page) for page in sorted(source_pages)])
                answer += f"\n\nSources: Pages {page_refs}"
        else:
            answer += "\n\nNo specific pages found for this question."
        
        return answer
    except Exception as e:
        return f"Error processing the question: {str(e)}"

# Create Gradio chat interface
def create_chatbot_interface(qa_chain, book_metadata):
    with gr.Blocks() as demo:
        gr.Markdown(f"# Chatbot for the book: {book_metadata['title']}")
        gr.Markdown(f"## Author: {book_metadata['author']}")
        gr.Markdown(f"### Number of pages: {book_metadata['pages']}")
        
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(placeholder="Ask a question about the book...")
        clear = gr.Button("Clear conversation")
        
        def user(user_message, history):
            return "", history + [[user_message, None]]
        
        def bot(history):
            user_message = history[-1][0]
            bot_response = chatbot_response(user_message, qa_chain, history)
            history[-1][1] = bot_response
            return history
        
        def clear_history():
            return []
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(clear_history, None, chatbot)
        
    return demo

# Main function
def main(book_path):
    if not os.path.exists(book_path):
        raise FileNotFoundError(f"Book file not found at {book_path}")
    
    print("Extracting book metadata...")
    book_metadata = extract_book_metadata(book_path)
    print(f"Book: {book_metadata['title']} by {book_metadata['author']}")
    
    print("Loading book...")
    documents = load_book(book_path)
    print(f"Book loaded: {len(documents)} pages")
    
    print("Splitting text into chunks...")
    chunks = split_text(documents)
    print(f"Text split into {len(chunks)} chunks")
    
    print("Creating vector database...")
    vector_store = create_vector_store(chunks)
    
    print("Setting up language model...")
    llm = setup_llm()
    
    print("Creating RAG chain...")
    qa_chain = create_rag_chain(vector_store, llm, book_metadata)
    
    print("Launching chatbot interface...")
    demo = create_chatbot_interface(qa_chain, book_metadata)
    demo.launch()  # Removed share=True for local testing

if __name__ == "__main__":
    book_path = "C:\\Users\\hp\\Desktop\\testDL\\the_road_-_text.pdf"
    main(book_path)