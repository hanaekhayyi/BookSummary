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
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from typing import Any, Dict, List, Optional

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

# Split text into smaller chunks for processing with improved settings
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Create vector embeddings for text chunks
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Generate a book summary to help with global context
def generate_book_summary(documents, llm):
    try:
        beginning_docs = documents[:min(3, len(documents))]
        middle_start = max(0, len(documents)//2 - 1)
        middle_docs = documents[middle_start:middle_start + min(3, len(documents) - middle_start)]
        end_start = max(0, len(documents) - 3)
        end_docs = documents[end_start:]
        
        beginning_text = "\n".join([doc.page_content for doc in beginning_docs])
        middle_text = "\n".join([doc.page_content for doc in middle_docs])
        end_text = "\n".join([doc.page_content for doc in end_docs])
        
        summary_prompt = f"""
        You are tasked with creating a comprehensive summary of a book based on excerpts from its beginning, middle, and end.
        
        Beginning of the book:
        {beginning_text[:1000]}...
        
        Middle of the book:
        {middle_text[:1000]}...
        
        End of the book:
        {end_text[:1000]}...
        
        Based on these excerpts, create a detailed summary of the book that captures the main plot points, character development, and overall narrative arc.
        Focus especially on how the story concludes, as this will be important for answering questions about the ending.
        """
        
        summary = llm.invoke(summary_prompt)
        
        summary_doc = Document(
            page_content=f"BOOK SUMMARY: {summary}",
            metadata={"source": "book_summary", "page": "summary", "content_type": "summary"}
        )
        
        print("Book summary generated successfully")
        return summary_doc
    except Exception as e:
        print(f"Error generating book summary: {e}")
        return Document(
            page_content="Unable to generate book summary.",
            metadata={"source": "book_summary", "page": "summary", "content_type": "error"}
        )

# Set up the language model
def setup_llm():
    mistral_api_key = "0rxT82ORNloJ3GIcnod40upRvjOFmEDU"
    print(f"Using MISTRAL_API_KEY: {mistral_api_key[:4]}**** (hidden for security)")
    
    model_name = "mistral-small-latest"
    try:
        llm = ChatMistralAI(
            model=model_name,
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            mistral_api_key=mistral_api_key
        )
        print(f"Mistral API model '{model_name}' configured successfully")
        return llm
    except Exception as e:
        print(f"Failed to initialize Mistral API: {str(e)}")
        raise

# Create an enhanced retriever
# Create an enhanced retriever
class EnhancedRetriever(BaseRetriever):
    vector_store: Any
    book_metadata: Dict[str, Any]
    basic_retriever: Any
    
    def __init__(self, vector_store: Any, book_metadata: Dict[str, Any]):
        # Initialize the Pydantic model with all required fields
        super().__init__(
            vector_store=vector_store,
            book_metadata=book_metadata,
            basic_retriever=vector_store.as_retriever(search_kwargs={"k": 10})
        )
        # Additional initialization if needed
        self.vector_store = vector_store
        self.book_metadata = book_metadata
        self.basic_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        docs = self.basic_retriever.invoke(query)  # Changed from get_relevant_documents to invoke
        
        if any(word in query.lower() for word in ["end", "ending", "finale", "conclusion", "final", "last"]):
            docs.sort(key=lambda doc: doc.metadata.get("page", 0) if isinstance(doc.metadata.get("page", 0), int) else 0, reverse=True)
            beginning_docs = docs[-3:] if len(docs) > 3 else []
            ending_docs = docs[:-3] if len(docs) > 3 else docs
            docs = ending_docs + beginning_docs
            print("Query about ending detected, prioritizing later pages")
        elif any(word in query.lower() for word in ["begin", "beginning", "start", "introduction", "first"]):
            docs.sort(key=lambda doc: doc.metadata.get("page", 0) if isinstance(doc.metadata.get("page", 0), int) else 0)
            print("Query about beginning detected, prioritizing earlier pages")
        
        summary_docs = [doc for doc in self.vector_store.similarity_search("BOOK SUMMARY", k=1) 
                       if doc.metadata.get("content_type") == "summary"]
        
        if summary_docs and not any(doc.metadata.get("content_type") == "summary" for doc in docs):
            docs = summary_docs + docs
        
        return docs
    
    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        return await self._get_relevant_documents(query)
    
# Create the RAG chain with improved context handling
def create_rag_chain(vector_store, llm, book_metadata):
    retriever = EnhancedRetriever(vector_store, book_metadata)
    
    template = """
    You are an assistant specialized in the book "{title}" written by {author}.
    
    Use the following context from the book to answer the user's question thoroughly and accurately.
    The context includes excerpts from different parts of the book, including a summary of the overall narrative.
    
    If the question is about the ending or conclusion of the book, make sure to focus on the relevant parts of the context that discuss how the story ends.
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
        enhanced_question = question
        
        if any(word in question.lower() for word in ["end", "ending", "finale", "conclusion", "final", "last"]):
            enhanced_question = f"Regarding the ending or final parts of the book: {question}"
        elif any(word in question.lower() for word in ["character", "who is", "what is", "describe"]):
            enhanced_question = f"Information about characters in the book: {question}"
        
        result = qa_chain({"query": enhanced_question})
        answer = result["result"]
        
        sources = result.get("source_documents", [])
        if sources:
            source_pages = set()
            for doc in sources:
                if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                    if doc.metadata['page'] != 'summary':
                        source_pages.add(doc.metadata['page'])
            
            if source_pages:
                page_refs = ", ".join([str(page) for page in sorted(source_pages)])
                answer += f"\n\nSources: Pages {page_refs}"
            
            if any(hasattr(doc, 'metadata') and doc.metadata.get('content_type') == 'summary' for doc in sources):
                answer += "\n(Information partially derived from book summary)"
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
    
    print("Setting up language model...")
    llm = setup_llm()
    
    print("Generating book summary...")
    summary_doc = generate_book_summary(documents, llm)
    
    print("Splitting text into chunks...")
    chunks = split_text(documents)
    print(f"Text split into {len(chunks)} chunks")
    
    chunks.append(summary_doc)
    print("Added book summary to chunks")
    
    print("Creating vector database...")
    vector_store = create_vector_store(chunks)
    
    print("Creating RAG chain...")
    qa_chain = create_rag_chain(vector_store, llm, book_metadata)
    
    print("Launching chatbot interface...")
    demo = create_chatbot_interface(qa_chain, book_metadata)
    demo.launch()

if __name__ == "__main__":
    book_path = "C:\\Users\\hp\\Desktop\\testDL\\thief_the__percy_jac_-_rick_riordan.pdf"
    main(book_path)