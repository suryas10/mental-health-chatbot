import os
import random
import re
from datetime import datetime
from pathlib import Path, WindowsPath

import pyttsx3
import spacy
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from textblob import TextBlob

# Create necessary directories using Windows paths
BASE_DIR = WindowsPath.cwd()
DATABASE_DIR = BASE_DIR / "database"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Create directories if they don't exist
DATABASE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Print current session info
print(
    f"Session Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"User: {os.getlogin()}")
print(f"Working Directory: {BASE_DIR}")

# Initialize spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize the language model


def initialize_llm():
    llm = ChatGroq(
        temperature=1.2,
        top_p=0.9,
        max_tokens=2000,
        presence_penalty=0.6,
        # Replace with your API key
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    return llm

# Initialize Windows text-to-speech


def initialize_tts():
    try:
        engine = pyttsx3.init()
        # Windows specific voice settings
        voices = engine.getProperty('voices')
        # Try to set a female voice if available
        for voice in voices:
            if "female" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        return engine
    except Exception as e:
        print(f"TTS initialization error: {e}")
        return None

# Create or load the vector database with Windows path handling


def create_vector_db():
    try:
        loader = DirectoryLoader(
            str(DATABASE_DIR),
            glob='*.pdf',
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        if not documents:
            print(f"\nNo PDF files found in: {DATABASE_DIR}")
            print("Please add PDF files with mental health content.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceBgeEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        vector_db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        vector_db.persist()
        print("ChromaDB created and data saved")
        return vector_db
    except Exception as e:
        print(f"Error creating vector database: {e}")
        return None

# Setup the QA chain with an enhanced prompt template


def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """

    You are Trixie, a compassionate and empathetic mental health chatbot. Respond thoughtfully with empathy, validation, and helpful guidance.
    Always maintain a supportive and non-judgmental tone, ensuring the user feels heard and understood.

    {context}
    User: {question}
    Trixie: 
    """
    PROMPT = PromptTemplate(template=prompt_templates,
                            input_variables=['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain
# Function to analyze sentiment


def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to extract named entities


def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Initialize the text-to-speech engine


def initialize_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # 0 = Male, 1 = Female

    return engine


tts_engine = initialize_tts()

# Function to add line breaks after a certain number of words


def add_line_breaks(text, words_per_line=25):
    words = text.split()
    lines = [' '.join(words[i:i + words_per_line])
             for i in range(0, len(words), words_per_line)]
    return '\n'.join(lines)


global qa_chain
qa_chain = None


# Handle chatbot response and track history
def chatbot_response(user_input, qa_chain, history=[]):
    if not user_input.strip():
        return "Please provide a valid input", history
    sentiment = analyze_sentiment(user_input)
    entities = extract_entities(user_input)
    response = qa_chain.run(user_input)
    formatted_response = add_line_breaks(response)
    history.append((user_input, formatted_response, sentiment, entities))
    return formatted_response, history

# The rest of your existing functions remain the same...
# (setup_qa_chain, analyze_sentiment, extract_entities, add_line_breaks, chatbot_response)


# Goodbye messages for the user
GOODBYE_MESSAGES = [
    "Take care of yourself, Goodbye!",
    "Hope to talk again soon. Stay safe!",
    "Wishing you a great day ahead!",
    "Goodbye! Remember to take breaks and stay positive!",
    "See you next time! Keep smiling!"
]
# Modify your main function to pass qa_chain to chatbot_response


def main():
    global qa_chain  # Declare global usage
    print("\n=== Trixie - Mental Health Chatbot ===")
    print(f"Database Directory: {DATABASE_DIR}")
    print(f"ChromaDB Directory: {CHROMA_DIR}")

    # Check for PDF files
    pdf_files = list(DATABASE_DIR.glob("*.pdf"))
    if not pdf_files:
        print("\nNo PDF files found in the database directory.")
        print(f"Please add PDF files to: {DATABASE_DIR}")
        return

    # Initialize components
    try:
        print("\nInitializing components...")
        llm = initialize_llm()
        tts_engine = initialize_tts()

        if tts_engine is None:
            print("Warning: Text-to-speech is not available")

        # Initialize or load vector database
        if not list(CHROMA_DIR.glob("*")):
            print("Creating new vector database...")
            vector_db = create_vector_db()
        else:
            print("Loading existing vector database...")
            embeddings = HuggingFaceBgeEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
            vector_db = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings
            )

        if vector_db is None:
            print("Failed to initialize vector database.")
            return

        qa_chain = setup_qa_chain(vector_db, llm)

        print("\nTrixie is ready to chat!")
        print("Type 'exit', 'bye', or 'quit' to end the conversation")

        while True:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "bye", "goodbye", "quit"]:
                farewell_message = random.choice(GOODBYE_MESSAGES)
                print(f"\nTrixie: {farewell_message}")
                if tts_engine:
                    tts_engine.say(farewell_message)
                    tts_engine.runAndWait()
                break

            response, _ = chatbot_response(
                user_input, qa_chain)  # Pass qa_chain here
            print(f"\nTrixie: {response}")
            if tts_engine:
                tts_engine.say(response)
                tts_engine.runAndWait()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
