from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pdfplumber
from pymongo import MongoClient, ASCENDING
from flask_cors import CORS
import logging
import numpy as np
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
import torch
import faiss
from dotenv import load_dotenv
from bson import ObjectId
import datetime
from httpx import ConnectError

# Load environment variables
load_dotenv()

torch.cuda.empty_cache()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# MongoDB setup
mongo_conn_url = os.getenv("MONGO_CONN_URL")
client = MongoClient(mongo_conn_url)
db = client['pdf_query_db']
chunks_collection = db['chunks']
discussions_collection = db['discussions']
chats_collection = db['chats']
users_collection = db['users']

# Create an index to prevent duplicate chunks
chunks_collection.create_index([('filename', ASCENDING), ('page_number', ASCENDING), ('text', ASCENDING)], unique=True)

FAISS_INDEX_PATH = './faiss_index.index'
query_count = 0
MAX_QUERIES_BEFORE_CLEAR = 10

# Initialize embedding model
def initialize_embedding_model(model_name, voyage_api_key):
    embed_model = VoyageEmbedding(model_name=model_name, voyage_api_key=voyage_api_key)
    sample_text = "Sample text to determine embedding dimension"
    sample_embedding = embed_model.get_text_embedding(sample_text)
    embedding_dimension = len(sample_embedding)
    logging.info(f"Embedding dimension is {embedding_dimension}")
    return embed_model, embedding_dimension

# Load the embedding model
voyage_embed_model, voyage_embedding_dimension = initialize_embedding_model("voyage-law-2", os.getenv("VOYAGE_API_KEY"))

# Initialize LLMs
ollama_llm = Ollama(
    model="llama3.1",
    max_length=4096,
    temperature=0.7,
    top_p=0.9,
    device_map="auto",
    server_url="http://localhost:11434",
    request_timeout=4600.0,
)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("No API key found for OpenAI. Please set the OPENAI_API_KEY in the .env file.")
openai_llm = OpenAI(
    api_key=openai_api_key,
    model="gpt-3.5-turbo",
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9,
)

# FAISS index setup
dimension = voyage_embedding_dimension
try:
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    logging.info("Loaded existing FAISS index")
    if faiss_index.d != dimension:
        raise ValueError(f"Loaded FAISS index has dimension {faiss_index.d}, but expected {dimension}.")
except (Exception, ValueError) as e:
    logging.info(f"Creating new FAISS index: {e}")
    faiss_index = faiss.IndexFlatL2(dimension)

def save_faiss_index():
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

def embed(text):
    if not text.strip():
        logging.warning("Attempted to embed an empty string")
        return []
    embedding = voyage_embed_model.get_text_embedding(text)
    logging.debug(f"Embedding for text '{text[:50]}...' is {embedding[:5]}...")
    return embedding

def split_text_to_chunks(text, chunk_size=1000, chunk_overlap=200, page_number=None):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_chunk) > chunk_size:
            chunks.append({'text': ' '.join(current_chunk), 'page_number': page_number})
            current_chunk = []
            current_length = 0

        current_chunk.append(word)
        current_length += len(word)

    if current_chunk:
        chunks.append({'text': ' '.join(current_chunk), 'page_number': page_number})

    return chunks

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                text_chunks = split_text_to_chunks(text, chunk_size=1000, chunk_overlap=200, page_number=page_number)
                for chunk in text_chunks:
                    if not chunks_collection.find_one({
                        'filename': filename,
                        'page_number': chunk['page_number'],
                        'text': chunk['text']
                    }):
                        chunks_collection.insert_one({
                            'filename': filename,
                            'page_number': chunk['page_number'],
                            'text': chunk['text']
                        })
                        embedding = embed(chunk['text'])
                        if len(embedding) != dimension:
                            raise ValueError(f"Embedding dimension {len(embedding)} does not match FAISS index dimension {dimension}.")
                        faiss_index.add(np.array([embedding]))

        save_faiss_index()
        return jsonify({'message': 'PDF uploaded and processed', 'filename': filename})
    except Exception as e:
        logging.error(f"Error processing PDF: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    global query_count, memory
    query_count += 1
    data = request.get_json()
    query_text = data.get('query')
    llm_choice = data.get('llm', 'ollama')
    discussion_id = data.get('discussion_id')

    # Ensure query text is provided
    if not query_text:
        return jsonify({'error': 'Query parameter is missing'}), 400

    # Retrieve document context based on discussion_id
    document_context = ""
    if discussion_id:
        chunks = chunks_collection.find({'discussion_id': ObjectId(discussion_id)})
        document_context = "\n\n".join([chunk['text'] for chunk in chunks])

    if not document_context:
        return jsonify({'response': "I'm sorry, but I cannot sum up the document as there is no document provided for me to analyze. If you have any specific information or text you would like me to summarize, please provide it and I would be happy to help."}), 200

    if query_count >= MAX_QUERIES_BEFORE_CLEAR:
        memory = ChatMemoryBuffer(token_limit=4096)
        query_count = 0
        logging.info("Memory automatically cleared after {} queries".format(MAX_QUERIES_BEFORE_CLEAR))

    try:
        logging.info(f"Processing query: {query_text} with LLM: {llm_choice}")

        # Set the LLM according to user's choice
        selected_llm = openai_llm if llm_choice == 'openai' else ollama_llm

        # Generate prompt with context
        prompt_with_context = f"Here is the context: {document_context}\n\nQuery: {query_text}"

        try:
            # Process the query using the LLM
            result = selected_llm.complete(prompt_with_context)
        except ConnectError:
            logging.error("Could not connect to Ollama server; switching to OpenAI as fallback.")
            result = openai_llm.complete(prompt_with_context)

        # Update chat memory with user query and assistant response
        memory.put(ChatMessage(role=MessageRole.USER, content=query_text))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=result))

        # Save the query and response in MongoDB
        user_message_id = chats_collection.insert_one({
            'discussion_id': ObjectId(discussion_id),
            'role': 'user',
            'query': query_text,
            'timestamp': datetime.datetime.utcnow()
        }).inserted_id

        chats_collection.insert_one({
            'discussion_id': ObjectId(discussion_id),
            'role': 'assistant',
            'response': result,
            'timestamp': datetime.datetime.utcnow(),
            'parent_message_id': user_message_id
        })

        discussions_collection.update_one(
            {'_id': ObjectId(discussion_id), 'name': {'$in': [None, 'New Discussion']}},
            {'$set': {'name': query_text[:50]}}
        )

        return jsonify({'response': result})

    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/delete_discussion/<discussion_id>', methods=['DELETE'])
def delete_discussion(discussion_id):
    try:
        result = discussions_collection.delete_one({'_id': ObjectId(discussion_id)})
        chats_collection.delete_many({'discussion_id': ObjectId(discussion_id)})

        if result.deleted_count == 1:
            return jsonify({'message': 'Discussion deleted successfully'}), 200
        else:
            return jsonify({'error': 'Discussion not found'}), 404
    except Exception as e:
        logging.error(f"Error deleting discussion: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    global memory
    memory = ChatMemoryBuffer(token_limit=4096)
    return jsonify({'message': 'Chat memory cleared'}), 200

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, use_reloader=False, port=5000)
