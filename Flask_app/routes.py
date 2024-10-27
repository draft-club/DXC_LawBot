from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
import os
import logging
import numpy as np
from bson import ObjectId
import datetime
import pdfplumber
from model_setup import ollama_llm, openai_llm, voyage_embed_model
from index_setup import faiss_index, save_faiss_index
from constants import UPLOAD_FOLDER, MAX_QUERIES_BEFORE_CLEAR, MONGO_CONN_URL
from pymongo import MongoClient, ASCENDING
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from httpx import ConnectError

# Blueprint setup for routes
bp = Blueprint('api', __name__)

# MongoDB setup
client = MongoClient(MONGO_CONN_URL)
db = client['pdf_query_db']
chunks_collection = db['chunks']
discussions_collection = db['discussions']
chats_collection = db['chats']
users_collection = db['users']

# Chat memory and query tracking
memory = ChatMemoryBuffer(token_limit=4096)
query_count = 0


# Utility function to embed text using the embedding model
def embed(text):
    if not text.strip():
        logging.warning("Attempted to embed an empty string")
        return []
    return voyage_embed_model.get_text_embedding(text)


# Route to upload and process PDF files
@bp.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    # Split text into chunks
                    chunks = [{'text': text[i:i + 1000], 'page_number': page_number} for i in range(0, len(text), 1000)]
                    for chunk in chunks:
                        if not chunks_collection.find_one(
                                {'filename': filename, 'page_number': page_number, 'text': chunk['text']}):
                            # Save chunk to MongoDB
                            chunks_collection.insert_one(
                                {'filename': filename, 'page_number': page_number, 'text': chunk['text']})

                            # Embed the chunk and add it to FAISS index
                            embedding = embed(chunk['text'])
                            faiss_index.add(np.array([embedding]))

        # Save FAISS index after processing
        save_faiss_index(faiss_index)
        return jsonify({'message': 'PDF uploaded and processed successfully', 'filename': filename})
    except Exception as e:
        logging.error(f"Error processing PDF: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Route to process and respond to user queries
@bp.route('/query', methods=['POST'])
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
        return jsonify({'response': "No document context provided."}), 200

    # Reset memory after MAX_QUERIES_BEFORE_CLEAR queries
    if query_count >= MAX_QUERIES_BEFORE_CLEAR:
        memory = ChatMemoryBuffer(token_limit=4096)
        query_count = 0
        logging.info("Memory automatically cleared after reaching max queries")

    try:
        selected_llm = openai_llm if llm_choice == 'openai' else ollama_llm
        prompt_with_context = f"Context: {document_context}\n\nQuery: {query_text}"

        # Attempt to complete the query
        try:
            result = selected_llm.complete(prompt_with_context)
        except ConnectError:
            logging.error("Could not connect to Ollama server; switching to OpenAI as fallback.")
            result = openai_llm.complete(prompt_with_context)

        # Update chat memory with user query and assistant response
        memory.put(ChatMessage(role=MessageRole.USER, content=query_text))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=result))

        # Save query and response in MongoDB
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

        # Update discussion name if it’s the first query
        discussions_collection.update_one(
            {'_id': ObjectId(discussion_id), 'name': {'$in': [None, 'New Discussion']}},
            {'$set': {'name': query_text[:50]}}
        )

        return jsonify({'response': result})

    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Route to delete a discussion and associated messages
@bp.route('/delete_discussion/<discussion_id>', methods=['DELETE'])
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


# Route to clear chat memory buffer
@bp.route('/clear_memory', methods=['POST'])
def clear_memory():
    global memory
    memory = ChatMemoryBuffer(token_limit=4096)
    return jsonify({'message': 'Chat memory cleared'}), 200
