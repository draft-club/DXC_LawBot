from flask import Flask
from flask_cors import CORS
import os
from constants import UPLOAD_FOLDER, MAX_CONTENT_LENGTH, SECRET_KEY
from routes import bp as api_bp


def create_app():
    # Initialize Flask app
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Configure app settings
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    app.config['SECRET_KEY'] = SECRET_KEY

    # Register blueprint for routes
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
