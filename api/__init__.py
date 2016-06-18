from flask import Flask

app = Flask(__name__)  # Holds the Flask instance
from api import views