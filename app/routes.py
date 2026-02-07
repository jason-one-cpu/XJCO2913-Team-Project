from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def index():
    # Placeholder until frontend is ready
    return render_template('index.html')