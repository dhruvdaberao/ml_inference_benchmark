import sys
import os

# Add parent directory to path so we can import 'analyze', 'engines', etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request
from analyze import run_analysis

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        input_str = request.form.get('input_str', '')
        batch_size = int(request.form.get('batch_size', 32))
        mode = request.form.get('mode', 'both')

        # Validate basic input
        if not input_str.strip():
            raise ValueError("Input vector cannot be empty.")

        results = run_analysis(input_str, batch_size, mode)

        return render_template('index.html', results=results, last_input=input_str, last_batch=batch_size, last_mode=mode)

    except ValueError as ve:
        return render_template('index.html', error=str(ve), last_input=request.form.get('input_str'), last_batch=request.form.get('batch_size'))
    except Exception as e:
        return render_template('index.html', error=f"System Error: {str(e)}", last_input=request.form.get('input_str'))

if __name__ == '__main__':
    # Render Requirement: Bind to 0.0.0.0 and use PORT env var
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
