import sys
import os
import logging
import traceback

# Setup basic logging to stdout/stderr
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import 'analyze', 'engines', etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request
# Import run_analysis safely
try:
    from analyze import run_analysis
except ImportError as e:
    logger.error(f"Failed to import run_analysis: {e}")
    run_analysis = None

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    logger.info("Received /analyze POST request")
    
    if run_analysis is None:
        return render_template('index.html', error="Server Configuration Error: Could not import analysis engine.")

    try:
        input_str = request.form.get('input_str', '')
        batch_size_str = request.form.get('batch_size', '32')
        mode = request.form.get('mode', 'both')
        
        logger.info(f"Params: Input='{input_str[:20]}...', Batch={batch_size_str}, Mode={mode}")

        # Validate input
        if not input_str.strip():
            raise ValueError("Input vector cannot be empty.")
            
        try:
            batch_size = int(batch_size_str)
        except ValueError:
            raise ValueError("Batch size must be an integer.")

        logger.info("Running analysis...")
        results = run_analysis(input_str, batch_size, mode)
        logger.info("Analysis complete.")

        return render_template('index.html', results=results, last_input=input_str, last_batch=batch_size, last_mode=mode)

    except ValueError as ve:
        logger.warning(f"User Error: {ve}")
        return render_template('index.html', error=str(ve), last_input=request.form.get('input_str'), last_batch=request.form.get('batch_size'))
    except Exception as e:
        logger.error(f"System Error: {e}")
        traceback.print_exc()
        return render_template('index.html', error=f"Internal Error: {str(e)}", last_input=request.form.get('input_str'))

# Catch-all for uncaught exceptions during rendering
@app.errorhandler(500)
def handle_500(e):
    logger.error(f"500 Error: {e}")
    traceback.print_exc()
    return f"<h1>Internal Server Error</h1><pre>{str(e)}</pre>", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)
