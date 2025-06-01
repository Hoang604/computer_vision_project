from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import base64
import numpy as np
from PIL import Image
import io
import logging
from datetime import datetime
import uuid

# Assuming diffusion_upscale.py is in the parent directory's 'scripts' folder
# Adjust the path if your structure is different
import sys
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Ensure the import alias matches what's used, or use the direct function name
from scripts.diffusion_upscale import upscale as RRDN_diff_upscale 

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Configuration for upload and generated content folders
UPLOAD_FOLDER = os.path.join(app.static_folder or 'static', 'uploads')
GENERATED_FOLDER = os.path.join(app.static_folder or 'static', 'generated_outputs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

# Ensure these folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# Allowed image extensions and resolution limits
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MIN_RESOLUTION = 10
MAX_RESOLUTION = 64 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_resolution(resolution_str):
    """Validate and return resolution as integer."""
    try:
        resolution = int(resolution_str)
        if not (MIN_RESOLUTION <= resolution <= MAX_RESOLUTION):
            return None, f'Resolution must be between {MIN_RESOLUTION} and {MAX_RESOLUTION}'
        return resolution, None
    except (ValueError, TypeError):
        return None, 'Invalid resolution format'

def generate_unique_filename(original_filename):
    """Generate a unique filename to avoid conflicts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(secure_filename(original_filename))
    return f"{name}_{timestamp}_{unique_id}{ext}"

def cleanup_temp_files(*file_paths):
    """Clean up temporary files safely."""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except OSError as e:
            logger.warning(f"Could not remove temporary file {file_path}: {e}")

@app.route('/', methods=['GET'])
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/upscale', methods=['POST'])
def handle_upscale():
    """Handle image upscaling requests."""
    # Validate request
    if 'img_file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['img_file']
    resolution_str = request.form.get('resolution')

    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not resolution_str:
        return jsonify({'error': 'No resolution provided'}), 400

    # Validate resolution
    resolution, error_msg = validate_resolution(resolution_str)
    if error_msg:
        return jsonify({'error': error_msg}), 400

    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Generate unique filename and save file
    unique_filename = generate_unique_filename(file.filename)
    temp_lr_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        file.save(temp_lr_path)
        logger.info(f"Saved uploaded file: {temp_lr_path}")

        # Validate image file
        try:
            with Image.open(temp_lr_path) as img:
                img.verify()  # Verify it's a valid image
        except Exception as e:
            return jsonify({'error': 'Invalid or corrupted image file'}), 400

        # Perform upscaling
        logger.info(f"Starting upscaling process for {unique_filename} with resolution {resolution}")
        
        upscaled_image_np, video_relative_filename, plot_path_none = RRDN_diff_upscale(
            input_lr_image_path=temp_lr_path,
            target_lr_edge_size=resolution
        )

        # Validate upscaling result
        if not isinstance(upscaled_image_np, np.ndarray):
            logger.error("Upscale function did not return a NumPy array")
            return jsonify({'error': 'Upscaling process failed'}), 500

        # Convert to PIL Image and encode as base64
        upscaled_image_np_uint8 = np.clip(upscaled_image_np * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(upscaled_image_np_uint8, 'RGB')
        
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG", optimize=True)
        base64_upscaled_img = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Handle video URL if available
        video_url = None
        if video_relative_filename:
            try:
                video_file_path_for_url = f"{os.path.basename(app.config['GENERATED_FOLDER'])}/{video_relative_filename}"
                video_url = url_for('static', filename=video_file_path_for_url)
            except Exception as e:
                logger.warning(f"Could not generate video URL: {e}")

        # Generate URL for original image
        lr_image_file_path_for_url = f"{os.path.basename(app.config['UPLOAD_FOLDER'])}/{unique_filename}"
        lr_image_url = url_for('static', filename=lr_image_file_path_for_url)
        
        logger.info(f"Upscaling completed successfully for {unique_filename}")
        
        return jsonify({
            'message': 'Image upscaled successfully',
            'lr_image_url': lr_image_url, 
            'upscaled_image_b64': base64_upscaled_img,
            'video_url': video_url,
            'plot_url': None,
            'resolution': resolution,
            'original_filename': file.filename
        })

    except Exception as e:
        logger.error(f"Error during upscaling: {e}", exc_info=True)
        return jsonify({'error': f'Error during upscaling: {str(e)}'}), 500
    
    finally:
        # Clean up temporary files in production
        # Comment out the next line if you want to keep uploaded files
        # cleanup_temp_files(temp_lr_path)
        pass

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

@app.after_request
def after_request_func(response):
    """Add headers after each request."""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Handle video streaming
    if response.status_code == 206 and request.path.endswith('.mp4'):
        response.headers['Connection'] = 'keep-alive'
        response.headers['Accept-Ranges'] = 'bytes'
    
    # CORS headers for development (remove in production or configure properly)
    if app.debug:
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    
    return response

if __name__ == '__main__':
    # Use environment variables for configuration
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug_mode})")
    app.run(host=host, port=port, debug=debug_mode)