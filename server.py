from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from gradio_client import Client, handle_file
import tempfile
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Gradio client
print("Connecting to IDM-VTON model...")
client = Client("yisol/IDM-VTON")
print("✅ Connected successfully!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "message": "Virtual Try-On Proxy Server with Gradio Client",
        "endpoint": "/api/virtual-tryon"
    })

@app.route('/api/virtual-tryon', methods=['POST'])
def virtual_tryon():
    try:
        # Check if files are in request
        if 'person_image' not in request.files or 'clothing_image' not in request.files:
            return jsonify({"error": "Both person_image and clothing_image are required"}), 400
        
        person_file = request.files['person_image']
        clothing_file = request.files['clothing_image']
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded files temporarily
        person_path = os.path.join(temp_dir, secure_filename(person_file.filename))
        clothing_path = os.path.join(temp_dir, secure_filename(clothing_file.filename))
        
        person_file.save(person_path)
        clothing_file.save(clothing_path)
        
        print(f"Processing virtual try-on...")
        print(f"Person image: {person_path}")
        print(f"Clothing image: {clothing_path}")
        
        # Call Gradio API
        # The IDM-VTON Space expects these parameters
        result = client.predict(
            dict={"background": handle_file(person_path), "layers": [], "composite": None},
            garm_img=handle_file(clothing_path),
            garment_des="A beautiful garment",  # Description of the garment
            is_checked=True,  # Use auto-masking
            is_checked_crop=False,  # Don't use auto-crop
            denoise_steps=30,  # Number of denoising steps
            seed=42,  # Random seed for reproducibility
            api_name="/tryon"
        )
        
        print(f"✅ Result received: {result}")
        
        # Result is a tuple, first element is the output image path
        output_image_path = result[0]
        
        # Clean up input files
        try:
            os.remove(person_path)
            os.remove(clothing_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        # Send the result image
        return send_file(output_image_path, mimetype='image/png')
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "IDM-VTON",
        "backend": "Gradio Client"
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Virtual Try-On Server Starting...")
    print("="*50)
    print("📍 Server running on: http://localhost:3000")
    print("🎯 Endpoint: POST /api/virtual-tryon")
    print("🤖 Model: IDM-VTON (via Gradio Client)")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=3000, debug=True)
