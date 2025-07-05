import os
from flask import Blueprint, request, jsonify, current_app, render_template
from werkzeug.utils import secure_filename
from app.models.equipment import Equipment
from app.models.hunter import Hunter
from app.services.ocr import OCRService
from app.services.calculator import CalculatorService
from app import db

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@bp.route('/api/equipment/upload', methods=['POST'])
def upload_equipment():
    """Upload equipment image and extract stats."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract stats from image
            stats = OCRService.extract_stats(filepath)
            
            # Create equipment record
            equipment = Equipment.from_dict(stats)
            db.session.add(equipment)
            db.session.commit()
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(equipment.to_dict()), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@bp.route('/api/equipment', methods=['GET'])
def list_equipment():
    """List all equipment."""
    equipment = Equipment.query.all()
    return jsonify([e.to_dict() for e in equipment])

@bp.route('/api/hunter', methods=['POST'])
def create_hunter():
    """Create a new hunter."""
    data = request.json
    hunter = Hunter.from_dict(data)
    db.session.add(hunter)
    db.session.commit()
    return jsonify(hunter.to_dict()), 201

@bp.route('/api/calculate/optimal', methods=['POST'])
def calculate_optimal():
    """Calculate optimal equipment build."""
    data = request.json
    
    # Get hunter
    hunter = Hunter.query.get(data['hunter_id'])
    if not hunter:
        return jsonify({'error': 'Hunter not found'}), 404
    
    # Get available equipment
    available_equipment = {}
    for equip_type in data['equipment_types']:
        available_equipment[equip_type] = Equipment.query.filter_by(type=equip_type).all()
    
    # Calculate optimal build
    result = CalculatorService.find_optimal_build(
        hunter=hunter,
        available_equipment=available_equipment,
        weights=data['stat_weights'],
        max_items_per_type=data['max_items_per_type']
    )
    
    return jsonify(result)

@bp.route('/api/calculate/compare', methods=['POST'])
def compare_builds():
    """Compare two equipment builds."""
    data = request.json
    
    # Get hunter
    hunter = Hunter.query.get(data['hunter_id'])
    if not hunter:
        return jsonify({'error': 'Hunter not found'}), 404
    
    # Get equipment for both builds
    build1 = [Equipment.query.get(id) for id in data['build1_ids']]
    build2 = [Equipment.query.get(id) for id in data['build2_ids']]
    
    if None in build1 or None in build2:
        return jsonify({'error': 'Equipment not found'}), 404
    
    # Compare builds
    result = CalculatorService.compare_builds(hunter, build1, build2)
    return jsonify(result) 