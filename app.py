from flask import Flask, render_template, request, jsonify
import numpy as np
from inference import analyze_vulnerability

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    code = request.form['code']
    result = analyze_vulnerability(code)
    return render_template('result.html', result=result)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    code = request.json['code']
    result = analyze_vulnerability(code)
    
    # Convert result to a JSON-serializable format
    serializable_result = {
        'is_vulnerable': bool(result['is_vulnerable']),
        'vulnerability_type': result['vulnerability_type'],
        'severity': result['severity'],
        'confidence': float(result['confidence']),
        'repair_suggestion': result['repair_suggestion']
    }
    
    return jsonify(serializable_result)


