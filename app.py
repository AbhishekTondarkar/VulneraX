from flask import Flask, render_template, request, jsonify
from models.career_model import CareerModel
import json

app = Flask(__name__)
career_model = CareerModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fields')
def fields():
    with open('data/fields.json', 'r') as f:
        fields = json.load(f)
    return render_template('fields.html', fields=fields)

@app.route('/skills/<field>')
def skills(field):
    with open('data/skills.json', 'r') as f:
        skills = json.load(f)
    return render_template('skills.html', field=field, skills=skills[field])

@app.route('/analyze', methods=['POST'])
def analyze():
    skills = request.json['skills']
    field = request.json['field']
    
    print(f"Received field: {field}, skills: {skills}")  # Debug print

    recommendations = career_model.get_recommendations(field, skills)

    print(f"Recommendations found: {recommendations}")  # Debug print

    return jsonify(recommendations)

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

if __name__ == '__main__':
    app.run(debug=True)
