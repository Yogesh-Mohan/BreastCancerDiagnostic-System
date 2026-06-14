import os
import pickle
import sqlite3
import csv
import io
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, make_response
import numpy as np

# ReportLab imports for professional PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = Flask(__name__)
app.secret_key = 'breast_cancer_diagnostic_secret_key'

DATABASE = 'predictions.db'
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

# Load scaler and model
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Machine learning model and scaler loaded successfully.")
else:
    model = None
    scaler = None
    print("Warning: model.pkl or scaler.pkl not found. Run train_model.py first.")

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS patient_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            patient_name TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            radius_mean REAL,
            texture_mean REAL,
            perimeter_mean REAL,
            area_mean REAL,
            smoothness_mean REAL,
            compactness_mean REAL,
            concavity_mean REAL,
            concave_points_mean REAL,
            symmetry_mean REAL,
            fractal_dimension_mean REAL,
            prediction TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB at startup
init_db()

@app.route('/')
def index():
    # Fetch summary stats for home dashboard
    conn = get_db_connection()
    total_records = conn.execute('SELECT COUNT(*) FROM patient_predictions').fetchone()[0]
    malignant_count = conn.execute("SELECT COUNT(*) FROM patient_predictions WHERE prediction = 'Malignant'").fetchone()[0]
    benign_count = conn.execute("SELECT COUNT(*) FROM patient_predictions WHERE prediction = 'Benign'").fetchone()[0]
    
    # Get last 5 predictions
    recent_records = conn.execute('SELECT * FROM patient_predictions ORDER BY id DESC LIMIT 5').fetchall()
    conn.close()
    
    return render_template('index.html', 
                           total_records=total_records, 
                           malignant_count=malignant_count, 
                           benign_count=benign_count,
                           recent_records=recent_records)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get patient details
        patient_name = request.form.get('patient_name', '').strip()
        patient_id = request.form.get('patient_id', '').strip()
        
        # Validation checks
        if not patient_name or not patient_id:
            flash("Patient Name and ID are required.", "danger")
            return redirect(url_for('predict'))
            
        # Feature list
        features = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
        ]
        
        feature_vals = {}
        input_list = []
        
        try:
            for feature in features:
                val = float(request.form.get(feature, 0))
                if val <= 0:
                    flash(f"Feature '{feature.replace('_', ' ').title()}' must be a positive number.", "danger")
                    return redirect(url_for('predict'))
                feature_vals[feature] = val
                input_list.append(val)
        except ValueError:
            flash("All cell feature inputs must be numerical.", "danger")
            return redirect(url_for('predict'))
            
        if model is None or scaler is None:
            flash("Model or Scaler not loaded. Train the models first.", "danger")
            return redirect(url_for('predict'))
            
        # Inference
        input_arr = np.array(input_list).reshape(1, -1)
        input_scaled = scaler.transform(input_arr)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]
        
        prediction_label = "Malignant" if pred == 1 else "Benign"
        confidence_score = float(prob[1] if pred == 1 else prob[0]) * 100
        
        # Save to DB
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patient_predictions (
                patient_name, patient_id, radius_mean, texture_mean, perimeter_mean, area_mean,
                smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
                symmetry_mean, fractal_dimension_mean, prediction, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_name, patient_id,
            feature_vals['radius_mean'], feature_vals['texture_mean'], feature_vals['perimeter_mean'], feature_vals['area_mean'],
            feature_vals['smoothness_mean'], feature_vals['compactness_mean'], feature_vals['concavity_mean'], feature_vals['concave_points_mean'],
            feature_vals['symmetry_mean'], feature_vals['fractal_dimension_mean'],
            prediction_label, confidence_score
        ))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        return redirect(url_for('result', record_id=record_id))
        
    return render_template('predict.html')

@app.route('/result/<int:record_id>')
def result(record_id):
    conn = get_db_connection()
    record = conn.execute('SELECT * FROM patient_predictions WHERE id = ?', (record_id,)).fetchone()
    conn.close()
    
    if not record:
        flash("Record not found.", "danger")
        return redirect(url_for('index'))
        
    return render_template('result.html', record=record)

@app.route('/download_pdf/<int:record_id>')
def download_pdf(record_id):
    conn = get_db_connection()
    record = conn.execute('SELECT * FROM patient_predictions WHERE id = ?', (record_id,)).fetchone()
    conn.close()
    
    if not record:
        return "Record not found", 404
        
    # Generate PDF in-memory
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#0d6efd'),
        spaceAfter=15
    )
    section_style = ParagraphStyle(
        'SecHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor('#212529'),
        spaceBefore=10,
        spaceAfter=8
    )
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.gray,
        spaceBefore=20,
        leading=10
    )
    
    story = []
    
    # Header Title
    story.append(Paragraph("Breast Cancer Diagnostic Analysis Report", title_style))
    story.append(Spacer(1, 10))
    
    # Patient Metadata
    meta_html = f"""
    <b>Patient Name:</b> {record['patient_name']}<br/>
    <b>Patient ID:</b> {record['patient_id']}<br/>
    <b>Date of Analysis:</b> {record['timestamp']}<br/>
    <b>Report ID:</b> BCDS-{record['id']:06d}
    """
    story.append(Paragraph(meta_html, meta_style))
    story.append(Spacer(1, 15))
    
    # Diagnosis Results Box
    color_hex = '#dc3545' if record['prediction'] == 'Malignant' else '#198754'
    result_text = f"<b>DIAGNOSIS PROGNOSIS:</b> {record['prediction'].upper()}"
    confidence_text = f"<b>CONFIDENCE LEVEL:</b> {record['confidence']:.2f}%"
    
    result_style = ParagraphStyle(
        'ResultBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.white,
        alignment=1 # Centered
    )
    
    box_data = [
        [Paragraph(result_text, result_style)],
        [Paragraph(confidence_text, result_style)]
    ]
    box_table = Table(box_data, colWidths=[540])
    box_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(color_hex)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(color_hex)),
    ]))
    
    story.append(Paragraph("Diagnostic Summary", section_style))
    story.append(box_table)
    story.append(Spacer(1, 15))
    
    # Feature Metrics Table
    story.append(Paragraph("Measured Cell Nuclei Mean Features", section_style))
    
    features_list = [
        ("Radius Mean", f"{record['radius_mean']:.4f} \u03bcm"),
        ("Texture Mean", f"{record['texture_mean']:.4f} Gray-scale value"),
        ("Perimeter Mean", f"{record['perimeter_mean']:.4f} \u03bcm"),
        ("Area Mean", f"{record['area_mean']:.2f} \u03bcm\u00b2"),
        ("Smoothness Mean", f"{record['smoothness_mean']:.4f}"),
        ("Compactness Mean", f"{record['compactness_mean']:.4f}"),
        ("Concavity Mean", f"{record['concavity_mean']:.4f}"),
        ("Concave Points Mean", f"{record['concave_points_mean']:.4f}"),
        ("Symmetry Mean", f"{record['symmetry_mean']:.4f}"),
        ("Fractal Dimension Mean", f"{record['fractal_dimension_mean']:.4f}")
    ]
    
    table_data = [["Feature Parameter", "Measured Value"]]
    for item in features_list:
        table_data.append([item[0], item[1]])
        
    metrics_table = Table(table_data, colWidths=[270, 270])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0,0), (1,0), colors.HexColor('#212529')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fcfcfc')])
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 15))
    
    # Disclaimer
    disclaimer_html = """
    <b>Medical Disclaimer:</b> This report is generated programmatically by the Breast Cancer Diagnostic System (BCDS) 
    using a machine learning model trained on historical clinical datasets. This system is designed as an educational and 
    assistive triage tool. It does not replace clinical consultation, professional pathology review, or formal oncological 
    diagnosis. All diagnoses should be verified through gold-standard clinical biopsy procedures.
    """
    story.append(Paragraph(disclaimer_html, disclaimer_style))
    
    # Build Document
    doc.build(story)
    pdf_buffer.seek(0)
    
    filename = f"BCDS_Report_{record['patient_id']}.pdf"
    return send_file(pdf_buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')

@app.route('/admin')
def admin():
    search = request.args.get('search', '').strip()
    prediction_filter = request.args.get('prediction', '').strip()
    
    query = 'SELECT * FROM patient_predictions WHERE 1=1'
    params = []
    
    if search:
        query += ' AND (patient_name LIKE ? OR patient_id LIKE ?)'
        params.append(f'%{search}%')
        params.append(f'%{search}%')
        
    if prediction_filter:
        query += ' AND prediction = ?'
        params.append(prediction_filter)
        
    query += ' ORDER BY id DESC'
    
    conn = get_db_connection()
    records = conn.execute(query, params).fetchall()
    conn.close()
    
    return render_template('admin.html', records=records, search=search, prediction_filter=prediction_filter)

@app.route('/admin/export')
def export_csv():
    search = request.args.get('search', '').strip()
    prediction_filter = request.args.get('prediction', '').strip()
    
    query = 'SELECT * FROM patient_predictions WHERE 1=1'
    params = []
    
    if search:
        query += ' AND (patient_name LIKE ? OR patient_id LIKE ?)'
        params.append(f'%{search}%')
        params.append(f'%{search}%')
        
    if prediction_filter:
        query += ' AND prediction = ?'
        params.append(prediction_filter)
        
    query += ' ORDER BY id DESC'
    
    conn = get_db_connection()
    records = conn.execute(query, params).fetchall()
    conn.close()
    
    # Create CSV in-memory
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    
    # Header
    writer.writerow([
        'ID', 'Timestamp', 'Patient Name', 'Patient ID',
        'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean',
        'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean',
        'Symmetry Mean', 'Fractal Dimension Mean', 'Prediction', 'Confidence %'
    ])
    
    # Data rows
    for r in records:
        writer.writerow([
            r['id'], r['timestamp'], r['patient_name'], r['patient_id'],
            r['radius_mean'], r['texture_mean'], r['perimeter_mean'], r['area_mean'],
            r['smoothness_mean'], r['compactness_mean'], r['concavity_mean'], r['concave_points_mean'],
            r['symmetry_mean'], r['fractal_dimension_mean'], r['prediction'], f"{r['confidence']:.2f}"
        ])
        
    response = make_response(csv_buffer.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=bcds_prediction_history.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
