import urllib.request
import urllib.parse
import urllib.error
import sqlite3

base_url = "http://127.0.0.1:5000"

# 1. Test Home page
print("Testing Home page...")
try:
    response = urllib.request.urlopen(base_url)
    html = response.read().decode('utf-8')
    assert response.status == 200
    assert "OncoShield" in html
    print("Home page loaded successfully.")
except Exception as e:
    print(f"Home page failed: {e}")

# 2. Submit Benign prognosis
print("\nSubmitting Benign prognosis for Alice Smith...")
data_benign = urllib.parse.urlencode({
    'patient_name': 'Alice Smith',
    'patient_id': 'P-1001',
    'radius_mean': '11.0',
    'texture_mean': '14.5',
    'perimeter_mean': '72.0',
    'area_mean': '380.0',
    'smoothness_mean': '0.085',
    'compactness_mean': '0.065',
    'concavity_mean': '0.03',
    'concave_points_mean': '0.02',
    'symmetry_mean': '0.17',
    'fractal_dimension_mean': '0.06'
}).encode('utf-8')

record_id = None
try:
    class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
            
    opener = urllib.request.build_opener(NoRedirectHandler)
    req = urllib.request.Request(f"{base_url}/predict", data=data_benign, method='POST')
    try:
        res = opener.open(req)
        redirect_url = res.headers.get('Location')
    except urllib.error.HTTPError as e:
        if e.code == 302:
            redirect_url = e.headers.get('Location')
        else:
            raise e
            
    print(f"Redirect location: {redirect_url}")
    assert "/result/" in redirect_url
    
    if not redirect_url.startswith('http'):
        redirect_url = base_url + redirect_url
        
    record_id = int(redirect_url.split('/')[-1])
    print(f"Created record ID: {record_id}")
    
    # Fetch result page
    res_page = urllib.request.urlopen(redirect_url)
    html_page = res_page.read().decode('utf-8')
    assert "Benign Tumor" in html_page
    print("Benign prognosis verified successfully!")
except Exception as e:
    print(f"Benign prognosis failed: {e}")

# 3. Submit Malignant prognosis
print("\nSubmitting Malignant prognosis for Bob Jones...")
data_malignant = urllib.parse.urlencode({
    'patient_name': 'Bob Jones',
    'patient_id': 'P-1002',
    'radius_mean': '21.5',
    'texture_mean': '24.0',
    'perimeter_mean': '145.0',
    'area_mean': '1450.0',
    'smoothness_mean': '0.12',
    'compactness_mean': '0.21',
    'concavity_mean': '0.28',
    'concave_points_mean': '0.14',
    'symmetry_mean': '0.23',
    'fractal_dimension_mean': '0.075'
}).encode('utf-8')

try:
    req = urllib.request.Request(f"{base_url}/predict", data=data_malignant, method='POST')
    try:
        res = opener.open(req)
        redirect_url = res.headers.get('Location')
    except urllib.error.HTTPError as e:
        if e.code == 302:
            redirect_url = e.headers.get('Location')
        else:
            raise e
            
    print(f"Redirect location: {redirect_url}")
    assert "/result/" in redirect_url
    
    if not redirect_url.startswith('http'):
        redirect_url = base_url + redirect_url
        
    record_id_mal = int(redirect_url.split('/')[-1])
    print(f"Created record ID: {record_id_mal}")
    
    res_page = urllib.request.urlopen(redirect_url)
    html_page = res_page.read().decode('utf-8')
    assert "Malignant Tumor" in html_page
    print("Malignant prognosis verified successfully!")
except Exception as e:
    print(f"Malignant prognosis failed: {e}")

# 4. Check database
print("\nChecking SQLite database contents...")
try:
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row
    records = conn.execute('SELECT * FROM patient_predictions ORDER BY id DESC LIMIT 2').fetchall()
    for r in records:
        print(f"ID: {r['id']}, Patient: {r['patient_name']}, Diagnosis: {r['prediction']}, Confidence: {r['confidence']:.2f}%")
    conn.close()
except Exception as e:
    print(f"Database check failed: {e}")

# 5. Check PDF generation
if record_id:
    print("\nVerifying PDF download...")
    try:
        pdf_res = urllib.request.urlopen(f"{base_url}/download_pdf/{record_id}")
        pdf_data = pdf_res.read()
        print(f"PDF Size: {len(pdf_data)} bytes")
        assert pdf_res.status == 200
        assert pdf_data.startswith(b'%PDF')
        print("PDF download verified successfully!")
    except Exception as e:
        print(f"PDF download failed: {e}")
else:
    print("Skipping PDF check as record_id is undefined.")
