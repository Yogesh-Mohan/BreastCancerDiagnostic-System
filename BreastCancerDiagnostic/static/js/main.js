// Bootstrap 5 Form Validation and Clinical Range warnings
document.addEventListener('DOMContentLoaded', function() {
    
    // 1. Fetch form for prognosis analysis
    const form = document.getElementById('predictForm');
    if (form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    }

    // 2. Clinical Bounds Checker for typed input fields
    const bounds = {
        'radius_mean': { min: 5.0, max: 30.0, label: 'Radius Mean' },
        'texture_mean': { min: 5.0, max: 45.0, label: 'Texture Mean' },
        'perimeter_mean': { min: 35.0, max: 200.0, label: 'Perimeter Mean' },
        'area_mean': { min: 100.0, max: 2600.0, label: 'Area Mean' },
        'smoothness_mean': { min: 0.04, max: 0.20, label: 'Smoothness Mean' },
        'compactness_mean': { min: 0.01, max: 0.40, label: 'Compactness Mean' },
        'concavity_mean': { min: 0.0, max: 0.50, label: 'Concavity Mean' },
        'concave_points_mean': { min: 0.0, max: 0.25, label: 'Concave Points Mean' },
        'symmetry_mean': { min: 0.08, max: 0.35, label: 'Symmetry Mean' },
        'fractal_dimension_mean': { min: 0.03, max: 0.12, label: 'Fractal Dimension Mean' }
    };

    // Attach real-time bounds warnings to prevent typing mistakes
    Object.keys(bounds).forEach(function(id) {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('input', function() {
                const val = parseFloat(input.value);
                const bound = bounds[id];
                
                // Clear existing custom warnings if any
                let warningEl = document.getElementById(id + '-warning');
                if (warningEl) {
                    warningEl.remove();
                }

                if (!isNaN(val)) {
                    if (val < bound.min || val > bound.max) {
                        // Create warning badge below input
                        const warning = document.createElement('div');
                        warning.id = id + '-warning';
                        warning.className = 'text-warning small mt-1';
                        warning.innerHTML = `<i class="bi bi-exclamation-circle-fill"></i> Warning: Typical values for ${bound.label} are between ${bound.min} and ${bound.max}. Check for typos.`;
                        input.parentNode.appendChild(warning);
                    }
                }
            });
        }
    });

});
