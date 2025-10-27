from flask import Flask, request, render_template
import os
import sys
import traceback

# Debug: Check if files exist before importing
print("=== DEBUG: Checking file structure ===")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current dir: {os.listdir('.')}")

try:
    print("Checking src directory...")
    if os.path.exists('src'):
        print("src exists!")
        print(f"Files in src: {os.listdir('src')}")
    else:
        print("src directory NOT FOUND!")
        
    # Now try imports
    from src.logger import logging
    from src.pipeline.predict_pipeline import CustomData, PredictPipeline
    print("✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    # Create dummy classes if imports fail
    class CustomData:
        def get_data_as_data_frame(self): return None
    class PredictPipeline:
        def predict(self, data): return [0]

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/health')
def health():
    return "OK"

@application.route('/debug')
def debug():
    import os
    info = f"""
    Current dir: {os.getcwd()}<br>
    Files: {os.listdir('.')}<br>
    """
    if os.path.exists('src'):
        info += f"src files: {os.listdir('src')}<br>"
    else:
        info += "src folder NOT FOUND<br>"
    return info

@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            prediction_value = float(results[0])

            return render_template('home.html', results=prediction_value)

        except Exception as e:
            return f"Prediction error: {str(e)}<br>{traceback.format_exc()}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    application.run(host="0.0.0.0", port=port, debug=False)