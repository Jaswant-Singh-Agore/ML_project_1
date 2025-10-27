🎯 Student Performance Prediction — End-to-End ML Project

A complete Machine Learning web application that predicts a student’s Math Score based on demographic and academic details.
This project covers everything from data ingestion → model training → web deployment.

🧠 About the Project

This project was built to understand how a real-world ML system works — from preparing data and training multiple models to finally serving predictions through a Flask app.

You can input:

Gender

Race/Ethnicity

Parental Education Level

Lunch Type

Test Preparation Course

Reading & Writing Scores

And it will predict the expected Math Score for that student.

⚙️ Tech Stack
Category	Tools
Language	Python 3.11
Framework	Flask
Libraries	Pandas, NumPy, Scikit-learn, CatBoost, XGBoost
Server (Deployment)	Render / AWS Elastic Beanstalk
Version Control	Git & GitHub
🧩 Project Structure
📂 ML_Project
├── application.py          # Flask web app
├── requirements.txt        # Dependencies
├── Procfile                # For production server
├── artifacts/              # Saved model & preprocessor
├── src/                    # All ML pipeline components
│   ├── components/         # Data ingestion, transformation, model trainer
│   ├── pipeline/           # Training & prediction scripts
│   ├── utils.py, logger.py, exception.py
│
├── templates/              # HTML frontend
│   ├── home.html
│   └── index.html
└── README.md

🧾 How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/Jaswant-Singh-Agore/Student_perfromance_predictor.git

2️⃣ Create Virtual Environment
python -m venv myenv
myenv\Scripts\activate    # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train Model (optional)
python -m src.pipeline.train_pipeline

5️⃣ Run Flask App
python application.py


Go to 👉 http://127.0.0.1:5000/

🌐 Deployment

The project can be deployed easily on:

Render – simple, free-tier hosting for Flask

AWS Elastic Beanstalk – scalable production hosting

(Contains working Procfile and .config for server setup.)

📊 Model Performance

Best Model: CatBoost Regressor

R² Score: ~0.85

Metric Used: R², MAE, RMSE

🧰 Future Improvements

REST API version

Better UI with charts

Cloud-based dataset storage

👨‍💻 Author

Jaswant Singh
💼 Machine Learning Engineer (Fresher) | ML & Deep Learning | Python, SQL, MLOps (Docker/AWS) | Model Deployment

🔗 LinkedIn Profile
https://www.linkedin.com/in/jaswant-singh-agore/

⭐ Support

If you find this project useful, please consider giving it a ⭐ star on GitHub.
It helps others discover it and motivates me to build more cool projects!

🏁 Final Note

This project was coded, debugged, and deployed entirely from scratch — no prebuilt templates.
It reflects both technical skill and practical understanding of ML deployment.