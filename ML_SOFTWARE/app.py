import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from werkzeug.utils import secure_filename

import re
from flask import Flask, render_template

# Create the custom filter function
def strip_html(value):
    return re.sub(r'<[^>]*>', '', value)

# Initialize Flask app
app = Flask(__name__)

# Register the custom filter
app.jinja_env.filters['strip_html'] = strip_html
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Route: Home (File Upload)
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('display_dataset', filename=filename))
        else:
            return "Please upload a valid CSV file."
    return render_template('index.html')

# Route: Display Dataset
@app.route('/display/<filename>')
def display_dataset(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
        data_html = data.head(10).to_html(index=False)  # Drop the index here
        return render_template('display.html', tables=[data_html], titles=data.columns.values)
    except Exception as e:
        return str(e)


# Route: Dataset Statistics
@app.route('/stats/<filename>')
def dataset_stats(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
        stats = data.describe(include='all').transpose()
        return render_template('stats.html', stats=stats)
    except Exception as e:
        return str(e)


# Route: Model Selection
@app.route('/select_model/<filename>', methods=['GET', 'POST'])
def select_model(filename):
    if request.method == 'POST':
        selected_model = request.form['model']
        return redirect(url_for('evaluate_model', filename=filename, model=selected_model))
    return render_template('select_model.html', filename=filename)

# Route: Evaluate Model
@app.route('/evaluate_model/<filename>/<model>', methods=['GET'])
def evaluate_model(filename, model):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1]  # Features
        y = data.iloc[:, -1]   # Target
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(),
            "Logistic Regression": LogisticRegression()
        }

        selected_model = models[model]
        selected_model.fit(X_train, y_train)
        y_pred = selected_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Pass the processed results to the template
        return render_template(
            'results.html',
            model=model,
            accuracy=accuracy,
            classification_report=report,
            conf_matrix=conf_matrix
        )
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
