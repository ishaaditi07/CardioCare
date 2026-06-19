import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="*",
    database="heart_disease_db"
)
cursor = db.cursor()

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------------
# CATEGORY MAPS
# ----------------------------------------------------------------------
# Your CSV's sex/cp/fbs/restecg/exang/slope/thal columns are already
# numeric (0,1,2...), NOT text. That's why the old LabelEncoder step
# silently did nothing for them (it only fires on dtype == 'object').
# These dicts convert the human-readable text the HTML form sends
# into the SAME numeric codes used in the standard Cleveland dataset.
#
# IMPORTANT: check the printed "unique values" output below against
# this mapping the first time you run this. If your specific CSV uses
# a different numbering (e.g. cp as 1-4 instead of 0-3), adjust the
# dict to match your data, not the other way around.
CATEGORY_MAPS = {
    'sex':     {'female': 0, 'male': 1},
    'cp':      {'typical angina': 0, 'atypical angina': 1,
                'non-anginal': 2, 'asymptomatic': 3},
    'fbs':     {'false': 0, 'true': 1},
    'restecg': {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2},
    'exang':   {'false': 0, 'true': 1},
    'slope':   {'upsloping': 0, 'flat': 1, 'downsloping': 2},
    'thal':    {'normal': 0, 'fixed defect': 1, 'reversable defect': 2},
}

# ----------------------------------------------------------------------
# LOAD + CLEAN DATA
# ----------------------------------------------------------------------
df = pd.read_csv("C:/Users/isha0/Desktop/heart/Heart_disease_cleveland_new.csv")
print("Columns:", list(df.columns))

if 'dataset' in df.columns:
    df = df[df['dataset'] == 'Cleveland']

df.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in df.columns]

# Diagnostic: show what's ACTUALLY in each categorical column so you can
# confirm it lines up with CATEGORY_MAPS above.
for col in CATEGORY_MAPS:
    if col in df.columns:
        print(f"{col} unique values in CSV: {sorted(df[col].dropna().unique().tolist())}")

X = df.drop('target', axis=1)
y = df['target']

X = X.fillna(X.mode().iloc[0])
y = y.fillna(y.mode()[0])

FEATURE_COLUMNS = list(X.columns)  # exact order the model was trained on

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

clf = LogisticRegression(solver='lbfgs', max_iter=300)
clf.fit(X_res, y_res)

joblib.dump(clf, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

model = clf  # no need to round-trip through joblib right after saving


@app.route('/')
def home():
    return "Heart Disease Prediction API running"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    # Make sure every feature the model needs is present
    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    encoded = {}
    for col in FEATURE_COLUMNS:
        raw_value = data[col]

        if col in CATEGORY_MAPS:
            key = str(raw_value).strip().lower()
            if key not in CATEGORY_MAPS[col]:
                return jsonify({
                    "error": f"Invalid value '{raw_value}' for {col}. "
                             f"Allowed: {list(CATEGORY_MAPS[col].keys())}"
                }), 400
            encoded[col] = CATEGORY_MAPS[col][key]
        else:
            try:
                encoded[col] = float(raw_value)
            except (TypeError, ValueError):
                return jsonify({"error": f"Invalid numeric value for {col}: {raw_value}"}), 400

    # Build the row in the exact column order the model was trained on
    input_df = pd.DataFrame([[encoded[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    severity = int(prediction[0])

    # Store the original human-readable values in the DB (columns are VARCHAR)
    sql = """
    INSERT INTO predictions (
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal,
        predicted_severity
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    values = (
        data['age'], data['sex'], data['cp'],
        data['trestbps'], data['chol'], data['fbs'],
        data['restecg'], data['thalach'], data['exang'],
        data['oldpeak'], data['slope'], data['ca'],
        data['thal'], severity
    )

    try:
        db.ping(reconnect=True, attempts=1, delay=0)
        cursor.execute(sql, values)
        db.commit()
    except mysql.connector.Error as db_err:
        db.rollback()
        return jsonify({
            "predicted_severity": severity,
            "warning": f"Prediction succeeded but saving to DB failed: {db_err}"
        })

    return jsonify({'predicted_severity': severity})


if __name__ == '__main__':
    app.run(debug=True)