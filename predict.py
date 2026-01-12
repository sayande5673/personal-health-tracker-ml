# src/predict.py
import os
import joblib
import numpy as np
import pandas as pd

FEATURES = [
    'Age',
    'Sex',
    'Weight_kg',
    'Height_cm',
    'Sleep_hour',
    'Rest_hour',
    'Activity_level',
    'Steps_per_day',
    'Stress_level',
    'bmr',
    'activity_factor'
]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "calorie_model.joblib")
FAT_MODEL_PATH = os.path.join(BASE_DIR, "models", "fatigue_model.joblib")


class Predictor:
    def __init__(self, cal_model_path=CAL_MODEL_PATH, fat_model_path=FAT_MODEL_PATH):
        self.cal_model = joblib.load(cal_model_path)
        self.fat_model = joblib.load(fat_model_path)

    def predict(self, row: dict):
      df = pd.DataFrame([row])

    # ---- feature engineering (MUST MATCH TRAINING) ----
      df['bmr'] = (
        10 * df['Weight_kg']
        + 6.25 * df['Height_cm']
        - 5 * df['Age']
        + np.where(df['Sex'] == 1, 5, -161)
    )

      df['activity_factor'] = np.select(
        [
            df['Activity_level'] == 0,
            df['Activity_level'] == 1,
            df['Activity_level'] == 2,
            df['Activity_level'] == 3
        ],
        [1.2, 1.375, 1.55, 1.725],
        default=1.2
    )

    # ---- enforce order ----
      df = df[FEATURES]

      cal = int(self.cal_model.predict(df)[0])
      fat_prob = float(self.fat_model.predict_proba(df)[0, 1])
      fat_label = int(fat_prob > 0.5)

      return {
        "calories_needed": cal,
        "fatigue_prob": fat_prob,
        "fatigue": fat_label
    }

if __name__ == '__main__':
   p = Predictor()
   sample = {'age': 30, 'sex':1, 'weight_kg':75,'height_cm':175,
'activity_level':2, 'sleep_hours':6.5, 'resting_hr':70,
'steps_per_day':7000, 'stress_level':0.3
   }
   print(p.predict(sample))
