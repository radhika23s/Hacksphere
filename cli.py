#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HealthifyAI CLI Interface
For testing the model and making predictions
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import difflib
import warnings

warnings.filterwarnings("ignore")

# ========== Utility Functions ==========
def normalize(sym):
    return sym.strip().lower().replace(" ", "_")

# ========== Load Datasets ==========
print("Loading datasets...")
try:
    dataset = pd.read_csv("dataset.csv")
    severity = pd.read_csv("Symptom-severity.csv")
    description = pd.read_csv("symptom_Description.csv")
    precaution = pd.read_csv("symptom_precaution.csv")
    workout = pd.read_csv("gym dataset.csv")
    print("✅ All datasets loaded")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ========== Train Model ==========
symptom_cols = [col for col in dataset.columns if "Symptom_" in col]
dataset["Symptoms"] = dataset[symptom_cols].apply(
    lambda row: [normalize(s) for s in row.dropna().values], axis=1
)

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(dataset["Symptoms"])
y = dataset["Disease"]
known_symptoms = set(mlb.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%\n")

severity_dict = {
    normalize(sym): wt
    for sym, wt in zip(severity["Symptom"], severity["weight"])
}

def match_symptom(sym):
    matches = difflib.get_close_matches(sym, known_symptoms, n=1, cutoff=0.6)
    return matches[0] if matches else None

def predict_disease(user_symptoms):
    user_input = mlb.transform([user_symptoms])
    pred = clf.predict(user_input)[0]
    risk_score = sum(severity_dict.get(sym, 1) for sym in user_symptoms)
    try:
        desc = description.loc[description["Disease"] == pred, "Description"].values[0]
    except:
        desc = "No description available."
    try:
        prec = precaution.loc[
            precaution["Disease"] == pred,
            ["Precaution_1", "Precaution_2", "Precaution_3"]
        ].values[0]
        prec_str = ", ".join([p for p in prec if pd.notnull(p)])
    except:
        prec_str = "No specific precautions found."
    return pred, risk_score, desc, prec_str

def get_fitness_plan(age, gender, height_cm, weight_kg, goal, hypertension):
    sex = "Male" if str(gender).strip().lower() in ["male", "m"] else "Female"
    height_m = float(height_cm) / 100
    bmi = float(weight_kg) / (height_m ** 2)
    hyper_str = "Yes" if str(hypertension).strip().lower() == "yes" else "No"
    goal_map = {
        "lose weight": "Weight Loss",
        "gain muscle": "Weight Gain",
        "stay fit": "Weight Gain"
    }
    user_goal = str(goal).strip().lower()
    if user_goal == "stay fit":
        target_goal = "Weight Loss" if bmi >= 25 else "Weight Gain"
    else:
        target_goal = goal_map.get(user_goal, "Weight Gain")
    df_filtered = workout[
        (workout['Sex'] == sex) &
        (workout['Hypertension'] == hyper_str) &
        (workout['Fitness Goal'] == target_goal)
    ].copy()
    if df_filtered.empty:
        df_filtered = workout[
            (workout['Sex'] == sex) &
            (workout['Fitness Goal'] == target_goal)
        ].copy()
    if df_filtered.empty:
        df_filtered = workout[workout['Fitness Goal'] == target_goal].copy()
    if df_filtered.empty:
        return "No plan", "No exercises", "Consult professional"
    df_filtered['dist'] = (
        ((df_filtered['Age'] - age) / 100) ** 2 +
        ((df_filtered['BMI'] - bmi) / 50) ** 2
    )
    best = df_filtered.loc[df_filtered['dist'].idxmin()]
    return best.get('Diet', ''), best.get('Exercises', ''), best.get('Recommendation', '')

# ========== CLI Interface ==========
def main():
    print("=" * 60)
    print("Welcome to HealthifyAI - Health Prediction System")
    print("=" * 60)
    
    print("\n=== User Details ===")
    name = input("Name: ")
    try:
        age = int(input("Age: "))
        height_cm = float(input("Height (cm): "))
        weight_kg = float(input("Weight (kg): "))
    except ValueError:
        print("Invalid input, using defaults")
        age, height_cm, weight_kg = 25, 170.0, 70.0
    
    gender = input("Gender (Male/Female): ") or "Male"
    goal = input("Fitness Goal (Lose Weight / Gain Muscle / Stay Fit): ") or "Stay Fit"
    hypertension = input("Hypertension (yes/no): ") or "no"
    
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
    
    print("\n=== Symptoms ===")
    common = ["fever", "headache", "fatigue", "body pain", "cough", "nausea", "diarrhea"]
    symptoms = []
    
    for sym in common:
        ans = input(f"Do you have {sym}? (yes/no): ").strip().lower()
        if ans == "yes":
            symptoms.append(normalize(sym))
    
    other = input("Other symptoms (comma separated): ").strip()
    if other:
        for s in other.split(","):
            s_norm = normalize(s.strip())
            if s_norm in known_symptoms:
                symptoms.append(s_norm)
            else:
                m = match_symptom(s_norm)
                if m:
                    print(f"Using closest match: {m}")
                    symptoms.append(m)
    
    if not symptoms:
        print("No symptoms entered. Exiting.")
        return
    
    # Predict
    disease, risk, desc, precautions = predict_disease(symptoms)
    diet, exercises, recommendation = get_fitness_plan(age, gender, height_cm, weight_kg, goal, hypertension)
    
    # Report
    print("\n" + "=" * 60)
    print("HEALTH & WELLNESS REPORT")
    print("=" * 60)
    print(f"\n📋 PATIENT PROFILE")
    print(f"  Name: {name}")
    print(f"  Age: {age}")
    print(f"  Gender: {gender}")
    print(f"  BMI: {bmi} kg/m²")
    print(f"  Hypertension: {hypertension}")
    print(f"  Fitness Goal: {goal}")
    print(f"\n🔍 DISEASE DIAGNOSIS")
    print(f"  Predicted Disease: {disease}")
    print(f"  Risk Score: {risk}")
    print(f"  Description: {desc}")
    print(f"  Precautions: {precautions}")
    print(f"\n🥗 DIET PLAN")
    print(f"  {diet}")
    print(f"\n💪 WORKOUT")
    print(f"  {exercises}")
    print(f"\n✨ RECOMMENDATION")
    print(f"  {recommendation}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
