# -*- coding: utf-8 -*-

!pip install -q google-generativeai pandas matplotlib

import pandas as pd
import numpy as np

import google.auth
import google.generativeai as genai
import matplotlib.pyplot as plt

# Paste your API key here
genai.configure(api_key="AIzaSyAYdqevWbiZpTVpQnXBlMWuLwjGtpqhAJY")

# Load the Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

def call_llm_api(prompt, model_name='gemini-2.0-flash'):
    """Calls the Gemini API to get a text response."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "LLM Error"

from google.colab import files

d1 = files.upload()
patient_df = pd.read_csv('patients.csv')

d1 = files.upload()
condition_df = pd.read_csv('conditions.csv')

d1 = files.upload()
observation_df = pd.read_csv('observations.csv')

patient_df

condition_df

observation_df

patient_df = pd.read_csv('patients.csv')
  condition_df = pd.read_csv('conditions.csv')
  observation_df = pd.read_csv('observations.csv')

  diabetes_conditions = condition_df[condition_df['DESCRIPTION'].str.contains("diabetes", case=False)]
merged = pd.merge(diabetes_conditions, patient_df, left_on="PATIENT", right_on="Id")
sample = merged.sample(5)

def format_patient(row):
    return (
        f"Age: {row['BIRTHDATE']}, Gender: {row['GENDER']}, "
        f"Condition: Diabetes, Weight: {row.get('WEIGHT', 'N/A')}, BMI: {row.get('BMI', 'N/A')}"
    )

"""## In-Context Learning (Zero-shot, One-shot, Few-shot)"""

# Zero-shot
zero_shot_prompt = "Assess diabetes risk: Age: 52, BMI: 31.5, Family history: Yes, A1C: 6.8%"

# One-shot
one_shot_prompt = (
    "Patient: Age: 65, BMI: 33.2, A1C: 7.5%, Family History: Yes\n"
    "Response: High risk due to age, high BMI, and elevated A1C.\n\n"
    "Patient: Age: 52, BMI: 31.5, A1C: 6.8%, Family History: Yes\n"
    "Response:"
)

# Few-shot
few_shot_prompt = (
    "Evaluate diabetes risk based on patient data:\n"
    "Patient 1: Age: 60, BMI: 35.0, A1C: 8.1% — High risk.\n"
    "Patient 2: Age: 45, BMI: 25.2, A1C: 5.4% — Low risk.\n"
    "Patient 3: Age: 52, BMI: 31.5, A1C: 6.8% —"
)

print("Zero-shot:\n", model.generate_content(zero_shot_prompt).text)
print("\nOne-shot:\n", model.generate_content(one_shot_prompt).text)
print("\nFew-shot:\n", model.generate_content(few_shot_prompt).text)

"""## Chain of Thought Reasoning"""

cot_prompt = (
    "Analyze patient step by step for diabetes risk:\n\n"
    "Patient: Age: 52, BMI: 31.5, A1C: 6.8%, Family History: Yes\n"
    "Step 1: Age is above 50 → moderate risk.\n"
    "Step 2: BMI is over 30 → high risk.\n"
    "Step 3: A1C is close to diabetic range.\n"
    "Step 4: Family history confirms additional risk.\n"
    "Conclusion:"
)
print(model.generate_content(cot_prompt).text)

"""## Tree of Thought Prompting"""

tot_prompt = '''
Evaluate diabetes risk through a decision tree:

START
-> Check A1C: 6.8%
   -> Borderline high → go to BMI
      -> BMI: 31.5 → High
         -> Go to family history
            -> Family History: Yes → High Risk
END

Based on this path, conclude the patient's diabetes risk and explain why each node led to that conclusion.
'''
print(model.generate_content(tot_prompt).text)

"""## Combined Prompting Method"""

combined_prompt = (
    "Assess diabetes risk by combining examples and tree reasoning.\n"
    "Example: Patient 1: Age: 65, BMI: 33.2, A1C: 7.5% — High risk.\n"
    "Example: Patient 2: Age: 45, BMI: 24.8, A1C: 5.1% — Low risk.\n"
    "\nNow, for this new patient:\n"
    "Patient: Age: 58, BMI: 30.2, A1C: 6.9%, Family History: Yes\n"
    "Use step-by-step logic and decision tree structure to reach a conclusion:"
)
print(model.generate_content(combined_prompt).text)

"""## Visualize Risk Classification"""

risk_data = {"Low": 5, "Moderate": 7, "High": 12}
plt.bar(risk_data.keys(), risk_data.values(), color=["#66bb6a", "#ffc107", "#ef5350"])
plt.title("Diabetes Risk Classification Output")
plt.ylabel("Number of Patients")
plt.xlabel("Risk Level")
plt.grid(axis='y', alpha=0.75)
plt.show()