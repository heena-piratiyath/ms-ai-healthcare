from google.colab import files
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# Load patients data
d1 = files.upload()
patients = pd.read_csv('PATIENTS.csv')

# Load admissions data
d2 = files.upload()
admissions = pd.read_csv('ADMISSIONS.csv')

# Load ICUSTAYS data
d3 = files.upload()
icustays = pd.read_csv('ICUSTAYS.csv')

# Load Prescriptions data
d4 = files.upload()
prescriptions = pd.read_csv('PRESCRIPTIONS.csv')

# Load D_LABITEMS data
d5 = files.upload()
d_labitems = pd.read_csv('D_LABITEMS.csv')

# Load LABEVENTS data
d6 = files.upload()
labevents = pd.read_csv('LABEVENTS.csv')

# Load PROCEDURES_ICD data
d7 = files.upload()
procedures_icd = pd.read_csv('PROCEDURES_ICD.csv')

# Load D_ICD_PROCEDURES data
d8 = files.upload()
d_icd_procedures = pd.read_csv('D_ICD_PROCEDURES.csv')

# Load DIAGNOSES_ICD data
d9 = files.upload()
diagnoses_icd = pd.read_csv('DIAGNOSES_ICD.csv')

"""**1. Patient Demographics**"""

import sqlite3

query = """
SELECT
  a.ethnicity,
  p.gender,
  CAST(COUNT(CASE WHEN a.diagnosis = 'SEPSIS' THEN 1 END) AS REAL) * 100 / COUNT(*) AS sepsis_percentage,
  AVG(STRFTIME('%Y', p.dod) - STRFTIME('%Y', p.dob)) AS average_age_at_death
FROM PATIENTS p
JOIN ADMISSIONS a ON p.subject_id = a.subject_id
GROUP BY a.ethnicity, p.gender
ORDER BY a.ethnicity, p.gender;
"""
# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

patients.to_sql('PATIENTS', conn, if_exists='replace', index=False)
admissions.to_sql('ADMISSIONS', conn, if_exists='replace', index=False)

result_df = pd.read_sql_query(query, conn)
print(result_df)

conn.close()

"""**2. Insurance Coverage**"""

query = """
SELECT a.insurance, COUNT(*) AS insurance_count
FROM ADMISSIONS a
WHERE a.diagnosis = 'SEPSIS'
GROUP BY a.insurance;
"""

# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')
admissions.to_sql('ADMISSIONS', conn, if_exists='replace', index=False)

result_df = pd.read_sql_query(query, conn)
print(result_df)

conn.close()

"""**3. Sepsis Patients with Pre-existing Conditions**"""

query = """
SELECT COUNT(DISTINCT p.subject_id) AS patients_with_preexisting_conditions
FROM PATIENTS p
WHERE p.subject_id IN (
    SELECT DISTINCT di.subject_id
    FROM DIAGNOSES_ICD di
    WHERE di.icd9_code IN ('25000', '4019', '2724', '41401') -- ICD-9 codes for Diabetes, Hypertension, Hyperlipidemia, and Coronary Artery Disease
)
AND p.subject_id IN (
    SELECT DISTINCT a.subject_id
    FROM ADMISSIONS a
    WHERE a.diagnosis = 'SEPSIS'
);
"""

# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')
patients.to_sql('PATIENTS', conn, if_exists='replace', index=False)
diagnoses_icd.to_sql('DIAGNOSES_ICD', conn, if_exists='replace', index=False)
admissions.to_sql('ADMISSIONS', conn, if_exists='replace', index=False)

result_df = pd.read_sql_query(query, conn)
print(result_df)

conn.close()

"""**4. Top 5 Prescribed Medications**"""

query = """
SELECT pr.drug_name_generic, COUNT(*) AS prescription_count
FROM PRESCRIPTIONS pr
JOIN ADMISSIONS a ON pr.hadm_id = a.hadm_id
WHERE a.diagnosis = 'SEPSIS'
GROUP BY pr.drug_name_generic
ORDER BY prescription_count DESC
LIMIT 5;
"""

# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

prescriptions.to_sql('PRESCRIPTIONS', conn, if_exists='replace', index=False)
admissions.to_sql('ADMISSIONS', conn, if_exists='replace', index=False)

result_df = pd.read_sql_query(query, conn)
print(result_df)

conn.close()

"""**5. Length of Stay Distribution**"""

query = """
SELECT
  MIN(i.los) AS min_los,
  MAX(i.los) AS max_los,
  AVG(i.los) AS average_los,
  AVG(CASE WHEN quartile = 1 THEN i.los END) AS q1_los,
  AVG(CASE WHEN quartile = 2 THEN i.los END) AS median_los,
  AVG(CASE WHEN quartile = 3 THEN i.los END) AS q3_los
FROM ICUSTAYS i
JOIN ADMISSIONS a ON i.hadm_id = a.hadm_id
JOIN (
    SELECT
        hadm_id,
        NTILE(4) OVER (ORDER BY los) AS quartile
    FROM ICUSTAYS
) AS quartiles ON i.hadm_id = quartiles.hadm_id
WHERE a.diagnosis = 'SEPSIS';
"""

# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

icustays.to_sql('ICUSTAYS', conn, if_exists='replace', index=False)
admissions.to_sql('ADMISSIONS', conn, if_exists='replace', index=False)

result_df = pd.read_sql_query(query, conn)
print(result_df)

conn.close()

"""**6. Readmission Rates**"""

query = """
SELECT
  CAST(COUNT(CASE WHEN admission_count > 1 THEN 1 END) AS REAL) * 100 / COUNT(*) AS readmission_percentage
FROM (
  SELECT p.subject_id, COUNT(a.hadm_id) AS admission_count
  FROM PATIENTS p
  JOIN ADMISSIONS a ON p.subject_id = a.subject_id
  WHERE a.diagnosis = 'SEPSIS'
  GROUP BY p.subject_id
);
"""

# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

patients.to_sql('PATIENTS', conn, if_exists='replace', index=False)
admissions.to_sql('ADMISSIONS', conn, if_exists='replace', index=False)

result_df = pd.read_sql_query(query, conn)
print(result_df)

conn.close()

"""**7. Top 5 Lab Tests Performed**"""

query = """
SELECT dli.label, COUNT(*) AS lab_test_count
FROM LABEVENTS le
JOIN D_LABITEMS dli ON le.itemid = dli.itemid
JOIN ADMISSIONS a ON le.hadm_id = a.hadm_id
WHERE a.diagnosis = 'SEPSIS'
GROUP BY dli.label
ORDER BY lab_test_count DESC
LIMIT 5;
"""

# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

labevents.to_sql('LABEVENTS', conn, if_exists='replace', index=False)
d_labitems.to_sql('D_LABITEMS', conn, if_exists='replace', index=False)
admissions.to_sql('ADMISSIONS', conn, if_exists='replace', index=False)

result_df = pd.read_sql_query(query, conn)
print(result_df)

conn.close()

"""**8. Mechanical Ventilation in Sepsis Patients**"""

query = """
SELECT
  SUM(CASE WHEN a.diagnosis = 'SEPSIS' THEN 1 ELSE 0 END) AS sepsis_patients,
  SUM(CASE WHEN a.diagnosis = 'SEPSIS' AND pm.short_title LIKE '%VENTILATION%' THEN 1 ELSE 0 END) AS ventilated_sepsis_patients,
  CAST(SUM(CASE WHEN a.diagnosis = 'SEPSIS' AND pm.short_title LIKE '%VENTILATION%' THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN a.diagnosis = 'SEPSIS' THEN 1 ELSE 0 END) AS ventilation_percentage
FROM ADMISSIONS a
LEFT JOIN PROCEDURES_ICD pi ON a.hadm_id = pi.hadm_id
LEFT JOIN D_ICD_PROCEDURES pm ON pi.icd9_code = pm.icd9_code;
"""

# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

procedures_icd.to_sql('PROCEDURES_ICD', conn, if_exists='replace', index=False)
d_icd_procedures.to_sql('D_ICD_PROCEDURES', conn, if_exists='replace', index=False)
admissions.to_sql('ADMISSIONS', conn, if_exists='replace', index=False)

result_df = pd.read_sql_query(query, conn)
print(result_df)

conn.close()

"""**9. Common Procedures Performed**"""

query = """
SELECT dip.short_title, COUNT(*) AS procedure_count
FROM PROCEDURES_ICD pi
JOIN D_ICD_PROCEDURES dip ON pi.icd9_code = dip.icd9_code
JOIN ADMISSIONS a ON pi.hadm_id = a.hadm_id
WHERE a.diagnosis = 'SEPSIS'
GROUP BY dip.short_title
ORDER BY procedure_count DESC
LIMIT 5;
"""

# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

procedures_icd.to_sql('PROCEDURES_ICD', conn, if_exists='replace', index=False)
d_icd_procedures.to_sql('D_ICD_PROCEDURES', conn, if_exists='replace', index=False)
admissions.to_sql('ADMISSIONS', conn, if_exists='replace', index=False)

result_df = pd.read_sql_query(query, conn)
print(result_df)

conn.close()

"""**10. Correlation between Length of Stay and Number of Medications**"""

query = """
SELECT
  (SUM(i.los * medication_count) - (SUM(i.los) * SUM(medication_count)) / COUNT(*)) /
  (SQRT(SUM(i.los * i.los) - (SUM(i.los) * SUM(i.los)) / COUNT(*)) * SQRT(SUM(medication_count * medication_count) - (SUM(medication_count) * SUM(medication_count)) / COUNT(*))) AS los_medication_corr
FROM ICUSTAYS i
JOIN (
  SELECT a.hadm_id, COUNT(pr.drug_name_generic) AS medication_count
  FROM ADMISSIONS a
  JOIN PRESCRIPTIONS pr ON a.hadm_id = pr.hadm_id
  WHERE a.diagnosis = 'SEPSIS'
  GROUP BY a.hadm_id
) AS medication_counts ON i.hadm_id = medication_counts.hadm_id;
"""

# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

icustays.to_sql('ICUSTAYS', conn, if_exists='replace', index=False)
prescriptions.to_sql('PRESCRIPTIONS', conn, if_exists='replace', index=False)
admissions.to_sql('ADMISSIONS', conn, if_exists='replace', index=False)

result_df = pd.read_sql_query(query, conn)
print(result_df)

conn.close()