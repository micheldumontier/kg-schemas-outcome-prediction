-- Cohort for plan C: adults with a first admission for non-traumatic subarachnoid haemorrhage in MIMIC-IV v3.1.
-- Parameter {P} is the Parquet root (for example /home/coder/mimic-iv/parquet). Run with DuckDB.

CREATE OR REPLACE TEMP VIEW sah_dx AS
SELECT DISTINCT d.subject_id, d.hadm_id, min(d.seq_num) AS best_seq
FROM '{P}/mimiciv_hosp/diagnoses_icd.parquet' d
WHERE (d.icd_version = 10 AND d.icd_code LIKE 'I60%')
   OR (d.icd_version = 9 AND d.icd_code = '430')
GROUP BY d.subject_id, d.hadm_id;

CREATE OR REPLACE TEMP VIEW traumatic AS
SELECT DISTINCT hadm_id
FROM '{P}/mimiciv_hosp/diagnoses_icd.parquet'
WHERE (icd_version = 10 AND icd_code LIKE 'S066%')
   OR (icd_version = 9 AND (icd_code LIKE '8520%' OR icd_code LIKE '8521%'));

CREATE OR REPLACE TEMP VIEW candidates AS
SELECT s.subject_id, s.hadm_id, s.best_seq, a.admittime, a.dischtime, a.discharge_location,
       a.hospital_expire_flag, a.admission_type,
       p.anchor_age + (year(a.admittime) - p.anchor_year) AS age_at_admission,
       s.hadm_id IN (SELECT hadm_id FROM traumatic) AS has_traumatic_sah
FROM sah_dx s
JOIN '{P}/mimiciv_hosp/admissions.parquet' a USING (hadm_id)
JOIN '{P}/mimiciv_hosp/patients.parquet' p ON p.subject_id = s.subject_id;

CREATE OR REPLACE TEMP VIEW cohort AS
SELECT * EXCLUDE (rn) FROM (
  SELECT c.*, row_number() OVER (PARTITION BY subject_id ORDER BY admittime) AS rn
  FROM candidates c
  WHERE age_at_admission >= 18 AND NOT has_traumatic_sah
) WHERE rn = 1;
