"""Aggregate-only cohort report: python cohort_report.py cohort.sql <parquet root>. Counts below 11 are suppressed (PhysioNet small-cell guidance)."""
import sys
import duckdb
P = sys.argv[2]
con = duckdb.connect()
sql = open(sys.argv[1]).read().replace('{P}', P)
for stmt in [s for s in sql.split(';') if s.strip() and not s.strip().startswith('--') or 'CREATE' in s]:
    if stmt.strip():
        con.execute(stmt)
sup = lambda n: n if n is None or n >= 11 else '<11'
def q(label, query):
    rows = con.execute(query).fetchall()
    print(f'== {label}')
    for r in rows:
        print('  ', tuple(r[:-1]) + (sup(r[-1]),))      # suppress only the count (last column)
q('flow: admissions with SAH code / with traumatic SAH / adult non-traumatic / first per patient',
  """select (select count(*) from sah_dx), (select count(*) from candidates where has_traumatic_sah),
            (select count(*) from candidates where age_at_admission>=18 and not has_traumatic_sah), (select count(*) from cohort)""")
q('SAH as principal diagnosis (seq_num = 1) in cohort', "select count(*) from cohort where best_seq = 1")
q('in-hospital death', "select hospital_expire_flag, count(*) from cohort group by 1 order by 1")
q('discharge location', "select coalesce(discharge_location,'(missing)'), count(*) from cohort group by 1 order by 2 desc")
q('admission type', "select admission_type, count(*) from cohort group by 1 order by 2 desc")
q('age at admission (mean, sd, median)', "select round(avg(age_at_admission),1), round(stddev(age_at_admission),1), median(age_at_admission) from cohort")
q('admissions with an ICU stay', f"select count(distinct c.hadm_id) from cohort c join '{P}/mimiciv_icu/icustays.parquet' i using (hadm_id)")
q('admissions with any GCS charted', f"select count(distinct c.hadm_id) from cohort c join '{P}/mimiciv_icu/chartevents.parquet' e using (hadm_id) where e.itemid in (220739,223900,223901)")
q('admissions with nimodipine (eMAR or prescriptions)', f"""select count(distinct hadm_id) from (
   select hadm_id from '{P}/mimiciv_hosp/emar.parquet' where medication ilike '%nimodipine%'
   union select hadm_id from '{P}/mimiciv_hosp/prescriptions.parquet' where drug ilike '%nimodipine%') x
   where hadm_id in (select hadm_id from cohort)""")
