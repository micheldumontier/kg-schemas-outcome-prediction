"""Build landmark feature tables for the SAH cohort on the A40. Prints aggregate counts only (cells < 11 suppressed).

Usage: python build_features.py cohort.sql <parquet root> <output dir>
Writes <output dir>/events.parquet and features_L{24,48,72}.parquet. Patient-level files stay on the A40.
"""
import sys
from pathlib import Path
import duckdb
import pandas as pd
from landmark import outcome_class, event_times, gcs_totals, features_at, name_measurements, MEASURE_ITEMS

LANDMARKS = (24, 48, 72)
EMAR_OK = "('Administered', 'Partial Administered', 'Started', 'Restarted', 'Applied', 'Confirmed')"
EVENT_SQL = {
    # eMAR administrations and prescriptions, timestamped
    'nimodipine': ["select hadm_id, charttime as time, false as date_only from '{P}/mimiciv_hosp/emar.parquet' where medication ilike '%nimodipine%' and event_txt in " + EMAR_OK,
                   "select hadm_id, starttime as time, false from '{P}/mimiciv_hosp/prescriptions.parquet' where drug ilike '%nimodipine%'"],
    'paracetamol': ["select hadm_id, charttime, false from '{P}/mimiciv_hosp/emar.parquet' where (medication ilike '%acetaminophen%' or medication ilike '%paracetamol%') and event_txt in " + EMAR_OK],
    'morphine': ["select hadm_id, charttime, false from '{P}/mimiciv_hosp/emar.parquet' where medication ilike '%morphine%' and event_txt in " + EMAR_OK],
    # ICU infusions
    'norepinephrine': ["select hadm_id, starttime, false from '{P}/mimiciv_icu/inputevents.parquet' where itemid = 221906"],
    'vasopressor': ["select hadm_id, starttime, false from '{P}/mimiciv_icu/inputevents.parquet' where itemid in (221906, 221749, 229630, 229632, 222315, 221289, 229617, 221662)"],
    'milrinone': ["select hadm_id, starttime, false from '{P}/mimiciv_icu/inputevents.parquet' where itemid = 221986"],
    # procedures, timestamped (ICU) or dated (ICD)
    'evd': ["select hadm_id, starttime, false from '{P}/mimiciv_icu/procedureevents.parquet' where itemid in (225456, 226475, 229519, 229520)",
            "select hadm_id, cast(chartdate as timestamp), true from '{P}/mimiciv_hosp/procedures_icd.parquet' where (icd_version = 10 and icd_code in ('009600Z','009630Z','009640Z')) or (icd_version = 9 and icd_code in ('0221','022'))"],
    'intubation': ["select hadm_id, starttime, false from '{P}/mimiciv_icu/procedureevents.parquet' where itemid in (224385, 225792)",
                   "select hadm_id, cast(chartdate as timestamp), true from '{P}/mimiciv_hosp/procedures_icd.parquet' where (icd_version = 10 and icd_code = '0BH17EZ') or (icd_version = 9 and icd_code = '9604')"],
    'coiling': ["select hadm_id, cast(chartdate as timestamp), true from '{P}/mimiciv_hosp/procedures_icd.parquet' where (icd_version = 10 and icd_code in ('03VG3DZ','03LG3DZ')) or (icd_version = 9 and icd_code in ('3972','3975','3976','3979'))"],
    'clipping': ["select hadm_id, cast(chartdate as timestamp), true from '{P}/mimiciv_hosp/procedures_icd.parquet' where (icd_version = 10 and icd_code in ('03VG0CZ','03VG4CZ','03LG0CZ')) or (icd_version = 9 and icd_code in ('3951','3952'))"],
    'angioplasty': ["select hadm_id, cast(chartdate as timestamp), true from '{P}/mimiciv_hosp/procedures_icd.parquet' where (icd_version = 10 and icd_code in ('037G3ZZ','037G3DZ')) or (icd_version = 9 and icd_code = '0062')"],
}


def main(cohort_sql, P, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for stmt in open(cohort_sql).read().replace('{P}', P).split(';'):
        if 'CREATE' in stmt:
            con.execute(stmt)
    adm = con.execute(f"""select c.hadm_id, c.admittime, c.dischtime, a.deathtime, c.discharge_location, c.hospital_expire_flag,
                                 c.age_at_admission as age, p.gender
                          from cohort c join '{P}/mimiciv_hosp/admissions.parquet' a using (hadm_id)
                          join '{P}/mimiciv_hosp/patients.parquet' p on p.subject_id = c.subject_id""").df()
    adm['outcome'] = outcome_class(adm)
    con.register('ids', adm[['hadm_id']])
    parts = []
    for event, queries in EVENT_SQL.items():
        for q in queries:
            df = con.execute(f"select * from ({q.replace('{P}', P)}) x where hadm_id in (select hadm_id from ids)").df()
            df.columns = ['hadm_id', 'time', 'date_only']
            parts.append(df.assign(event=event))
    raw = pd.concat(parts, ignore_index=True).dropna(subset=['hadm_id', 'time'])
    raw.to_parquet(out / 'events.parquet')
    ev = event_times(raw)
    ce = con.execute(f"""select hadm_id, itemid, charttime, valuenum from '{P}/mimiciv_icu/chartevents.parquet'
                         where hadm_id in (select hadm_id from ids) and valuenum is not null""").df()
    gcs = gcs_totals(ce)
    lab = con.execute(f"""select hadm_id, itemid, charttime as time, valuenum as value from '{P}/mimiciv_hosp/labevents.parquet'
                          where hadm_id in (select hadm_id from ids) and valuenum is not null""").df()
    meas = name_measurements(pd.concat([ce.rename(columns={'charttime': 'time', 'valuenum': 'value'}), lab], ignore_index=True))

    sup = lambda n: n if n >= 11 else '<11'
    print(f'cohort admissions {len(adm)}; outcome known {sup(int(adm.outcome.notna().sum()))}')
    print('outcome classes', {k: sup(int(v)) for k, v in adm.outcome.value_counts().items()})
    for L in LANDMARKS:
        f = features_at(adm, ev, gcs, meas, L, list(EVENT_SQL))
        f = f[f['outcome'].notna()]
        f.to_parquet(out / f'features_L{L}.parquet')
        print(f'== landmark {L} h: {len(f)} eligible with known outcome')
        print('   outcomes', {k: sup(int(v)) for k, v in f.outcome.value_counts().items()})
        print('   events before landmark', {e: sup(int(f[e].sum())) for e in EVENT_SQL})
        print('   with GCS', sup(int(f.gcs_first.notna().sum())), '| with sodium', sup(int(f.get('sodium_min', pd.Series(dtype=float)).notna().sum())),
              '| columns', f.shape[1])


if __name__ == '__main__':
    main(*sys.argv[1:4])
