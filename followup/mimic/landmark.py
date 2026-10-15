"""Landmark features for the MIMIC-IV subarachnoid haemorrhage cohort (plan C, tasks C3 and C4).

Pure functions on DataFrames, so they can be tested on synthetic patients. Index time is hospital admission.
A patient enters landmark L (hours) only if alive and still in hospital at admission + L, and every feature
uses information timestamped strictly before admission + L. Events known only by date (ICD procedures) are
placed at the end of that day, so a same-day procedure cannot leak into an earlier landmark.
"""
import numpy as np
import pandas as pd

HOME = {'HOME', 'HOME HEALTH CARE', 'ASSISTED LIVING'}
REHAB = {'REHAB', 'SKILLED NURSING FACILITY', 'CHRONIC/LONG TERM ACUTE CARE', 'ACUTE HOSPITAL',
         'OTHER FACILITY', 'PSYCH FACILITY', 'HEALTHCARE FACILITY'}
DEATH = {'DIED', 'HOSPICE'}
GCS_ITEMS = {220739: 'eye', 223900: 'verbal', 223901: 'motor'}


def outcome_class(adm):
    """home / rehab / death from discharge destination; None when unknown (missing, against advice)."""
    def one(loc, died):
        if died == 1 or loc in DEATH:
            return 'death'
        if loc in REHAB:
            return 'rehab'
        if loc in HOME:
            return 'home'
        return None
    return pd.Series([one(l, d) for l, d in zip(adm['discharge_location'], adm['hospital_expire_flag'])], index=adm.index, dtype=object)


def event_times(raw):
    """First occurrence per (hadm_id, event). raw has hadm_id, event, time, date_only."""
    t = raw['time'].where(~raw['date_only'], raw['time'].dt.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    return t.groupby([raw['hadm_id'], raw['event']]).min()


def eligible_at(adm, landmark_h):
    lm = adm['admittime'] + pd.Timedelta(hours=landmark_h)
    alive = adm['deathtime'].isna() | (adm['deathtime'] > lm)
    return (adm['dischtime'] > lm) & alive


def gcs_totals(ce):
    """Total GCS where eye, verbal and motor are charted at the same time."""
    g = ce[ce['itemid'].isin(GCS_ITEMS)].assign(part=lambda d: d['itemid'].map(GCS_ITEMS))
    wide = g.pivot_table(index=['hadm_id', 'charttime'], columns='part', values='valuenum', aggfunc='min').dropna()
    wide = wide.reset_index()
    wide['gcs'] = wide[['eye', 'verbal', 'motor']].sum(axis=1)
    return wide[['hadm_id', 'charttime', 'gcs']]


def features_at(adm, ev, gcs, meas, landmark_h, events):
    """One row per eligible admission. ev is the Series from event_times; meas has hadm_id, name, time, value."""
    a = adm[eligible_at(adm, landmark_h)].copy()
    lm = a.set_index('hadm_id')['admittime'] + pd.Timedelta(hours=landmark_h)
    t0 = a.set_index('hadm_id')['admittime']
    out = a[['hadm_id', 'age', 'gender', 'outcome']].set_index('hadm_id')
    for e in events:
        first = pd.Series({h: ev.get((h, e), pd.NaT) for h in out.index}, dtype='datetime64[ns]')
        before = first < lm.reindex(first.index)
        out[e] = before.astype(int)
        out[e + '_h'] = ((first - t0.reindex(first.index)).dt.total_seconds() / 3600).where(before)
    g = gcs[gcs['hadm_id'].isin(out.index)]
    g = g[g['charttime'] < g['hadm_id'].map(lm)].sort_values('charttime')
    out['gcs_first'] = g.groupby('hadm_id')['gcs'].first()
    out['gcs_min'] = g.groupby('hadm_id')['gcs'].min()
    m = meas[meas['hadm_id'].isin(out.index)]
    m = m[m['time'] < m['hadm_id'].map(lm)]
    if len(m):
        agg = m.groupby(['hadm_id', 'name'])['value'].agg(['min', 'max']).unstack('name')
        agg.columns = [f'{name}_{stat}' for stat, name in agg.columns]
        out = out.join(agg)
    return out.reset_index()


MEASURE_ITEMS = {
    220045: 'hr', 220050: 'sbp', 220179: 'sbp', 220052: 'map', 220181: 'map', 220210: 'rr', 220277: 'spo2',
    223762: 'temp', 223761: 'temp',
    50983: 'sodium', 52623: 'sodium', 50971: 'potassium', 52610: 'potassium',
    50931: 'glucose', 52569: 'glucose', 50809: 'glucose', 50912: 'creatinine', 52546: 'creatinine',
    51222: 'haemoglobin', 50811: 'haemoglobin', 51640: 'haemoglobin', 51265: 'platelets', 53189: 'platelets',
    51301: 'wbc', 51755: 'wbc', 51756: 'wbc', 50813: 'lactate', 52442: 'lactate', 53154: 'lactate',
}


def name_measurements(raw):
    """Map chart and lab item IDs to measurement names; convert Fahrenheit temperatures to Celsius."""
    m = raw[raw['itemid'].isin(MEASURE_ITEMS)].copy()
    f = m['itemid'] == 223761
    m.loc[f, 'value'] = (m.loc[f, 'value'] - 32) * 5 / 9
    m['name'] = m['itemid'].map(MEASURE_ITEMS)
    return m[['hadm_id', 'name', 'time', 'value']]
