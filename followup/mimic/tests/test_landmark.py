"""Tests for landmark feature construction, on synthetic patients only."""
import pandas as pd
import numpy as np
from landmark import outcome_class, event_times, eligible_at, gcs_totals, features_at

T0 = pd.Timestamp('2150-01-01 10:00')


def test_outcome_classes_follow_discharge_destination():
    adm = pd.DataFrame({'hadm_id': [1, 2, 3, 4, 5, 6],
                        'discharge_location': ['HOME', 'HOME HEALTH CARE', 'REHAB', 'HOSPICE', 'AGAINST ADVICE', None],
                        'hospital_expire_flag': [0, 0, 0, 0, 0, 1]})
    assert list(outcome_class(adm)) == ['home', 'home', 'rehab', 'death', None, 'death']


def test_icd_dated_events_are_known_at_end_of_day():
    raw = pd.DataFrame({'hadm_id': [1, 1], 'event': ['coiling', 'coiling'],
                        'time': [pd.Timestamp('2150-01-02'), pd.Timestamp('2150-01-03')], 'date_only': [True, True]})
    first = event_times(raw)
    assert first.loc[(1, 'coiling')] == pd.Timestamp('2150-01-02 23:59:59')


def test_eligibility_requires_alive_and_in_hospital_at_landmark():
    adm = pd.DataFrame({'hadm_id': [1, 2, 3], 'admittime': [T0] * 3,
                        'dischtime': [T0 + pd.Timedelta(hours=100), T0 + pd.Timedelta(hours=20), T0 + pd.Timedelta(hours=100)],
                        'deathtime': [pd.NaT, pd.NaT, T0 + pd.Timedelta(hours=30)]})
    assert list(eligible_at(adm, 24)) == [True, False, True]
    assert list(eligible_at(adm, 48)) == [True, False, False]


def test_gcs_total_needs_all_three_components_at_one_time():
    ce = pd.DataFrame({'hadm_id': [1, 1, 1, 1, 1], 'itemid': [220739, 223900, 223901, 220739, 223900],
                       'charttime': [T0] * 3 + [T0 + pd.Timedelta(hours=1)] * 2, 'valuenum': [3, 4, 5, 4, 5]})
    tot = gcs_totals(ce)
    assert len(tot) == 1 and tot.iloc[0]['gcs'] == 12


def test_features_exclude_everything_after_the_landmark():
    adm = pd.DataFrame({'hadm_id': [1], 'admittime': [T0], 'dischtime': [T0 + pd.Timedelta(days=10)], 'deathtime': [pd.NaT],
                        'age': [60], 'gender': ['F'], 'outcome': ['rehab']})
    ev = pd.Series({(1, 'nimodipine'): T0 + pd.Timedelta(hours=5), (1, 'evd'): T0 + pd.Timedelta(hours=30)})
    gcs = pd.DataFrame({'hadm_id': [1, 1], 'charttime': [T0 + pd.Timedelta(hours=2), T0 + pd.Timedelta(hours=40)], 'gcs': [9, 3]})
    meas = pd.DataFrame({'hadm_id': [1, 1], 'name': ['sodium', 'sodium'], 'time': [T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=26)],
                         'value': [140.0, 125.0]})
    f = features_at(adm, ev, gcs, meas, landmark_h=24, events=['nimodipine', 'evd'])
    row = f.iloc[0]
    assert row['nimodipine'] == 1 and row['nimodipine_h'] == 5.0
    assert row['evd'] == 0 and np.isnan(row['evd_h'])
    assert row['gcs_first'] == 9 and row['gcs_min'] == 9
    assert row['sodium_min'] == 140.0 and row['outcome'] == 'rehab'


def test_measurements_are_named_and_fahrenheit_converted():
    from landmark import name_measurements
    raw = pd.DataFrame({'hadm_id': [1, 1, 1], 'itemid': [223761, 223762, 50983], 'time': [T0] * 3, 'value': [98.6, 37.5, 140.0]})
    m = name_measurements(raw)
    assert list(m['name']) == ['temp', 'temp', 'sodium']
    assert round(m['value'].iloc[0], 1) == 37.0
