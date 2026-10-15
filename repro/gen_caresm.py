"""Verbatim port of notebooks/Data Generation/CARESM-graph_generation.ipynb (cells 1-9).

Only changes vs. the notebook (marked CHANGED):
  - input CSV / N / output paths are CLI arguments
  - 'output' column renamed to 'outcome' (data/syn_data.csv uses 'output'; the notebook expects 'outcome')
  - timing prints
"""
import sys
import time as _time
import pandas as pd
import numpy as np
import joblib
import uuid
from rdflib import ConjunctiveGraph
from string import Template
from scipy.stats import norm
from datetime import datetime, timedelta

csv_path, N, out_prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]  # CHANGED

df = pd.read_csv(csv_path, index_col=0)
df = df.rename(columns={'output': 'outcome'})  # CHANGED
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

numerical = ['hospital_stay_length', 'gcs', 'nb_acte', 'age']
categorical = ['gender', 'entry', 'entry_code', 'ica', 'ttt', 'ica_therapy', 'fever', 'o2_clinic', 'o2', 'hta', 'hct', 'tabagisme', 'etOH', 'diabete', 'headache', 'instable', 'vasospasme', 'ivh', 'outcome']
events = ['nimodipine',  'paracetamol', 'nad', 'corotrop', 'morphine', 'dve', 'atl', 'iot']

events_codes = {
    'nimodipine': 'C08CA06', # ACT
    'paracetamol': 'N02BE01', # ACT
    'nad': 'C01CA03', # ACT
    'corotrop': 'C01CE02', # ACT
    'morphine': 'N02AA01', # ACT
    'dve': '00P6X0Z', # Removal of Drainage Device from Cerebral Ventricle External Approach (ICD-10)
    'atl': 'Z98.6', # ICD-10
    'iot': '0BH17EZ' # ICD-10
}

prefix  = """
@prefix sio: <http://semanticscience.org/resource/> .
@prefix ncit: <http://purl.obolibrary.org/obo/NCIT_> .
@prefix nvasc: <http://nvasc.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

caresm_diagnosis_code_date_template = Template("""

        nvasc:synth_patient_$patient_id a sio:Person ;
            sio:has-role nvasc:role_$patient_id .

        nvasc:role_$patient_id a sio:Role, <http://purl.obolibrary.org/obo/OBI_0000093> ;
            sio:is-realized-in nvasc:diag_$diag_id .

        nvasc:diag_$diag_id a sio:Process, <http://purl.obolibrary.org/obo/NCIT_C18020> ;
            sio:has-output nvasc:diag_output_$diag_out_id ;
            nvasc:hasTimePoint nvasc:context_$context_id .

        nvasc:diag_output_$diag_out_id a sio:InformationContentEntity,
            ncit:$diag_code.

        nvasc:context_$context_id sio:SIO_000680 "$diag_start_date"^^xsd:dateTime ; # start date
                sio:SIO_000681 "$diag_end_date"^^xsd:dateTime ; # end date
                sio:SIO_000068 nvasc:timeline_$patient_id . # part-of

        nvasc:timeline_$patient_id a sio:SIO_000417, <http://purl.obolibrary.org/obo/NCIT_C54576> ;
                sio:SIO_000332 nvasc:synth_patient_$patient_id ; # is-about
                sio:SIO_000028 nvasc:context_$context_id . # this is a materialization of the has-part property
    """)

caresm_diagnosis_code_template = Template("""
    nvasc:synth_patient_$patient_id a sio:Person ;
        sio:has-role nvasc:role_$patient_id .
    nvasc:role_$patient_id a sio:Role, <http://purl.obolibrary.org/obo/OBI_0000093> ;
        sio:is-realized-in nvasc:diag_$diag_id .

    nvasc:diag_$diag_id a sio:Process, <http://purl.obolibrary.org/obo/NCIT_C18020> ;
        sio:has-output nvasc:diag_output_$diag_out_id .

    nvasc:diag_output_$diag_out_id a sio:InformationContentEntity,
        ncit:$diag_code.
    """)

caresm_diagnosis_quantity_template = Template("""
    nvasc:synth_patient_$patient_id a sio:Person ;
        sio:has-role nvasc:role_$patient_id .
    nvasc:role_$patient_id a sio:Role, <http://purl.obolibrary.org/obo/OBI_0000093> ;
        sio:is-realized-in nvasc:diag_$diag_id .

    nvasc:diag_$diag_id a sio:Process, <http://purl.obolibrary.org/obo/NCIT_C18020> ;
        sio:has-output nvasc:diag_output_$diag_out_id .

    nvasc:diag_output_$diag_out_id a sio:InformationContentEntity,
        ncit:$diag_code ; # TODO should be a proper dereferenceable URI
        sio:has-value "$diag_value"^^xsd:float ;
        sio:has-unit  "$diag_unit" . # TODO should be a URI, subClassOf sio:Unit
    """)

nvasc_outcome = Template(
    """
    nvasc:synth_patient_$patient_id nvasc:hasOutcome nvasc:outcome_$outcome .
    """
)

def gen_start_event(y_min=2020, y_max=2023):
    n_days = (y_max - y_min) * 365
    d0 = datetime.fromisoformat(f"{y_min}-01-01")
    day_rand = round(np.random.uniform(n_days))
    delta = timedelta(days=day_rand, hours=round(norm.rvs(12, 5)), minutes=round(np.random.uniform(60)))
    d_out = d0 + delta
    return(d_out)


def gen_patient_rdf(row, kg):
    _i = row.name
    d_start = gen_start_event()
    for f in row.index:
        if f in events:
            if row[f] != -1:
                h = row[f]
                if h == 0:
                    h = 1
                d_event = d_start + timedelta(hours=h)
                rdf = caresm_diagnosis_code_date_template.substitute(
                    diag_id=uuid.uuid4(),
                    diag_out_id=uuid.uuid4(),
                    context_id=uuid.uuid4(),
                    diag_label=f,
                    diag_code=events_codes[f],
                    diag_start_date=d_event.isoformat(),
                    diag_end_date=d_event.isoformat(),
                    patient_id=_i,
                )
                kg.parse(data=prefix+rdf, format="turtle")

        elif f in numerical:
            value = None
            unit = None
            if f == "hospital_stay_length":
                value = round(row[f])
                unit = "days"
            elif  f == "gcs":
                value = round(row[f],2)
                unit = "gcs"
            elif  f == "nb_acte":
                value = round(row[f])
                unit = "received medical treatments"
            elif  f == "age":
                value = round(row[f])
                unit = "age"

            rdf = caresm_diagnosis_quantity_template.substitute(
                diag_id=uuid.uuid4(),
                diag_out_id=uuid.uuid4(),
                diag_label=f,
                diag_code=str(f) + "_" + str(row[f]),
                diag_value=value,
                diag_unit=unit,
                patient_id=_i)
            kg.parse(data=prefix + rdf, format="turtle")

        elif f in categorical:
            if f in ["outcome"]:
                outcome_value = row[f]
                rdf = nvasc_outcome.substitute(outcome=outcome_value, patient_id=_i)
                kg.parse(data=prefix + rdf, format="turtle")
            else:
                rdf = caresm_diagnosis_code_template.substitute(
                    diag_id=uuid.uuid4(),
                    diag_out_id=uuid.uuid4(),
                    diag_label=f,
                    diag_code=str(f) + "_" + str(row[f]),
                    patient_id=_i)
                kg.parse(data=prefix + rdf, format="turtle")

t0 = _time.time()
no_outcome = df.iloc[:N].copy()
no_outcome = no_outcome.drop(columns=["outcome"])

# Serialize data
kg = ConjunctiveGraph()
no_outcome.apply(gen_patient_rdf, axis=1, kg=kg)
print(f"parsed {len(kg)} triples in {_time.time()-t0:.0f}s", flush=True)

# Add timestamps and time relations
before_query = """
CONSTRUCT {
    ?context1 time:before ?context2 .
} WHERE {
#SELECT * WHERE {
    ?timeline rdf:type <http://purl.obolibrary.org/obo/NCIT_C54576> .
    ?timeline sio:SIO_000028 ?context1, ?context2 .
    ?context1 sio:SIO_000680 ?start1 .
    ?context2 sio:SIO_000680 ?start2 .

    filter((?start1 < ?start2) && (?context1 != ?context2))
}
"""

res = kg.query(before_query)
for t in res:
    kg.add(t)
print(f"Generated {len(kg)} RDF triples in {_time.time()-t0:.0f}s", flush=True)

kg.serialize(f"{out_prefix}.nt", format="ntriples")
joblib.dump(df["outcome"][:N].astype(int).to_list(), f"{out_prefix}_outcomes.joblib")
print("done", flush=True)
