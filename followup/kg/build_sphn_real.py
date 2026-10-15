"""Patient graphs that use the released SPHN RDF Schema (2026.1) and real terminology codes (pilot for plan A).

Differences from the template-style SPHN graphs in build_graphs.py:
- classes and properties are real SPHN 2026.1 terms (https://biomedit.ch/rdf/sphn-schema/sphn#);
- each record points to the patient (sphn:SubjectPseudoIdentifier) through sphn:hasSubjectPseudoIdentifier,
  as in SPHN, so the model must pass messages in both directions for the patient to receive information;
- drugs carry ATC codes, procedures ICD-10-PCS codes, laboratory tests LOINC codes, sex and the Glasgow coma scale
  SNOMED CT codes;
- every shared node has a human-readable label (shared_labels) for text-derived node embeddings;
- with_hierarchy adds the terminology and SPHN class hierarchies as a materialised closure (plan A, hierarchy arm).
Literal datatypes, patient IRIs and the runner file format are those of build_graphs.py.
"""
from build_graphs import patient_iri, lit, _numeric_fields, _events
from terminology import LOINC, LOINC_LABS, ancestors

SPHN = 'https://biomedit.ch/rdf/sphn-schema/sphn#'
ATC = 'https://www.whocc.no/atc_ddd_index/?code='
PCS = 'http://purl.bioontology.org/ontology/ICD10PCS/'
SNOMED = 'http://snomed.info/id/'
NV = 'http://nvasc.org/'
TYPE = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
LABEL = '<http://www.w3.org/2000/01/rdf-schema#label>'

CLASS_LABELS = {
    'SubjectPseudoIdentifier': 'Subject Pseudo Identifier', 'Age': 'Age', 'AdministrativeSex': 'Administrative Sex',
    'Quantity': 'Quantity', 'HeartRateMeasurement': 'Heart Rate Measurement', 'HeartRate': 'Heart Rate',
    'BloodPressureMeasurement': 'Blood Pressure Measurement', 'BloodPressure': 'Blood Pressure',
    'RespiratoryRateMeasurement': 'Respiratory Rate Measurement', 'RespiratoryRate': 'Respiratory Rate',
    'OxygenSaturationMeasurement': 'Oxygen Saturation Measurement', 'OxygenSaturation': 'Oxygen Saturation',
    'BodyTemperatureMeasurement': 'Body Temperature Measurement', 'BodyTemperature': 'Body Temperature',
    'AssessmentEvent': 'Assessment Event', 'Assessment': 'Assessment', 'AssessmentResult': 'Assessment Result',
    'LabTestEvent': 'Lab Test Event', 'LabTest': 'Lab Test', 'LabResult': 'Lab Result',
    'DrugAdministrationEvent': 'Drug Administration Event', 'Drug': 'Drug', 'Substance': 'Substance',
    'MedicalProcedure': 'Medical Procedure',
}
VITALS = {'hr': ('HeartRateMeasurement', 'HeartRate', 'hasQuantity', 'heart rate'),
          'sbp': ('BloodPressureMeasurement', 'BloodPressure', 'hasSystolicPressure', 'systolic blood pressure'),
          'map': ('BloodPressureMeasurement', 'BloodPressure', 'hasMeanPressure', 'mean arterial pressure'),
          'rr': ('RespiratoryRateMeasurement', 'RespiratoryRate', 'hasQuantity', 'respiratory rate'),
          'spo2': ('OxygenSaturationMeasurement', 'OxygenSaturation', 'hasQuantity', 'oxygen saturation'),
          'temp': ('BodyTemperatureMeasurement', 'BodyTemperature', 'hasQuantity', 'body temperature')}
LABS = {'sodium': 'Sodium', 'potassium': 'Potassium', 'glucose': 'Glucose', 'creatinine': 'Creatinine',
        'haemoglobin': 'Haemoglobin', 'platelets': 'Platelet count', 'wbc': 'White blood cell count', 'lactate': 'Lactate'}
DRUG_CODES = {'nimodipine': ('C08CA06', 'nimodipine'), 'paracetamol': ('N02BE01', 'paracetamol'),
              'morphine': ('N02AA01', 'morphine'), 'norepinephrine': ('C01CA03', 'norepinephrine'),
              'vasopressor': ('C01CA', 'adrenergic and dopaminergic agents'), 'milrinone': ('C01CE02', 'milrinone')}
PROC_CODES = {'evd': ('009630Z', 'Drainage of Cerebral Ventricle with Drainage Device, Percutaneous Approach'),
              'intubation': ('0BH17EZ', 'Insertion of Endotracheal Airway into Trachea, Via Natural or Artificial Opening'),
              'coiling': ('03VG3DZ', 'Restriction of Intracranial Artery with Intraluminal Device, Percutaneous Approach'),
              'clipping': ('03VG0CZ', 'Restriction of Intracranial Artery with Extraluminal Device, Open Approach'),
              'angioplasty': ('037G3ZZ', 'Dilation of Intracranial Artery, Percutaneous Approach')}
SNOMED_LABELS = {'248152002': 'Female', '248153007': 'Male', '386554004': 'Glasgow coma scale'}
STAT = {'min': 'minimum', 'max': 'maximum', 'first': 'first'}


def s(term):
    return f'<{SPHN}{term}>'


def sphn_real_triples(rec, i):
    p, n = patient_iri(i), lambda x: f'<{NV}p{i}/{x}>'
    t = [(p, TYPE, s('SubjectPseudoIdentifier'))]

    def record(node, cls, label=None):
        out = [(node, TYPE, s(cls)), (node, s('hasSubjectPseudoIdentifier'), p)]
        return out + ([(node, LABEL, f'"{label}"')] if label else [])

    sex = {'F': '248152002', 'M': '248153007'}.get(rec.get('gender'))
    if sex:
        t += record(n('sex'), 'AdministrativeSex') + [(n('sex'), s('hasCode'), f'<{SNOMED}{sex}>')]
    for f, v in _numeric_fields(rec):
        base, _, stat = f.rpartition('_')
        if f == 'age':
            t += record(n('age'), 'Age') + [(n('age'), s('hasQuantity'), n('age_q')), (n('age_q'), TYPE, s('Quantity')),
                                            (n('age_q'), s('hasValue'), lit(v, 'age'))]
        elif base == 'gcs':
            ev, res, q = n(f + '_event'), n(f + '_result'), n(f + '_q')
            t += record(ev, 'AssessmentEvent', f'Glasgow coma scale {STAT.get(stat, stat)}')
            t += [(ev, s('hasAssessment'), n('gcs_assessment')), (n('gcs_assessment'), TYPE, s('Assessment')),
                  (n('gcs_assessment'), s('hasCode'), f'<{SNOMED}386554004>'),
                  (ev, s('hasResult'), res), (res, TYPE, s('AssessmentResult')), (res, s('hasQuantity'), q),
                  (q, TYPE, s('Quantity')), (q, s('hasValue'), lit(v, f))]
        elif base in VITALS:
            mcls, rcls, prop, name = VITALS[base]
            m, r, q = n(f + '_measurement'), n(f + '_result'), n(f + '_q')
            t += record(m, mcls, f'{name} {STAT.get(stat, stat)}')
            t += [(m, s('hasResult'), r), (r, TYPE, s(rcls)), (r, s(prop), q), (q, TYPE, s('Quantity')),
                  (q, s('hasValue'), lit(v, f))]
        elif base in LABS:
            ev, test, res, q = n(f + '_event'), n(f + '_test'), n(f + '_result'), n(f + '_q')
            t += record(ev, 'LabTestEvent', f'{LABS[base].lower()} {STAT.get(stat, stat)}')
            t += [(ev, s('hasLabTest'), test), (test, TYPE, s('LabTest')), (test, s('hasCode'), f'<{LOINC}{LOINC_LABS[base][0]}>'),
                  (ev, s('hasResult'), res), (res, TYPE, s('LabResult')), (res, s('hasQuantity'), q),
                  (q, TYPE, s('Quantity')), (q, s('hasValue'), lit(v, f))]
    for e, h in _events(rec):
        node = n(e)
        if e in DRUG_CODES:
            code, _ = DRUG_CODES[e]
            t += record(node, 'DrugAdministrationEvent') + [
                (node, s('hasDrug'), n(e + '_drug')), (n(e + '_drug'), TYPE, s('Drug')),
                (n(e + '_drug'), s('hasActiveIngredient'), n(e + '_substance')), (n(e + '_substance'), TYPE, s('Substance')),
                (n(e + '_substance'), s('hasCode'), f'<{ATC}{code}>')]
        else:
            code, _ = PROC_CODES[e]
            t += record(node, 'MedicalProcedure') + [(node, s('hasCode'), f'<{PCS}{code}>')]
        t.append((node, s('hasStartDateTime'), lit(h, 'hours')))
    return t


def shared_labels(triples, extra=None):
    """Human-readable label for every node shared across patients (classes, codes, ancestors, label strings)."""
    nodes = {x for (h, _, o) in triples for x in (h, o)}
    labels = {a: lab for (_, r, o) in triples if r in (TYPE, s('hasCode')) for a, lab in ancestors(o, extra) if a in nodes}
    for x in nodes:
        if x.startswith(f'<{SPHN}'):
            labels.setdefault(x, CLASS_LABELS.get(x[len(SPHN) + 1:-1], x[len(SPHN) + 1:-1]))
        elif x.startswith(f'<{ATC}'):
            code = x[len(ATC) + 1:-1]
            labels.setdefault(x, next((name for c, name in DRUG_CODES.values() if c == code), None))
        elif x.startswith(f'<{PCS}'):
            code = x[len(PCS) + 1:-1]
            labels.setdefault(x, next((name for c, name in PROC_CODES.values() if c == code), None))
        elif x.startswith(f'<{SNOMED}'):
            labels.setdefault(x, SNOMED_LABELS.get(x[len(SNOMED) + 1:-1]))
        elif x.startswith(f'<{LOINC}'):
            labels.setdefault(x, next((name for c, name in LOINC_LABS.values() if c == x[len(LOINC) + 1:-1]), None))
        elif x.startswith('"') and '^^' not in x:
            labels[x] = x.strip('"')
    missing = [x for x, lab in labels.items() if lab is None]
    assert not missing, f'no label for {missing[:3]}'
    return labels


SUBCLASS = '<http://www.w3.org/2000/01/rdf-schema#subClassOf>'


def with_hierarchy(per_patient, extra=None):
    """Materialised closure of the code and class hierarchies.

    For every record typed with an SPHN class or coded with an ATC or ICD-10-PCS code, adds the same edge to each
    ancestor (record rdf:type superclass, record hasCode ancestor code), so ancestors sit at the depth of the code and
    the model depth does not change. `extra` (terminology.load_extra) supplies OMOP ancestors. Returns the extended per-patient triples and the shared ontology triples
    (class or code rdfs:subClassOf each ancestor), which are added once.
    """
    out, onto = [], set()
    for tr in per_patient:
        added = []
        for (h, r, o) in tr:
            if r in (TYPE, s('hasCode')):
                anc = [a for a, _ in ancestors(o, extra)]
                added += [(h, r, a) for a in anc]
                onto.update((o, SUBCLASS, a) for a in anc)
        out.append(tr + added)
    return out, sorted(onto)
