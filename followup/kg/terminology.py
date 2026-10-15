"""Terminology labels and hierarchies for the codes used in the real-SPHN graphs (plan A, semantic enrichment).

Sources, checked 14 October 2026:
- ATC: WHO Collaborating Centre ATC/DDD index (https://atcddd.fhi.no/atc_ddd_index/); hierarchy follows the code itself
  (anatomical main group, therapeutic subgroup, pharmacological subgroup, chemical subgroup, substance).
- ICD-10-PCS: CMS 2026 code tables (icd10pcs_tables_2026.xml); every code below was found in the tables and the axis
  labels are copied from them. Ancestors are the prefixes for section, body system, root operation and body part.
- LOINC: our mapping of the MIMIC-IV laboratory items (chemistry items to serum or plasma codes, haematology items
  to blood codes, the blood-gas lactate to the whole-blood code). MIMIC-IV v3.1 carries no LOINC codes.
- SPHN: superclasses read from the SPHN RDF Schema 2026.1 (tests check them against the local copy when present).
SNOMED CT ancestors need the licensed release and are not included.
"""
SPHN = 'https://biomedit.ch/rdf/sphn-schema/sphn#'
ATC = 'https://www.whocc.no/atc_ddd_index/?code='
PCS = 'http://purl.bioontology.org/ontology/ICD10PCS/'
LOINC = 'https://loinc.org/rdf/'

ATC_LABELS = {
    'C': 'cardiovascular system', 'C01': 'cardiac therapy', 'C01C': 'cardiac stimulants excl. cardiac glycosides',
    'C01CA': 'adrenergic and dopaminergic agents', 'C01CA03': 'norepinephrine',
    'C01CE': 'phosphodiesterase inhibitors', 'C01CE02': 'milrinone',
    'C08': 'calcium channel blockers', 'C08C': 'selective calcium channel blockers with mainly vascular effects',
    'C08CA': 'dihydropyridine derivatives', 'C08CA06': 'nimodipine',
    'N': 'nervous system', 'N02': 'analgesics', 'N02A': 'opioids', 'N02AA': 'natural opium alkaloids',
    'N02AA01': 'morphine', 'N02B': 'other analgesics and antipyretics', 'N02BE': 'anilides', 'N02BE01': 'paracetamol',
}

# CMS axis labels by character position: section, body system, root operation, body part, approach, device, qualifier
PCS_AXES = {
    '009630Z': ('Medical and Surgical', 'Central Nervous System and Cranial Nerves', 'Drainage', 'Cerebral Ventricle',
                'Percutaneous', 'Drainage Device', 'No Qualifier'),
    '0BH17EZ': ('Medical and Surgical', 'Respiratory System', 'Insertion', 'Trachea', 'Via Natural or Artificial Opening',
                'Intraluminal Device, Endotracheal Airway', 'No Qualifier'),
    '03VG3DZ': ('Medical and Surgical', 'Upper Arteries', 'Restriction', 'Intracranial Artery', 'Percutaneous',
                'Intraluminal Device', 'No Qualifier'),
    '03VG0CZ': ('Medical and Surgical', 'Upper Arteries', 'Restriction', 'Intracranial Artery', 'Open',
                'Extraluminal Device', 'No Qualifier'),
    '037G3ZZ': ('Medical and Surgical', 'Upper Arteries', 'Dilation', 'Intracranial Artery', 'Percutaneous',
                'No Device', 'No Qualifier'),
}

LOINC_LABS = {   # lab name in the feature table: (LOINC code, long common name)
    'sodium': ('2951-2', 'Sodium [Moles/volume] in Serum or Plasma'),
    'potassium': ('2823-3', 'Potassium [Moles/volume] in Serum or Plasma'),
    'glucose': ('2345-7', 'Glucose [Mass/volume] in Serum or Plasma'),
    'creatinine': ('2160-0', 'Creatinine [Mass/volume] in Serum or Plasma'),
    'haemoglobin': ('718-7', 'Hemoglobin [Mass/volume] in Blood'),
    'platelets': ('777-3', 'Platelets [#/volume] in Blood by Automated count'),
    'wbc': ('6690-2', 'Leukocytes [#/volume] in Blood by Automated count'),
    'lactate': ('32693-4', 'Lactate [Moles/volume] in Blood'),
}

SPHN_PARENTS = {   # direct superclass other than SPHNConcept, from the 2026.1 schema
    **{c: 'Measurement' for c in ('HeartRateMeasurement', 'BloodPressureMeasurement', 'RespiratoryRateMeasurement',
                                  'OxygenSaturationMeasurement', 'BodyTemperatureMeasurement')},
    **{c: 'Result' for c in ('HeartRate', 'BloodPressure', 'RespiratoryRate', 'OxygenSaturation', 'BodyTemperature',
                             'AssessmentResult', 'LabResult')},
}
SPHN_LABELS = {'Measurement': 'Measurement', 'Result': 'Result', 'SPHNConcept': 'SPHN Concept'}


def _pcs_prefix_label(code, n):
    a = PCS_AXES[code]
    return {1: a[0], 2: a[1], 3: f'{a[2]} of {a[1]}', 4: f'{a[2]} of {a[3]}'}[n]


def load_extra(path):
    """Code IRI -> [(ancestor IRI, label)] from omop_ancestors.py output, nearest first."""
    extra = {}
    for line in open(path):
        code, anc, label, steps = line.rstrip('\n').split('\t')
        extra.setdefault(code, []).append((int(steps), anc, label))
    return {c: [(a, l) for _, a, l in sorted(v)] for c, v in extra.items()}


def ancestors(iri, extra=None):
    """Ancestor IRIs (nearest first, brackets included) and their labels for a code or SPHN class IRI.

    `extra` (from load_extra) replaces the built-in hierarchy for the codes it covers, and gives ancestors of its own
    ancestors no further hierarchy (they are already in each code's closure).
    """
    if extra is not None:
        if iri in extra:
            return extra[iri]
        if any(iri == a for v in extra.values() for a, _ in v):
            return []
    x = iri[1:-1]
    if x.startswith(ATC):
        code = x[len(ATC):]
        return [(f'<{ATC}{code[:n]}>', ATC_LABELS[code[:n]]) for n in (5, 4, 3, 1) if n < len(code)]
    if x.startswith(PCS):
        prefix = x[len(PCS):]
        code = next(c for c in PCS_AXES if c.startswith(prefix))       # a full code that carries the axis labels
        return [(f'<{PCS}{code[:n]}>', _pcs_prefix_label(code, n)) for n in (4, 3, 2, 1) if n < len(prefix)]
    if x.startswith(SPHN):
        cls = x[len(SPHN):]
        # SPHNConcept, the universal root, is left out: like owl:Thing it distinguishes nothing, and as a type of every
        # record it would become a hub joining all records of all patients
        return [(f'<{SPHN}{SPHN_PARENTS[cls]}>', SPHN_LABELS[SPHN_PARENTS[cls]])] if cls in SPHN_PARENTS else []
    return []                                  # LOINC, SNOMED CT: no hierarchy included
