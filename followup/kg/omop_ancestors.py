"""Ancestors of the graph's SNOMED CT, LOINC and ICD-10-PCS codes from an OMOP vocabulary export (Athena).

Usage: python omop_ancestors.py <vocabulary dir with CONCEPT.csv and CONCEPT_RELATIONSHIP.csv> <out.tsv>
Follows valid 'Is a' relationships transitively (OMOP links vocabularies, so a procedure code also reaches SNOMED CT
procedures). Vocabulary roots that every concept reaches (SNOMED CT Concept, LOINC {component}) are left out, as
SPHNConcept is. Writes code IRI, ancestor IRI, ancestor label and the shortest number of steps, one row per pair.
The output contains licensed SNOMED CT and LOINC content: keep it out of version control.
"""
import sys
from urllib.parse import quote
import duckdb
from terminology import PCS, LOINC, LOINC_LABS, PCS_AXES
from build_sphn_real import SNOMED, SNOMED_LABELS

PREFIX = {'SNOMED': SNOMED, 'LOINC': LOINC, 'ICD10PCS': PCS}
ROOTS = {('SNOMED', '138875005'), ('LOINC', 'LP432695-7')}
OURS = {'SNOMED': list(SNOMED_LABELS), 'LOINC': [c for c, _ in LOINC_LABS.values()], 'ICD10PCS': list(PCS_AXES)}


def iri(vocabulary, code):
    return f'<{PREFIX[vocabulary]}{quote(code, safe="-.")}>'


def closure(con):
    """Rows (vocabulary, code, ancestor vocabulary, ancestor code, ancestor name, steps) for our codes."""
    seeds = ' union all '.join(f"select '{v}' v, '{c}' code" for v, cs in OURS.items() for c in cs)
    return con.sql(f"""
        with recursive seed as ({seeds}),
        start as (select s.v, s.code, c.concept_id from seed s join concept c on c.vocabulary_id = s.v and c.concept_code = s.code),
        anc(v, code, id, d) as (select v, code, concept_id, 0 from start
                                union select anc.v, anc.code, isa.p, anc.d + 1 from anc join isa on isa.c = anc.id where anc.d < 20)
        select a.v, a.code, c.vocabulary_id, c.concept_code, c.concept_name, min(a.d) steps
        from anc a join concept c on c.concept_id = a.id
        where a.d > 0 and c.vocabulary_id in ('SNOMED', 'LOINC', 'ICD10PCS')
        group by 1, 2, 3, 4, 5 order by 1, 2, 6, 4""").fetchall()


def load(con, vocab_dir):
    read = lambda f: f"read_csv('{vocab_dir}/{f}', delim='\t', header=true, quote='', all_varchar=true)"
    con.sql(f"""create table concept as select concept_id::bigint concept_id, vocabulary_id, concept_code, concept_name
                from {read('CONCEPT.csv')} where vocabulary_id in ('SNOMED', 'LOINC', 'ICD10PCS')""")
    con.sql(f"""create table isa as select concept_id_1::bigint c, concept_id_2::bigint p from {read('CONCEPT_RELATIONSHIP.csv')}
                where relationship_id = 'Is a' and (invalid_reason is null or invalid_reason = '')""")


def main(vocab_dir, out):
    con = duckdb.connect()
    load(con, vocab_dir)
    rows = [r for r in closure(con) if (r[2], r[3]) not in ROOTS]
    with open(out, 'w') as f:
        for v, code, av, acode, name, steps in rows:
            f.write(f'{iri(v, code)}\t{iri(av, acode)}\t{name}\t{steps}\n')
    found = {(v, c) for v, c, *_ in rows}
    print(f'{len(rows)} ancestor pairs for {len(found)} of {sum(map(len, OURS.values()))} codes; wrote {out}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
