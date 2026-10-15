import gzip, json
import pyarrow.parquet as pq
from dump_to_parquet import convert, unescape

DUMP = """--
-- synthetic dump for tests (no real data)
--
CREATE TABLE s.people (
    id integer NOT NULL,
    born timestamp without time zone,
    code character(7),
    note text,
    score double precision
);

CREATE TABLE s.events (
    id integer,
    itemid integer,
    value character varying(20)
);

CREATE TABLE s.ignored (
    x integer
);

COPY s.people (id, born, code, note, score) FROM stdin;
1\t2180-05-06 22:23:00\t0010   \ta\\tb\t1.5
2\t\\N\tI60    \t\\N\t\\N
\\.

COPY s.events (id, itemid, value) FROM stdin;
1\t220739\t4
1\t999\tx
2\t223900\t5
\\.

COPY s.ignored (x) FROM stdin;
7
\\.
"""


def make(tmp_path):
    p = tmp_path / 'dump.sql.gz'
    with gzip.open(p, 'wt') as f:
        f.write(DUMP)
    return p


def test_unescape_handles_nulls_and_backslash_escapes():
    assert unescape('\\N') is None
    assert unescape('a\\tb') == 'a\tb'
    assert unescape('back\\\\slash') == 'back\\slash'
    assert unescape('line\\nbreak') == 'line\nbreak'


def test_convert_types_values_and_counts(tmp_path):
    report = convert(make(tmp_path), tmp_path / 'out', tables={'s.people': None})
    t = pq.read_table(tmp_path / 'out' / 's' / 'people.parquet')
    assert str(t.schema.field('id').type) == 'int64'
    assert str(t.schema.field('born').type) == 'timestamp[us]'
    assert str(t.schema.field('score').type) == 'double'
    rows = t.to_pylist()
    assert rows[0]['code'] == '0010' and rows[1]['code'] == 'I60'      # character(n) padding removed
    assert rows[0]['note'] == 'a\tb' and rows[1]['note'] is None
    assert rows[1]['born'] is None and rows[0]['score'] == 1.5
    assert report['s.people'] == {'rows_in_dump': 2, 'rows_written': 2}


def test_convert_filters_rows_and_skips_unrequested_tables(tmp_path):
    report = convert(make(tmp_path), tmp_path / 'out', tables={'s.events': ('itemid', {'220739', '223900'})})
    t = pq.read_table(tmp_path / 'out' / 's' / 'events.parquet')
    assert t.column('itemid').to_pylist() == [220739, 223900]
    assert report['s.events'] == {'rows_in_dump': 3, 'rows_written': 2}
    assert not (tmp_path / 'out' / 's' / 'ignored.parquet').exists() and 's.ignored' not in report


def test_convert_stops_after_last_requested_table(tmp_path):
    p = make(tmp_path)
    with open(p, 'ab') as f:          # corrupt bytes after the valid gzip stream; reading them would raise
        f.write(b'not gzip data' * 100)
    report = convert(p, tmp_path / 'out', tables={'s.people': None})
    assert report['s.people']['rows_written'] == 2


def test_timestamps_with_fractional_seconds(tmp_path):
    from dump_to_parquet import to_arrow
    arr = to_arrow(['2100-01-02 03:04:05', '2100-01-02 03:04:05.983', None], 'timestamp without time zone')
    assert str(arr.type) == 'timestamp[us]'
    assert arr.to_pylist()[1].microsecond == 983000 and arr.to_pylist()[2] is None


def test_conversion_errors_name_the_column_but_not_the_value(tmp_path):
    import pytest
    from dump_to_parquet import to_arrow
    with pytest.raises(ValueError) as e:
        to_arrow(['not-a-number-SECRET'], 'integer', column='mimiciv_x.t.col')
    assert 'mimiciv_x.t.col' in str(e.value) and 'SECRET' not in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__
