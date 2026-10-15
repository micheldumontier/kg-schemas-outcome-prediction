"""Stream a plain-text pg_dump (gzip) and write selected tables to typed Parquet, without a database server.

Reads CREATE TABLE statements for column types, then each requested COPY block, decodes the COPY text format
(tab separated, \\N for NULL, backslash escapes), optionally keeps only rows whose column is in a value set,
and writes one Parquet file per table under <out>/<schema>/<table>.parquet. Prints only counts.

Usage:
    python dump_to_parquet.py mimic.sql.gz out_dir tables.json
where tables.json maps "schema.table" to null (all rows) or ["column", ["value", ...]] (filtered rows).
"""
import gzip, json, re, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BATCH = 500_000
ESCAPES = {'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t', 'v': '\v', '\\': '\\'}
_ESC = re.compile(r'\\(?:([0-7]{1,3})|x([0-9A-Fa-f]{1,2})|(.))')


def unescape(field):
    if field == '\\N':
        return None
    if '\\' not in field:
        return field

    def repl(m):
        if m.group(1):
            return chr(int(m.group(1), 8))
        if m.group(2):
            return chr(int(m.group(2), 16))
        return ESCAPES.get(m.group(3), m.group(3))
    return _ESC.sub(repl, field)


def arrow_type(pg_type):
    t = pg_type.lower()
    if t.startswith(('integer', 'smallint', 'bigint', 'int')):
        return pa.int64()
    if t.startswith(('numeric', 'double precision', 'real', 'decimal')):
        return pa.float64()
    if t.startswith('timestamp'):
        return pa.timestamp('us')
    if t.startswith('date'):
        return pa.date32()
    if t.startswith('boolean'):
        return pa.bool_()
    return pa.string()


def parse_create(lines):
    """Return {schema.table: [(column, pg_type), ...]} from CREATE TABLE statements."""
    tables, current = {}, None
    for line in lines:
        m = re.match(r'CREATE TABLE ([\w.]+) \($', line)
        if m:
            current = m.group(1)
            tables[current] = []
            continue
        if current:
            if line.startswith(')'):
                current = None
                continue
            col = re.match(r'\s+"?(\w+)"?\s+(.+?)(?: NOT NULL)?(?: DEFAULT .*)?,?$', line)
            if col and not col.group(1).upper().startswith('CONSTRAINT'):
                tables[current].append((col.group(1), col.group(2).rstrip(',')))
    return tables


def to_arrow(values, pg_type, column=None):
    """Convert decoded strings to the column's Arrow type. Errors name the column, never a value."""
    try:
        arr = pa.array(values, pa.string())
        if pg_type.lower().startswith('character(') or pg_type.lower() == 'character':
            arr = pc.utf8_rtrim(arr, characters=' ')
        target = arrow_type(pg_type)
        if target == pa.string():
            return arr
        return pc.cast(arr, target)           # timestamps: ISO 8601, with or without fractional seconds
    except Exception:
        raise ValueError(f'cannot convert column {column} to {pg_type}') from None


def convert(dump_path, out_dir, tables):
    """tables: {schema.table: None | (column, set_of_string_values)}. Returns {table: {rows_in_dump, rows_written}}."""
    out_dir = Path(out_dir)
    report, ddl, schema_lines, in_ddl = {}, None, [], True
    with gzip.open(dump_path, 'rt', encoding='utf-8', newline='\n') as f:
        for line in f:
            line = line.rstrip('\n')
            if in_ddl and not line.startswith('COPY '):
                schema_lines.append(line)
                continue
            m = re.match(r'COPY ([\w.]+) \((.*)\) FROM stdin;$', line)
            if not m:
                continue
            if ddl is None:
                ddl, in_ddl = parse_create(schema_lines), False
            name, cols = m.group(1), [c.strip().strip('"') for c in m.group(2).split(',')]
            if name not in tables:
                for row in f:
                    if row == '\\.\n':
                        break
                continue
            types = dict(ddl[name])
            flt = tables[name]
            fcol = cols.index(flt[0]) if flt else None
            fvals = set(flt[1]) if flt else None
            schema = pa.schema([(c, arrow_type(types[c])) for c in cols])
            path = out_dir / name.split('.')[0] / f"{name.split('.')[1]}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(path, schema, compression='zstd')
            buf, n_in, n_out = [[] for _ in cols], 0, 0
            for row in f:
                if row == '\\.\n':
                    break
                n_in += 1
                fields = row.rstrip('\n').split('\t')
                if fcol is not None and fields[fcol] not in fvals:
                    continue
                if '\\' in row:
                    fields = [unescape(v) for v in fields]
                for b, v in zip(buf, fields):
                    b.append(v)
                n_out += 1
                if len(buf[0]) >= BATCH:
                    writer.write_table(pa.table([to_arrow(b, types[c], f'{name}.{c}') for b, c in zip(buf, cols)], schema=schema))
                    buf = [[] for _ in cols]
            if buf[0]:
                writer.write_table(pa.table([to_arrow(b, types[c], f'{name}.{c}') for b, c in zip(buf, cols)], schema=schema))
            writer.close()
            report[name] = {'rows_in_dump': n_in, 'rows_written': n_out}
            print(json.dumps({name: report[name]}), flush=True)
            if len(report) == len(tables):
                break                     # every requested table written; skip the rest of the dump
    return report


if __name__ == '__main__':
    spec = json.load(open(sys.argv[3]))
    tables = {k: (v[0], set(map(str, v[1]))) if v else None for k, v in spec.items()}
    convert(sys.argv[1], sys.argv[2], tables)
