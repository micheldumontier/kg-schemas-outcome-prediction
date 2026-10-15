import json, pytest
from aggregate import summarise, macros


def write(tmp, run, n, auc=0.9):
    d = tmp / run
    d.mkdir(parents=True)
    for k in range(n):
        (d / f'fold{k}.json').write_text(json.dumps(dict(run=run, fold=k, acc=0.8, auc_weighted=auc + k * 0.002,
            f1_backhome=0.8, f1_rehab=0.7, f1_death=0.6, f1_macro=0.7, f1_weighted=0.75, best_epoch=1,
            pred_class_counts=[1, 1, 1])))


def test_aggregate_rejects_incomplete_runs(tmp_path):
    write(tmp_path, 'A', 9)
    with pytest.raises(ValueError, match='A has 9 folds'):
        summarise(tmp_path, ['A'])


def test_macros_format_mean_and_sd(tmp_path):
    write(tmp_path, 'D2', 10)
    tex = macros(summarise(tmp_path, ['D2']))
    assert '\\newcommand{\\aucDtwo}{0.909\\,$\\pm$\\,0.006}' in tex   # mean of 0.900..0.918 step 0.002, SD ddof=1
    assert '\\newcommand{\\foneDtwoDE}{0.600\\,$\\pm$\\,0.000}' in tex
    assert '—' not in tex and '---' not in tex


def test_extra_runs_get_letter_only_macro_names(tmp_path):
    write(tmp_path, 'RF_SEPT2024', 10)
    write(tmp_path, 'EVENTS_RF', 10)
    tex = macros(summarise(tmp_path, ['RF_SEPT2024', 'EVENTS_RF']))
    assert '\\newcommand{\\aucRFsept}{' in tex and '\\newcommand{\\aucEventsRF}{' in tex


def test_partial_macros_fill_complete_runs_and_mark_the_rest(tmp_path):
    from aggregate import partial_macros
    write(tmp_path, 'LR', 10)
    write(tmp_path, 'A', 3)
    tex = partial_macros(tmp_path, ['LR', 'A'])
    assert '\\newcommand{\\aucLR}{0.909\\,$\\pm$\\,0.006}' in tex
    assert '\\newcommand{\\aucA}{\\textbf{??}}' in tex


def test_table_has_paper_column_and_groups(tmp_path):
    from aggregate import ROWS, table
    for _, run, _, _ in [r for r in ROWS if r[1]]:
        write(tmp_path, run, 10)
    tex = table(summarise(tmp_path, [r[1] for r in ROWS if r[1]]))
    assert 'Paper AUC' in tex
    assert '\\multicolumn{5}{l}{\\textit{As published}}' in tex
    sphn = next(line for line in tex.splitlines() if line.startswith('SPHN-ts'))
    assert sphn.rstrip(' \\').endswith('0.91')


def test_table_marks_runs_still_in_progress(tmp_path):
    from aggregate import table
    write(tmp_path, 'LR', 10)
    tex = table(summarise(tmp_path, ['LR']))
    a_row = next(line for line in tex.splitlines() if line.startswith('CARE-SM*-ts') and '& 3 &' in line)
    assert '\\textbf{??}' in a_row
