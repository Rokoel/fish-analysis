"""Compare roundness gates on the frozen consensus test set, one scan at a time.

Regenerate both cell detectors once per scan. Signal inference is independent
per cell (including suppression), so higher gates reuse surviving cell results.
The first nonempty 85% case can be rerun to verify that equivalence explicitly.
Temporary arrays and worker logs live outside the project and are discarded
after each scan; compact metrics and model/input identities support resuming.
"""

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtCore

from fish_viewer.evaluation import evaluate_run, frozen_models, paired_stats, save, sha
from fish_viewer.evaluation import channel_for, summarize_complete as summarize
from fish_viewer.analysis.controller import AnalysisController, default_models, default_proposal_model, default_refinement_model, sensitive_stardist_model
from fish_viewer.neural.audit import save_references, unique_references
from fish_viewer.pipeline.evaluation import rasterize_reference
from fish_viewer.pipeline.segmentation import Prepared
from fish_viewer.scenes import read_scene
from fish_viewer.workflow.volumes import surface_volume

THRESHOLDS = (.50, .75, .80, .82, .85)


def rebuild_references(case, collection, folder):
    """Recreate the archived reference grid/order without changing its split."""
    audit = case['audit']
    sources = {source['image']: source for source in collection['sources']}
    scenes = sorted({path for alias in audit['aliases'] for path in sources[alias]['scenes']})
    surfaces = [surface for path in scenes for surface in read_scene(Path(path)).surfaces
                if surface.time == 0 and surface_volume(surface) is not None]
    prepared = Prepared(np.empty(tuple(audit['shape_zyx']), np.uint8),
                        tuple(audit['spacing_zyx']), tuple(audit['origin_xyz']),
                        audit['channel'], [], audit['fingerprint'], {}, {})
    refs, names, _ = rasterize_reference(surfaces, prepared)
    refs, names = unique_references(refs, names)
    if len(refs) != audit['reference_count']:
        raise ValueError(f"Reference reconstruction changed: {case['image']}")
    folder.mkdir()
    save_references(folder / 'references.npz', refs, names)
    return dict(case, directory=str(folder))


def gated_result(report, metadata, labels, final_labels, minimum, folder):
    """Apply the stricter cell-only gate to unchanged per-cell measurements."""
    folder.mkdir()
    scores = deepcopy(metadata['roundness_filter']['scores'])
    for row in scores:
        row['accepted'] = row['sphericity'] >= minimum
    accepted = {row['cell_id'] for row in scores if row['accepted']}
    original = labels.copy()
    original[~np.isin(original, list(accepted))] = 0
    final = final_labels.copy()
    final[~np.isin(final, list(accepted))] = 0
    kept = {int(cell) for cell in np.unique(final) if cell}
    meta = deepcopy(metadata)
    meta['roundness_filter'] = dict(minimum=minimum, scores=scores,
                                    rejected=sum(not row['accepted'] for row in scores))
    meta['cells'] = [row for row in meta['cells'] if row['label'] in accepted]
    for name, array in [('unfiltered', original), ('final', final)]:
        np.savez_compressed(folder / f'{name}.npz', labels=array,
                            metadata=np.array(json.dumps(meta)))
    result = deepcopy(report)
    result.update(unfiltered_cell_predictions=str(folder / 'unfiltered.npz'),
                  cell_predictions=str(folder / 'final.npz'),
                  roundness_filter=meta['roundness_filter'])
    if 'signal_pairs' in result:
        result['signal_pairs'] = [pair for pair in result['signal_pairs'] if pair['cell_id'] in accepted]
    result['points'] = [point for point in result['points'] if point['cell_id'] in accepted]
    result['counts'] = [row for row in result['counts'] if row['cell_id'] in kept]
    result['cell_volumes'] = [row for row in result['cell_volumes'] if row['cell_id'] in kept]
    result['signal_filter'].update(original_cell_count=len(accepted), kept_cell_count=len(kept),
                                    removed_cell_count=len(accepted) - len(kept))
    save(folder / 'report.json', result)
    return result


def verify_signals(case, shaped_report, folder, device='auto'):
    """Prove reuse against an ordinary second signal worker on a stricter gate."""
    with (folder / 'verify-signals.log').open('w') as stream:
        subprocess.run([sys.executable, '-B', '-m', 'fish_viewer.signals', '--mode', 'apply',
                        '--image', case['image'], '--cell-predictions', shaped_report['unfiltered_cell_predictions'],
                        '--model', str(default_models()[1]), '--output', str(folder / 'verify'), '--device', device],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    events = []
    for line in (folder / 'verify-signals.log').read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    fresh = next(e['report'] for e in reversed(events) if e.get('kind') == 'finished')
    if fresh['status'] != 'complete':
        raise ValueError(fresh.get('error', 'Signal verification failed'))
    key = lambda point: (point['cell_id'], point['channel'], *point['xyz'])
    expected = sorted(shaped_report['points'], key=key)
    actual = sorted(fresh['points'], key=key)
    if len(expected) != len(actual):
        raise ValueError('Signal reuse changes detection count')
    for left, right in zip(expected, actual):
        if (left['cell_id'], left['channel']) != (right['cell_id'], right['channel']):
            raise ValueError('Signal reuse changes cell/channel assignments')
        np.testing.assert_allclose(left['xyz'], right['xyz'], rtol=0, atol=1e-5)
        np.testing.assert_allclose(left['score'], right['score'], rtol=1e-5, atol=1e-6)
    return dict(image=case['image'], minimum=.85, signals=len(actual),
                passed=True, xyz_tolerance_um=1e-5, score_atol=1e-6)


def validate_complete(rows, identity):
    expected = {(image, threshold) for image in identity['channels'] for threshold in identity['thresholds']}
    actual = [(row['image'], row['minimum_roundness']) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Every scan/threshold pair must complete exactly once before publishing')
    if any(row['cell_channel'] != identity['channels'][row['image']] for row in rows):
        raise ValueError('Evaluation used a different membrane channel')


def publish(output, rows, identity, verification):
    validate_complete(rows, identity)
    summaries = {}
    for minimum in THRESHOLDS:
        subset = [row for row in rows if row['minimum_roundness'] == minimum]
        summaries[str(minimum)] = summarize(subset)['test']
        overlaps = np.asarray([pair['iou'] for row in subset for pair in row['volume_pairs'] if not pair['cut']])
        summaries[str(minimum)]['contour_overlap'] = dict(
            n=len(overlaps), mean_iou=float(overlaps.mean()) if len(overlaps) else None,
            median_iou=float(np.median(overlaps)) if len(overlaps) else None,
            p10_iou=float(np.percentile(overlaps, 10)) if len(overlaps) else None,
            matches_at_iou_0_75=int(np.count_nonzero(overlaps >= .75)),
            matches_at_iou_0_90=int(np.count_nonzero(overlaps >= .90)))
    common_volumes = set.intersection(*[
        {(row['image'], pair['reference_id']) for row in rows if row['minimum_roundness'] == threshold
         for pair in row['volume_pairs'] if not pair['cut']} for threshold in THRESHOLDS])
    signal_key = lambda row, pair: (row['image'], pair['reference_id'], pair['channel'], tuple(pair['manual_xyz']))
    common_distances = set.intersection(*[
        {signal_key(row, pair) for row in rows if row['minimum_roundness'] == threshold
         for pair in row['distance_pairs'] if not row['cut_references'][pair['reference_id']]}
        for threshold in THRESHOLDS])
    common = {}
    common_signal = {}
    for minimum in THRESHOLDS:
        pairs = [dict(pair, group=row['group']) for row in rows if row['minimum_roundness'] == minimum
                 for pair in row['volume_pairs'] if (row['image'], pair['reference_id']) in common_volumes]
        common[str(minimum)] = paired_stats(pairs, 'manual_um3', 'predicted_um3', True)
        signals = [dict(pair, group=row['group']) for row in rows if row['minimum_roundness'] == minimum
                   for pair in row['distance_pairs'] if signal_key(row, pair) in common_distances]
        common_signal[str(minimum)] = paired_stats(signals, 'manual_um', 'predicted_um')
    report = dict(status='complete', identity=identity, results=rows, summary=summaries,
                  common_manual_volume_cohort=common, common_manual_distance_cohort=common_signal,
                  signal_reuse_verification=verification,
                  policy=f"All {identity['scans']} supplied test scans; frozen models and reviewed channels. "
                  'Both detectors regenerated once; unchanged per-cell signal inference reused across rejection gates. '
                  'Manual cells matched one-to-one at IoU >=0.5; signal centers within 0.6 um in the same cell/channel. '
                  'Volume and distance errors exclude cut manual cells; distances use original manual A-B lines. '
                  'All outliers included. Sparse manual labels do not establish whole-field precision. '
                  'This threshold sweep on a historically reused test fold is exploratory, not an independent threshold-selection validation.')
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'summary.json', report)
    tables = {'per-scan': [], 'volumes': [], 'distances': []}
    for row in rows:
        scan = {key: value for key, value in row.items()
                if isinstance(value, (str, int, float, bool))}
        for stage in ('before_filter', 'final'):
            scan[f'matched_complete_{stage}'] = sum(
                not row['cut_references'][pair['reference_id']]
                for pair in row[f'matches_{stage}'])
        tables['per-scan'].append(scan)
        for table, field in [('volumes', 'volume_pairs'), ('distances', 'distance_pairs')]:
            tables[table].extend(dict(pair, image=row['image'], group=row['group'],
                                      cut=row['cut_references'][pair['reference_id']],
                                      minimum_roundness=row['minimum_roundness']) for pair in row[field])
    for name, table in tables.items():
        with (output / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            fields = list(dict.fromkeys(key for row in table for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    def formatted(value, digits):
        return 'n/a' if value is None else f'{value:.{digits}f}'
    lines = ['# Consensus roundness sweep', '', report['policy'], '',
             '| Roundness | Cells before signal filtering | Retained cells | Matched complete manual cells | Volume MAPE | Distance MAE (µm) | Distance pairs |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for minimum in THRESHOLDS:
        row = summaries[str(minimum)]
        lines.append(f"| {minimum:.0%} | {row['cells_before_filter']} | {row['final_cells']} | "
                     f"{row['matched_complete_final']}/{row['complete_references']} | "
                     f"{formatted(row['volume'].get('mape_percent'), 2)}% | {formatted(row['distance'].get('mae'), 3)} | {row['distance']['n']} |")
    lines.extend(['', 'Model/input hashes, per-scan counts and all individual measurement pairs are retained in '
                  '[summary.json](summary.json), [per-scan.csv](per-scan.csv), [volumes.csv](volumes.csv) and [distances.csv](distances.csv).',
                  'Both measurement CSVs flag cut cells explicitly; exclude rows with `cut=True` to reproduce the primary metrics.',
                  '', 'A higher gate does not reshape a retained cell or change its per-cell signal prediction. '
                  'Changes in aggregate error therefore primarily reflect which cells/signals survive. '
                  'The JSON also reports volume error on the manual-cell cohort common to all thresholds.'])
    lines.extend(['', '## Error tails for a precision-first workflow', '',
                  'These are errors among matched complete manual cells/signals, with all outliers retained.', '',
                  '| Roundness | 95th-percentile volume error | 95th-percentile distance error (µm) |',
                  '|---|---:|---:|'])
    for minimum in THRESHOLDS:
        row = summaries[str(minimum)]
        lines.append(f"| {minimum:.0%} | {formatted(row['volume'].get('p95_ape_percent'), 2)}% | "
                     f"{formatted(row['distance'].get('p95_absolute_error'), 3)} |")
    lines.extend(['', '## Contour overlap with complete manual cells', '',
                  '| Roundness | Median IoU among matches | Matches at IoU ≥0.75 | Matches at IoU ≥0.90 |',
                  '|---|---:|---:|---:|'])
    for minimum in THRESHOLDS:
        row = summaries[str(minimum)]['contour_overlap']
        lines.append(f"| {minimum:.0%} | {formatted(row['median_iou'], 3)} | "
                     f"{row['matches_at_iou_0_75']} | {row['matches_at_iou_0_90']} |")
    lines.extend(['', f"Common cohorts across all five gates: {len(common_volumes)} complete manual cells and "
                  f"{len(common_distances)} matched manual signals.", '',
                  '![Cell retention and measurement errors](comparison.png)', '',
                  'The common-cohort lines compare the same manual identities across gates. '
                  'Shaded intervals show 95% acquisition-bootstrap intervals; changing matched populations '
                  'must be considered when comparing aggregate errors.'])
    lines.extend(['', '## Automatic selection: proposed next step', '',
                  'Calibrate on acquisition-separated validation data, using a specified quality target '
                  '(for example, an upper bound on volume error or on the fraction of inaccurate contours). '
                  'Choose the lowest roundness gate meeting that target with adequate confidence; '
                  'this retains the most cells within the requested quality limit. Mean error alone '
                  'does not control occasional large contour failures.', '',
                  'For adaptation to new scans, train a contour-quality estimator using predicted masks '
                  'and raw membrane crops, with roundness, boundary support and detector agreement as '
                  'inputs. Predicting mask quality separately from detection confidence has precedent in '
                  '[Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241); applying it here would require '
                  'training and validation for these 3D microscopy data. Fully reviewed regions, including '
                  'rejected and false-positive candidates, are needed to calibrate a reliable acceptance rate. '
                  'The existing sparse manual cells can assess matched-cell geometry, but cannot label '
                  'every unmatched prediction as wrong.', '',
                  'Keep the final threshold-selection calibration separate from this exploratory test sweep. '
                  'The current app still uses its manual minimum-roundness setting; no Auto selector or new '
                  'quality model is introduced by this benchmark.'])
    (output / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    plot_comparison(output, report)
    return report


def plot_comparison(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.asarray(THRESHOLDS) * 100
    summaries = [report['summary'][str(value)] for value in THRESHOLDS]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for field, color, label in [('final_cells', '#3277a8', 'Retained predictions'),
                                ('matched_complete_final', '#b26c23', 'Matched complete manual cells')]:
        values = [row[field] for row in summaries]
        axes[0].plot(x, values, 'o-', color=color, label=label)
        for a, b in zip(x, values):
            axes[0].annotate(str(b), (a, b), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8)
    axes[0].set(ylabel='Cells', title='Cell retention')
    axes[0].legend(fontsize=8)
    for axis, field, key, common_field, interval, title, unit in [
            (axes[1], 'volume', 'mape_percent', 'common_manual_volume_cohort',
             'mape_95ci_acquisition_bootstrap', 'Volume error', 'Mean absolute percentage error (%)'),
            (axes[2], 'distance', 'mae', 'common_manual_distance_cohort',
             'mae_95ci_acquisition_bootstrap', 'A–B distance error', 'Mean absolute error (µm)')]:
        values = [row[field].get(key, np.nan) for row in summaries]
        axis.plot(x, values, 'o-', color='#3277a8', label='All matched at each gate')
        bounds = [row[field].get(interval, [np.nan, np.nan]) for row in summaries]
        axis.fill_between(x, np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1], alpha=.15, color='#3277a8')
        paired = [report[common_field][str(value)].get(key, np.nan) for value in THRESHOLDS]
        axis.plot(x, paired, 's--', color='#b26c23', label='Same identities at every gate')
        for a, b, row in zip(x, values, summaries):
            axis.annotate(f"n={row[field]['n']}", (a, b), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
        axis.set(title=title, ylabel=unit)
        axis.legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel('Minimum roundness (%)')
        axis.set_xticks(x)
        axis.grid(alpha=.2)
    figure.suptitle(f"Both-detector consensus · {report['identity']['scans']} supplied test scans · frozen models")
    figure.savefig(output / 'comparison.png', dpi=180)
    figure.savefig(output / 'comparison.svg')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True, help='Resumable metrics; temporary arrays are removed per scan')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--channel-audit', type=Path, help='Optional reviewed channel metadata')
    parser.add_argument('--device', choices=['auto','cpu','gpu'], default='auto')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--start-index', type=int, default=0, help='Execution offset; retain the frozen full test-set identity')
    parser.add_argument('--share-results', type=Path, help='Share completed compact metrics with an identically frozen run')
    parser.add_argument('--verify-signal-reuse', action='store_true')
    args = parser.parse_args()
    collection_path = args.audit.resolve()
    channel_path = (args.channel_audit or args.audit).resolve()
    collection = json.loads(collection_path.read_text())
    cases = sorted([case for case in collection['cases'] if case['audit']['split'] == 'test'], key=lambda case: case['image'])
    if not cases:
        raise ValueError('No test scans in the supplied audit.')
    channel_cases = {case['image']: case for case in json.loads(channel_path.read_text())['cases']}
    channels = {case['image']: channel_for(channel_cases[case['image']])[0] for case in cases}
    models = frozen_models()
    training_groups = {case['audit']['group'] for case in collection['cases'] if case['audit']['split'] == 'training'}
    if training_groups & {case['audit']['group'] for case in cases}:
        raise ValueError('Training/test leakage')
    for source in collection['sources']:
        if not any(source['image'] in case['audit']['aliases'] for case in cases):
            continue
        for item in [dict(path=source['image'], size=source['size'], mtime_ns=source['mtime_ns']), *source.get('scene_stats', [])]:
            stat = Path(item['path']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
                raise ValueError('Audited source changed: ' + item['path'])
    identity = dict(models=models, audit_sha256=sha(collection_path), channel_audit_sha256=sha(channel_path),
                    channels=channels, thresholds=list(THRESHOLDS), scans=len(cases), device=args.device)
    args.work.mkdir(parents=True, exist_ok=True)
    identity_path = args.work / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Resume identity changed')
    save(identity_path, identity)
    if args.share_results and json.loads((args.share_results / 'identity.json').read_text()) != identity:
        raise ValueError('Shared inference identity differs')
    verification_path = args.work / 'signal-reuse-verification.json'
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else None
    rows, active = [], {}
    app = QtCore.QCoreApplication([])
    controller = AnalysisController()
    original_launch = controller._launch
    started, newly_done = time.monotonic(), 0

    def progress(event):
        save(args.work / 'progress.json', dict(completed_scans=len(rows) // len(THRESHOLDS), total_scans=len(cases),
             scan=active.get('image'), stage=event.get('phase'), elapsed_s=time.monotonic() - started))

    def fail(message):
        save(args.work / 'failure.json', dict(scan=active.get('image'), error=message))
        print(message, flush=True)
        app.exit(1)

    def launch(index, arguments):
        if controller.stages[index][0] == 'export':
            controller.state.update(status='complete', benchmark_only=True, export_omitted=True)
            controller._stop()
            complete(dict(controller.state, report_path=str(controller.directory / 'analysis-result.json')))
        else:
            original_launch(index, arguments)

    def complete(state):
        nonlocal verification, newly_done
        try:
            scratch, case = active['scratch'], active['case']
            run = Path(state['report_path']).parent
            measured = json.loads((run / state['signal_report']).read_text())
            with np.load(measured['unfiltered_cell_predictions'], allow_pickle=False) as archive:
                labels, metadata = archive['labels'], json.loads(str(archive['metadata']))
            with np.load(measured['cell_predictions'], allow_pickle=False) as archive:
                final_labels = archive['labels']
            rebuilt = rebuild_references(case, collection, scratch / 'references')
            results = []
            for minimum in THRESHOLDS:
                folder = scratch / f'gate-{minimum:.2f}'
                shaped = gated_result(measured, metadata, labels, final_labels, minimum, folder)
                if args.verify_signal_reuse and verification is None and minimum == .85 and shaped['points']:
                    verification = verify_signals(case, shaped, folder, args.device)
                    save(verification_path, verification)
                current = dict(report_path=str(folder / 'pipeline.json'), signal_report=str(folder / 'report.json'))
                row = evaluate_run(rebuilt, current, collection, scratch / 'references')
                row.update(minimum_roundness=minimum, cut_references=case['audit']['reference_cut'],
                           cell_channel=channels[case['image']])
                row.pop('pipeline_report', None)
                results.append(row)
            save(args.work / f"scan-{active['index']:03d}.json", results)
            if args.share_results:
                shared = args.share_results / f"scan-{active['index']:03d}.json"
                progress_path = args.share_results / 'progress.json'
                running = json.loads(progress_path.read_text()).get('scan') if progress_path.exists() else None
                if not shared.exists() and running != case['image']:
                    save(shared, results)
            rows.extend(results)
            newly_done += 1
            active['temporary'].cleanup()
            print(f"COMPLETE {len(rows) // len(THRESHOLDS)}/{len(cases)} {Path(case['image']).name}: "
                  + ', '.join(f"{row['minimum_roundness']:.0%}={row['final_cells']} cells/{sum(not row['cut_references'][m['reference_id']] for m in row['matches_final'])} manual" for row in results), flush=True)
            QtCore.QTimer.singleShot(0, next_scan)
        except Exception:
            import traceback
            fail(traceback.format_exc())

    def next_scan():
        if args.stop_after and newly_done >= args.stop_after:
            app.quit()
            return
        index = args.start_index + len(rows) // len(THRESHOLDS)
        if index == len(cases):
            app.quit()
            return
        saved = args.work / f'scan-{index:03d}.json'
        if not saved.exists() and args.share_results:
            shared = args.share_results / saved.name
            if shared.exists():
                save(saved, json.loads(shared.read_text()))
        if saved.exists():
            rows.extend(json.loads(saved.read_text()))
            QtCore.QTimer.singleShot(0, next_scan)
            return
        if shutil.disk_usage(args.work).free < 1_000_000_000:
            fail('Less than 1 GB free for temporary inference arrays')
            return
        temporary = tempfile.TemporaryDirectory(prefix='scan-', dir=args.work)
        case = cases[index]
        active.update(index=index, case=case, image=case['image'], scratch=Path(temporary.name), temporary=temporary)
        print(f"START {index + 1}/{len(cases)} {case['image']}", flush=True)
        controller.start(case['image'], Path(temporary.name) / 'runs', sensitive_stardist_model(), default_models()[1],
                         device=args.device, refinement=default_refinement_model(), stardist_proposals=True,
                         consensus_model=default_proposal_model(), roundness=.50, cell_channel=channels[case['image']])

    controller._launch = launch
    controller.failed.connect(fail)
    controller.completed.connect(complete)
    controller.progress.connect(progress)
    QtCore.QTimer.singleShot(0, next_scan)
    code = app.exec()
    if code:
        return code
    if len(rows) == len(cases) * len(THRESHOLDS):
        publish(args.output, rows, identity, verification)
    print(f"Completed scans: {len(rows) // len(THRESHOLDS)}/{len(cases)}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
