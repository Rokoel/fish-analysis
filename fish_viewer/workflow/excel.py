"""A single measurement sheet containing cell volumes and signal distances."""
from pathlib import Path
import os
import tempfile

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill


def write_excel(path, report):
    from .manual import manual_measurements
    manual = manual_measurements(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.remove(workbook.active)

    def sheet(name, headers, rows):
        page = workbook.create_sheet(name)
        page.append(headers)
        for row in rows:
            page.append(row)
            for cell in page[page.max_row]:
                if isinstance(cell.value, str):
                    cell.data_type = 's'
                elif isinstance(cell.value, float):
                    cell.number_format = '0.0000'
        page.freeze_panes = 'A2'
        page.auto_filter.ref = page.dimensions
        for cell in page[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='205E58')
        for column in page.columns:
            page.column_dimensions[column[0].column_letter].width = min(55, max(16, max(len(str(c.value or '')) for c in column) + 2))

    rows=[]
    volumes={c['cell_id']:c.get('volume_um3')
             for c in [*manual['cells'], *report.get('cell_volumes', [])]}
    for source, cells, points in [('Manual', manual['cells'], manual['points']),
                                  ('Program', report.get('cell_volumes', []), report.get('points', []))]:
        covered=set()
        for point in points:
            channel=point.get('channel')
            name=point.get('channel_name')
            if name is None and channel is not None:
                name=report['channels'][channel]['name']
            cell=point.get('cell_id'); covered.add(cell)
            rows.append([source,point.get('cell_name',cell),volumes.get(cell),point.get('signal_name',point['signal_id']),
                         channel+1 if channel is not None else None,name,point.get('distance_um')])
        rows.extend([source,c.get('cell_name',c['cell_id']),c.get('volume_um3'),None,None,None,None] for c in cells if c['cell_id'] not in covered)
    sheet('Measurements', ['Source','Cell','Volume (µm³)','Signal','Channel (1-based)','Channel name','Distance (µm)'], rows)
    if report.get('signal_pairs'):
        sheet('Signal pairs', ['Source', 'Cell', 'Time point (1-based)', 'Signal A', 'Channel A (1-based)',
                               'Channel A name', 'Signal B', 'Channel B (1-based)', 'Channel B name', 'Distance (µm)',
                               'Maximum pair distance (µm)'],
              [[p.get('source', 'Program'), p.get('cell_name', p['cell_id']), p.get('time', 0) + 1,
                p.get('signal_name_a', p['signal_id_a']), p['channel_a'] + 1, p.get('channel_name_a'),
                p.get('signal_name_b', p['signal_id_b']), p['channel_b'] + 1, p.get('channel_name_b'), p['distance_um'],
                report.get('signal_pairing', {}).get('maximum_distance_um')] for p in report['signal_pairs']])
    fd, temporary = tempfile.mkstemp(suffix='.xlsx', prefix='.fish-', dir=path.parent)
    os.close(fd)
    try:
        workbook.save(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
        workbook.close()
    return path
