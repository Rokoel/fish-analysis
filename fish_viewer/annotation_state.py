"""Stable annotation identities, independent of display names and run ownership."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from uuid import NAMESPACE_URL, uuid5


COLLECTIONS = ("surfaces", "measurements", "contours")
FIELDS = ("uid", "run_id", "created_at", "cell_uid", "channel", "score", "name", "kind", "signal_uids")


def annotation_report(dataset):
    """Read metadata bound to the exact native scene, including standalone IMS."""
    source = dataset.scene.source
    if source is None:
        return {}
    try:
        if source.resolve() == dataset.metadata.path.resolve():
            import h5py
            with h5py.File(source, 'r') as image:
                if 'FISHAnalysis/Annotations' not in image or 'Scene/Data' not in image:
                    return {}
                report = json.loads(image['FISHAnalysis/Annotations'][()].tobytes())
                data = image['Scene/Data'][0].tobytes()
        else:
            sidecar = dataset.metadata.path.with_suffix('.annotations.json')
            if source.suffix.lower() != '.imx' or not sidecar.is_file():
                return {}
            report = json.loads(sidecar.read_text(encoding='utf-8'))
            data = source.read_bytes()
        return report if report.get('editor_state', {}).get('scene_sha256') == hashlib.sha256(data).hexdigest() else {}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def split_measurements(scene):
    """Keep independent point/pair objects editable without joining their lines."""
    groups = []
    for m in scene.measurements:
        if (len(m.edges) > 1 and all(len(set(e)) == 2 for e in m.edges)
                and len({v for e in m.edges for v in e}) == 2 * len(m.edges)):
            parts = m.edges
            label = "pair"
        elif len(m.points) > 1 and not m.edges:
            parts = [(i,) for i in range(len(m.points))]
            label = "point"
        else:
            groups.append(m)
            continue
        used = {i for part in parts for i in part}
        parts = list(parts) + [(i,) for i in range(len(m.points)) if i not in used]
        for n, part in enumerate(parts):
            item = deepcopy(m)
            item.uid = str(uuid5(NAMESPACE_URL, f'{m.uid}#points/{tuple(part)}'))
            item.name = f"{m.name} · {label} {n + 1}"
            item.points = m.points[list(part)].copy()
            item.labels = [m.labels[i] for i in part]
            item.times = [m.times[i] for i in part]
            item.edges = [(0, 1)] if len(part) == 2 else []
            groups.append(item)
    scene.measurements = groups


def scene_state(scene, data):
    return dict(
        scene_sha256=hashlib.sha256(data).hexdigest(),
        objects={key: [{field: export_name(obj) if field == "name" else getattr(obj, field)
                       for field in FIELDS if hasattr(obj, field)}
                       for obj in getattr(scene, key)] for key in COLLECTIONS},
    )


def read_scene_state(dataset):
    """Restore identities only from metadata bound to these exact scene bytes.

    Provenance persists across exports. Program ownership is deliberately *not*
    restored here; only opening a specific analysis result activates that run.
    """
    try:
        state = annotation_report(dataset).get('editor_state', {})
        records = state.get("objects", {})
        if any(len(records.get(key, [])) != len(getattr(dataset.scene, key)) for key in COLLECTIONS):
            return
        ids = [row["uid"] for key in COLLECTIONS for row in records[key]]
        if len(set(ids)) != len(ids) or not all(isinstance(uid, str) and uid for uid in ids):
            return
        for key in COLLECTIONS:
            for obj, row in zip(getattr(dataset.scene, key), records[key]):
                for field in FIELDS:
                    if field in row and hasattr(obj, field):
                        setattr(obj, field, row[field])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return


def activate_run(scene, run_id):
    for key in COLLECTIONS:
        for obj in getattr(scene, key):
            obj.source = "program" if run_id and obj.run_id == run_id else "manual"


def export_name(obj):
    if obj.source == "manual":
        return obj.name
    if obj.created_at is None:
        obj.created_at = timestamp()
    suffix = f" [FISH {obj.created_at}]"
    return obj.name if obj.name.endswith(suffix) else obj.name + suffix
