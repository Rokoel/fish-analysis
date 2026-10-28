"""Locations of bundled models, regenerated caches and archived training metadata."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def bundled_model(name):
    return ROOT / 'models' / name


def cached_audit(family):
    return ROOT / 'artifacts' / family / 'audit' / 'audit.json'
