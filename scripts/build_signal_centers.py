"""Build U-Net cell crops from explicit sparse supervision or calibrated cell predictions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fish_viewer.spots.data import build
from fish_viewer.signals.data import build_supervision_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,help='Existing sparse signal-supervision audit')
    parser.add_argument('--collection-audit',type=Path)
    parser.add_argument('--cell-report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.audit and (args.collection_audit or args.cell_report):
        parser.error('Choose either --audit or --collection-audit with --cell-report.')
    if not args.audit and not (args.collection_audit and args.cell_report):
        parser.error('Supply --audit, or both --collection-audit and --cell-report.')
    emit=lambda event:print(json.dumps(event),flush=True)
    audit=args.audit or build_supervision_audit(args.cell_report,args.collection_audit,
                                              args.output.with_name(args.output.name+'-supervision'),emit)
    build(audit,args.output,emit=emit)


if __name__=='__main__':main()
