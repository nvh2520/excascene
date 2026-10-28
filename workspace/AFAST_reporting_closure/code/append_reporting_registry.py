"""Append actual reporting output provenance without altering historical rows."""
from pathlib import Path
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
path = OUT / 'publication_numbers_v5_2.json'
registry = json.loads(path.read_text())
registry['records'] = [r for r in registry['records'] if not r['id'].startswith('AFAST-reporting-')]
inventory = [
    ('secondary_descriptive_intervals.csv', '8: reporting-stage descriptive ordinary95 intervals'),
    ('posthoc_removal_interaction.csv', '9: explicitly post-hoc descriptive ordinary95 interactions'),
    ('mapping_decision_companions.csv', '8: reporting-stage descriptive ordinary95 intervals'),
    ('severe_BF_D_decision_transitions.csv', '10: saved-observation transitions'),
    ('primary_per_seed_leave_one_out_reconstructed.csv', '10: frozen primary seed effects and leave-one-seed-out diagnostics'),
]
for filename, evidence in inventory:
    source = OUT / filename
    with source.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row_number, row in enumerate(rows, 2):
        registry['records'].append({
            'id': 'AFAST-reporting-' + source.stem + '-row' + str(row_number),
            'source_file': 'AFAST_reporting_closure/' + filename,
            'source_sha256': digest(source),
            'row_selector': 'CSV row ' + str(row_number),
            'source_provenance': 'AFAST_reporting_closure/scalar_source_provenance.json',
            'source_provenance_sha256': digest(OUT / 'scalar_source_provenance.json'),
            'evidence_class': evidence,
            'values': row,
            'note': 'Actual reconstructed output. Original metric observations and saved draws only; reviewer targets were cross-checks, not calculation inputs. No new bootstrap draws or model predictions.',
        })
for filename, meaning in [
    ('scalar_reconstruction_summary.json', 'Counts, primary reconstruction, transitions and closure scope'),
    ('class_mapping.json', 'Actual simulator semantic/material class mapping'),
    ('condition_denominators.csv', 'Complete absolute attempts, executions, unsafe events, support and undefined quantities'),
    ('per_seed_condition_results_reconstructed.csv', 'All fitted-seed condition results; no seed selection'),
    ('pixel_class_diagnostics.csv', 'Full per-class IoU, truth and prediction supports; zero union only is undefined'),
    ('pixel_confusion_matrices_long.csv', 'Full semantic and material confusion matrices'),
    ('hazard_and_material_support.csv', 'Material support and hazard precision/recall denominators'),
    ('optimization_epoch_trajectories.csv', 'All six saved 14-epoch trajectories; component losses and absolute errors'),
]:
    source = OUT / filename
    registry['records'].append({
        'id': 'AFAST-reporting-' + source.stem,
        'source_file': 'AFAST_reporting_closure/' + filename,
        'source_sha256': digest(source), 'row_selector': 'all; named columns/keys',
        'evidence_class': '10: saved-observation scalar diagnostics',
        'note': meaning + '; no additional primary endpoint or mechanism experiment',
    })
registry['v5_2_scope']['reporting_inventory'] = {
    'descriptive_interval_rows': 180,
    'posthoc_interaction_rows': 42,
    'no_new_draws': True,
    'all_reporting_rows_source_hashed': True,
}
path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n')

# Correct current added locator fields only; the original52rows remain identical.
claim_path = OUT / 'claim_evidence_v5_2.csv'
with claim_path.open(newline='') as handle:
    reader = csv.DictReader(handle); fields = reader.fieldnames; rows = list(reader)
for row in rows[52:]:
    for key in ('source_artifact', 'verified_v5_evidence'):
        row[key] = row[key].replace('mapping_companions.csv', 'mapping_decision_companions.csv')
        if row[key] == 'decision_transitions.csv':
            row[key] = 'severe_BF_D_decision_transitions.csv'
with claim_path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
print(json.dumps({'status':'REPORTING_REGISTRY_APPENDED','records':len(registry['records'])}))
