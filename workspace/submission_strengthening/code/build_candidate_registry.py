"""Append source-linked descriptive/reproduction records; preserve v5.2 prefix."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'submission_strengthening'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old_path=ROOT/'AFAST_reporting_closure/publication_numbers_v5_2.json'
    old=json.loads(old_path.read_text())
    registry=json.loads(old_path.read_text())
    registry['schema_version']=4
    registry['predecessor']={'path':str(old_path.relative_to(ROOT)),'sha256':sha(old_path),'preserved_prefix_records':len(old['records'])}
    registry['evidence_classes'].update({
        '12':'New post-hoc descriptive simple references on inspected AFAST holdout; saved draws only; unadjusted95',
        '13':'New post-hoc existing geometric predictor with declared v5 scoring interface; saved draws only; unadjusted95',
        '14':'Post-hoc pooled-pixel material/support diagnostic and training-selected reference; no scene-primary inference',
        '15':'Bounded reproduction engineering checks; not new scientific cohort or training'})
    sources=[('references/reference_comparison_intervals.csv','12','reference'),
             ('references/geometric_descriptive_intervals.csv','13','anchor'),
             ('material/complete_v1/summary_metrics.csv','14','pixel_summary'),
             ('material/complete_v1/class_metrics.csv','14','pixel_class'),
             ('material/complete_v1/hazard_head_metrics.csv','14','hazard_counts')]
    for rel,cls,prefix in sources:
        path=OUT/rel; digest=sha(path)
        rows=list(csv.DictReader(path.open()))
        for index,row in enumerate(rows,2):
            registry['records'].append({'id':f'{prefix}_{index-1:04d}',
                'source_file':str(path.relative_to(ROOT)),'source_sha256':digest,
                'row_selector':f'CSV row {index}',
                'endpoint':row.get('metric',row.get('task','hazard pixel counts')),
                'units':row.get('unit',row.get('display_unit','counts / probability; display accuracy percent')),
                'population':row.get('condition',''),
                'checkpoint':'existing AFAST final14 epochs; three fitted seeds where applicable; deterministic references once',
                'selector':row.get('model_policy',row.get('policy','pixel diagnostic, no decision selector')),
                'mask':row.get('camera_mask',row.get('mask',row.get('camera_state',''))),
                'weights':row.get('weighting','pooled pixels; repeated same support across fitted seeds'),
                'seed_scope':row.get('seed','10;11;12 separately scored where applicable; deterministic reference once'),
                'interval_method':row.get('interval','saved20k scene percentile' if cls in ('12','13') else 'none: descriptive pixel counts'),
                'interval_level':'ordinary95; unadjusted' if cls in ('12','13') else 'not applicable',
                'evidence_class':cls+': '+registry['evidence_classes'][cls],
                'values':row,'note':'New reporting family only; preserves every original/corrected primary and historical record.'})
    jsons=[('references/training_material_prior.json','14','training_material_prior'),
           ('material/complete_v1/diagnostic_receipt.json','14','material_diagnostic_receipt'),
           ('material/convention_clarification_v1.json','14','probability_tie_convention'),
           ('reproduction/checkpoint_smoke64_recovery1/checkpoint_reproduction.json','15','checkpoint_replay'),
           ('reproduction/generator_smoke7/generator_reproduction.json','15','generator_replay')]
    for rel,cls,ident in jsons:
        path=OUT/rel
        if not path.exists():
            raise FileNotFoundError(path)
        registry['records'].append({'id':ident,'source_file':str(path.relative_to(ROOT)),'source_sha256':sha(path),'row_selector':'JSON object','evidence_class':cls+': '+registry['evidence_classes'][cls],'values':json.loads(path.read_text()),'interval_level':'not applicable'})
    assert registry['records'][:len(old['records'])]==old['records']
    target=OUT/'publication_numbers_submission_candidate.json'
    target.write_text(json.dumps(registry,indent=2,allow_nan=False)+'\n')
    copies=[ROOT/'paper_submission_candidate/data/strengthening',ROOT/'release_candidate/statistics/submission_candidate']
    for d in copies:
        d.mkdir(parents=True,exist_ok=True);shutil.copy2(target,d/target.name)
    receipt={'predecessor_records':len(old['records']),'new_records':len(registry['records'])-len(old['records']),'total_records':len(registry['records']),'predecessor_prefix_exact':True,'registry_sha256':sha(target),'source_files_new':len(sources)+len(jsons),'does_not_modify_original_registry':True}
    (OUT/'records/registry_construction.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
