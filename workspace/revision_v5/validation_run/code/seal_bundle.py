"""Append complete scientific-job ledger and resource snapshot after metered packaging.

This lightweight metadata closure is not a model/analysis job. Its own elapsed/CPU
time is recorded separately after sealing; no recursive self-hash is claimed.
"""
from common import *
import zipfile,resource

def main():
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
    locked();ledger=[json.loads(x) for x in (RUN/'ledger.jsonl').read_text().splitlines()]
    done=[x for x in ledger if x['kind']=='job_end']
    starts=[x for x in ledger if x['kind']=='job_start']
    assert {x['job'] for x in starts}=={x['job'] for x in done}
    assert not [x for x in done if (x['returncode']!=0 and not (x.get('job')=='fresh' and x.get('scientific_completion_verified'))) or x.get('stop_reason')]
    counter=json.loads((RUN/'forward_counter.json').read_text())
    assert counter['total']==63216 and counter['precision']==1920 and counter['by_phase']['fresh']==58608
    used=storage();cpu=sum(x['cpu_seconds'] for x in done);gpu=sum(x['gpu_reserved_seconds'] for x in done);wall=sum(x['wall_seconds'] for x in done)
    elapsed=time.time()-json.loads((RUN/'manifests/artifact_lock.json').read_text())['unix_time']
    assert cpu<CAPS['cpu_seconds'] and gpu<CAPS['gpu_seconds'] and wall<CAPS['active_wall_seconds'] and elapsed<CAPS['elapsed_seconds'] and used<CAPS['storage_bytes']
    usage=dict(recorded_at=utc(),recorded_cpu_core_hours=cpu/3600,recorded_gpu_reserved_hours=gpu/3600,
      recorded_job_wall_hours=wall/3600,elapsed_hours_since_lock=elapsed/3600,additional_storage_bytes_before_metadata_seal=used,
      forwards=counter,training_steps=0,conditional_model_forwards=0,weight_network_bytes=0,raw_acquisition_bytes=0,
      max_generation_workers=8,all_metered_jobs_finished=True,caps_pass=True,
      accounting_note='GPU charged conservatively by the full original job wall reservation, including monitor interruption and pause. Fresh CPU is observed process-lifetime CPU plus a separately recorded conservative final polling-tail bound; its reparented OS exit code is unavailable, with scientific completion verified separately. Sum of job wall times double-counts overlaps. Scientific/package jobs are accounted in the ledger; lightweight interactive/tool overhead is not a process CPU census. Initial wrapper dependency failure has unavailable exact resource use/time. Final metadata seal is separately measured in bundle_seal.json.')
    js(RUN/'resource_summary.json',usage)
    event('evidence_bundle_sealing',all_scientific_jobs_complete=True,portable_reconstruction='PASS',primary_slots=4)
    files=['ledger.jsonl','resource_summary.json','bundle_review_check.json','logs/bundle.log','logs/verify_bundle.log']
    rows=[dict(path=p,sha256=sha(RUN/p),bytes=(RUN/p).stat().st_size) for p in files]
    csvwrite(RUN/'bundle_final_snapshot_hashes.csv',rows)
    with zipfile.ZipFile(RUN/'evidence_bundle.zip','a',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            assert p not in z.namelist();z.write(RUN/p,p)
        z.write(RUN/'bundle_final_snapshot_hashes.csv','bundle_final_snapshot_hashes.csv')
    digest=sha(RUN/'evidence_bundle.zip')
    (RUN/'evidence_bundle.zip.sha256').write_text(digest+'  evidence_bundle.zip\n')
    after=resource.getrusage(resource.RUSAGE_SELF)
    js(RUN/'bundle_seal.json',dict(recorded_at=utc(),zip_sha256=digest,zip_bytes=(RUN/'evidence_bundle.zip').stat().st_size,
       metadata_seal_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
       metadata_seal_wall_seconds=time.monotonic()-start,additional_storage_bytes=storage(),
       note='External closure receipt; not included in the ZIP to avoid recursive self-hashing. Scientific job accounting is inside resource_summary.json.'))
    print(digest,flush=True)

if __name__=='__main__':main()
