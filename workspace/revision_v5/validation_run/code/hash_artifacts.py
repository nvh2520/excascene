"""Final byte custody of used inputs and retained generated arrays, one CPU worker."""
from common import *

def main():
    locked(); checks=[]
    for filename, base in [('generated_files.csv',RUN), ('source_input_hashes.csv',ROOT), ('rough_support_references.csv',RUN)]:
        with (RUN/'manifests'/filename).open() as f:
            for r in csv.DictReader(f):
                p=base/r['path'];actual=sha(p)
                assert actual==r['sha256'],('input mutated',filename,r['path'])
                checks.append(dict(manifest=filename,path=r['path'],sha256=actual,verified=True))
    csvwrite(RUN/'manifests/input_reverification.csv',checks)
    rows=[]
    for directory in ['predictions','precision/maps','latents','support_reference','generation','metrics','metrics_corrected']:
        count=0
        for p in sorted((RUN/directory).rglob('*')):
            if p.is_file():
                rows.append(dict(path=str(p.relative_to(RUN)),bytes=p.stat().st_size,sha256=sha(p)));count+=1
        print(directory,count,'hashed',flush=True)
    csvwrite(RUN/'manifests/retained_artifact_hashes.csv',rows)
    event('final_input_and_retained_array_hashes',verified_inputs=len(checks),retained_files=len(rows),bytes=sum(r['bytes'] for r in rows))

if __name__=='__main__':main()
