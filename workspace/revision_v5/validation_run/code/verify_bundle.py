"""Verify included file digests and reconstruct headlines from extracted ZIP members."""
from common import *
import zipfile,io,subprocess

def main():
    with zipfile.ZipFile(RUN/'evidence_bundle.zip') as z:
        manifest=list(csv.DictReader(io.StringIO(z.read('bundle_file_hashes.csv').decode())))
        for row in manifest:
            assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256'],row['path']
        dest=RUN/'review_bundle_check';assert not dest.exists();dest.mkdir()
        files=['code/inspect_headlines.py','headline_frame_metrics.csv','primary_results.csv','manifests/generation_manifest.json','manifests/bootstrap_multiplicities.npz']
        for name in files:z.extract(name,dest)
    result=subprocess.run([sys.executable,str(dest/'code/inspect_headlines.py'),str(dest)],capture_output=True,text=True)
    print(result.stdout,flush=True);print(result.stderr,flush=True)
    assert result.returncode==0
    js(RUN/'bundle_review_check.json',dict(status='PASS',scientific_file_hashes_verified=len(manifest),standalone_files=files,headline_reconstruction=json.loads(result.stdout),scope='Extracted compact bundle, no original repository imports, checkpoints or GPU required.'))
    event('portable_bundle_verified',files=len(manifest),headline_slots=4)

if __name__=='__main__':main()
