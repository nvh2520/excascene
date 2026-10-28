"""Create authorized-population multiplicities once, before any test inference."""
from common import *
import datetime
sys.path.insert(0,str(AFAST/'code'))
from statistical_core import multiplicities
assert not (ROOT/'study3/metrics').exists()
assert not (ROOT/'study3/predictions').exists()
assert not (ROOT/'study3/holdout/gate_scores.csv').exists()
auth=json.loads((ROOT/'protocol/deviations/2026_10_19_scene800034/authorization_and_absence.json').read_text())
assert auth['status']=='AUTHORIZED_PRE_INFERENCE_EXCLUSION' and auth['excluded_scene']==800034
cfg=json.loads((ROOT/'protocol/holdout_manifest.json').read_text())
ids=np.array(sorted(set(cfg['nominal_scene_ids'])-{800034}));rough=np.array(sorted(cfg['rough_scene_ids']))
assert len(ids)==511 and len(rough)==256
out=ROOT/'study3/bootstrap_plan';out.mkdir(exist_ok=False)
rng=np.random.default_rng(2026101808)
draws={'nominal_scene_ids':ids,'rough_scene_ids':rough,'nominal':multiplicities(ids,rng),'rough':multiplicities(rough,rng)}
np.savez_compressed(out/'bootstrap_multiplicities.npz',**draws)
js(out/'receipt.json',dict(status='CREATED_ONCE_AFTER_AUTHORIZED_EXCLUSION_BEFORE_TEST_INFERENCE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=2026101808,draws=20000,nominal_scenes=511,rough_scenes=256,nominal_shape=list(draws['nominal'].shape),rough_shape=list(draws['rough'].shape),sha256=sha(out/'bootstrap_multiplicities.npz'),code_sha256=sha(__file__),sampler_sha256=sha(AFAST/'code/statistical_core.py'),note='No prior Study3 multiplicities existed to replace. This is their first creation on the authorized retained population, with unchanged frozen seed and sequential nominal/rough sampler; all later comparisons reuse this file.'))
