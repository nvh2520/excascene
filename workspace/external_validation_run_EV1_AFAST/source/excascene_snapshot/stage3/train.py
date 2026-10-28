"""Frozen training loop with isolated dataset/model factories; no source modification."""
import sys
from stage3.common import output_path

def main():
 import kinafford.train as original
 from stage3.dataset import Stage3Dataset,EpochDataLoader
 from stage3.models import build_model
 if '--out' not in sys.argv:raise SystemExit('Stage-3 --out required')
 output_path(sys.argv[sys.argv.index('--out')+1],('runs_stage3',))
 load=original.load_config;holder={}
 def config(args):
  cfg=load(args);holder.update(cfg);return cfg
 def dataset(*args,**kw):
  m=holder['model']
  return Stage3Dataset(*args,**kw,modality_dropout=m.get('modality_dropout',0),epoch_rng=m.get('epoch_rng',False),training_seed=holder['seed'])
 original.load_config=config;original.ExcaSceneDataset=dataset;original.DataLoader=EpochDataLoader;original.build_model=build_model
 original.main()
if __name__=='__main__':main()
