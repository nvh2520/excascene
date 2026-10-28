"""Run the frozen training loop with new process-local dataset/model factories."""
import sys
from stage4.common import output_path,write_json
def main():
 import kinafford.train as original
 from stage4.dataset import Stage4Dataset,EpochDataLoader
 from stage4.models import build_model
 if '--out' not in sys.argv:raise SystemExit('Stage-4 --out required')
 out=output_path(sys.argv[sys.argv.index('--out')+1],('runs_stage4',));load=original.load_config;holder={};datasets=[]
 def config(args):
  cfg=load(args);holder.update(cfg);return cfg
 def dataset(*a,**kw):
  m=holder['model'];d=Stage4Dataset(*a,**kw,structured_dropout=m.get('structured_dropout',False),epochs=holder['epochs'],training_seed=holder['seed'])
  if d.train:datasets.append(d)
  return d
 original.load_config=config;original.ExcaSceneDataset=dataset;original.DataLoader=EpochDataLoader;original.build_model=build_model
 original.main()
 for d in datasets:
  write_json(out/'dropout_exposures.json',dict(seed=holder['seed'],tag=holder['variant'],source='observed persistent-worker accesses',epochs=d.exposure_records()))
if __name__=='__main__':main()
