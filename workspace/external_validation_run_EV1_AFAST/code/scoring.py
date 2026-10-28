"""Side-effect-free frozen v5 strict scorer definitions; no historical data access."""
import ast, pathlib, types
import numpy as np
from stage3.selectors import soft_composition
from kinafford.decision import compose_affordance
ROOT=pathlib.Path(__file__).resolve().parents[1]
original=ROOT.parent/'revision_v5/validation_run/code/selectors_v5.py'
frozen=ROOT/'source/selectors_v5_original.py'
if not frozen.exists():frozen.write_bytes(original.read_bytes())
tree=ast.parse(frozen.read_text())
nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef)]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(frozen),'exec'),globals())

def neutral(out,index):
    return dict(H=out['H'][index].detach().cpu().numpy().astype(np.float32),
        sem_prob=out['sem'][index].softmax(0).detach().cpu().numpy().astype(np.float32),
        mat_prob=out['mat'][index].softmax(0).detach().cpu().numpy().astype(np.float32),
        mat_label=out['mat'][index].argmax(0).detach().cpu().numpy().astype(np.int16),
        haz_prob=out['haz'][index].sigmoid().detach().cpu().numpy().astype(np.float32),
        aff=out['aff'][index].sigmoid().detach().cpu().numpy().astype(np.float32),
        aff_sigma=(out['aff_logvar'][index]/2).exp().detach().cpu().numpy().astype(np.float32))

def truth(batch,index):
    keys=['gt_H','gt_sem','gt_mat','gt_haz','gt_aff','gt_reach','gt_vis','z_target']
    return {k:batch[k][index].detach().cpu().numpy() for k in keys}
