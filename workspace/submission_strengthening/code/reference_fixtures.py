"""Authored analytical/strict-selector fixtures; no random data, models or draws."""
import argparse,json
from pathlib import Path
import numpy as np
from reference_policies import load_selector,score_one,digest,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--selector',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();choose=load_selector(a.selector)
    d=dict(gt_aff=np.array([[.8,0],[0,0]],np.float64),gt_reach=np.array([[.05,1],[np.nan,0.04]]),
           gt_haz=np.array([[0,1],[0,0]]),gt_sem=np.zeros((2,2),np.int64))
    pri=dict(mean_A_train=np.array([[.4,.9],[1,1]]),mean_h_train=np.array([[.5,.6],[0,0]]))
    r0=score_one(d,'R0',choose,pri);r1=score_one(d,'R1',choose,pri)
    assert r0['selected_utility']==.4 and r0['regret']==.4 and r0['unsafe']==.5
    assert r0['unsafe_event']=='' and (r0['expected_unsafe_numerator'],r0['expected_unsafe_denominator'])==(1,2)
    assert (r1['pick_row'],r1['pick_col'])==(0,0) and r1['regret']==0 and r1['unsafe_event']==0
    pick,ties,n=choose(np.ones((2,2)),np.ones((2,2)),np.full((2,2),.5))
    assert pick==(0,0) and ties==4 and n==4
    # Entirely excluded candidates abstain; reachable oracle still contributes regret.
    empty=score_one(d,'R1',choose,dict(mean_A_train=np.ones((2,2)),mean_h_train=np.ones((2,2))))
    assert empty['executed']==0 and empty['regret']==.8 and empty['unsafe_event']==0
    d['gt_reach']=np.zeros((2,2))
    for ref in ['R0','R1']:
        x=score_one(d,ref,choose,pri);assert x['abstained']==1 and x['regret']==0 and x['unsafe']==0
    pick,_,n=choose(np.array([[np.nan,1],[2,3]]),np.ones((2,2)),np.array([[0,np.inf],[0,.9]]))
    assert pick==(1,0) and n==1
    # Nonterrain status alone counts as unsafe even without a hazard-buffer label.
    d=dict(gt_aff=np.array([[0.,0.]]),gt_reach=np.ones((1,2)),gt_haz=np.zeros((1,2)),gt_sem=np.array([[0,3]]))
    x=score_one(d,'R0',choose,{});assert x['unsafe']==.5
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.output,dict(status='PASS',fixtures=['analytic uniform expected utility/regret/unsafe','no invented R0 events',
        'reach>=0.05 and hazard<=0.5 exact boundaries','row-major exact ties','all-filtered abstention retains reachable oracle loss',
        'empty reach abstains zero utility','nonfinite candidate exclusion','nonterrain unsafe independent of hazard buffer'],
        selector_sha256=digest(a.selector),new_neural_forwards=0,new_random_draws=0))
    print(a.output)

if __name__=='__main__':main()
