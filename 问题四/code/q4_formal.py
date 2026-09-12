"""Current inherited formal entry; stale monthly checkpoints are never resumed."""
import argparse
from q4_revision_run import formal, publish_results, cfg_load as _cfg, state_load as _state, state_dict as _state_dict

def run_branch(branch, resume=True):
    # Recompute into a fingerprinted staging area; legacy checkpoints cannot
    # prove compatibility after Q3/Q4 physics/controller changes.
    return formal(branch)

def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=['q4_2','q4_3','both'],default='both');p.add_argument('--no-resume',action='store_true');a=p.parse_args()
    for b in (('q4_2','q4_3') if a.branch=='both' else (a.branch,)):run_branch(b)
    if a.branch=='both':publish_results()
if __name__=='__main__':main()
