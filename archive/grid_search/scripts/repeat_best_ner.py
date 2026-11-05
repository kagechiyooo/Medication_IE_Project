"""Repeat selected hyperparameters; no test data or new grid search."""
import argparse
from train_ner import load_config, run_with_lock

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    run_with_lock(load_config(args.config),repeat_best=True)

if __name__=='__main__':
    main()
