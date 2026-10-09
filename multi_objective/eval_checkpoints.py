import argparse
import os
import pathlib
import sys
import json
import torch
from torch.utils import data

import utils
from main import parse_args, method_from_name, evaluate
from methods import SingleTaskMethod
from scores import from_objectives
from objectives import from_name


def eval(settings, checkpoint_dir):
    """
    The full evaluation loop. Generate scores for all checkpoints found in checkpoint_dir.

    Uses the same ArgumentParser as main.py to determine the method and dataset, so pass
    the same arguments that were used for training.
    """

    print("start evaluation with settings", settings)
    utils.set_seed(settings['seed'])

    # create the experiment folders
    logdir = os.path.join(settings['logdir'], 'eval', settings['method'], settings['dataset'], utils.get_runname(settings))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    # prepare
    train_set = utils.dataset_from_name(split='train', **settings)
    val_set = utils.dataset_from_name(split='val', **settings)
    test_set = utils.dataset_from_name(split='test', **settings)

    train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=True, num_workers=settings['num_workers'])
    val_loader = data.DataLoader(val_set, settings['batch_size'], shuffle=True, num_workers=settings['num_workers'])
    test_loader = data.DataLoader(test_set, settings['batch_size'], num_workers=settings['num_workers'])

    objectives = from_name(settings.pop('objectives'), train_set.task_names(), settings)
    scores = from_objectives(objectives, settings)

    method = method_from_name(objectives=objectives, **settings)
    device = next(method.model.parameters()).device

    train_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    val_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    test_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))

    with open(pathlib.Path(logdir) / "settings.json", "w") as file:
        json.dump(train_results, file)

    # checkpoints are saved by main.py as c_{start}-{epoch:03d}.pth
    checkpoints = sorted(pathlib.Path(checkpoint_dir).glob('**/c_*.pth'))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    for c in checkpoints:
        print("checkpoint", c)
        j, e = (int(x) for x in c.stem.replace('c_', '').split('-'))

        for results in (train_results, val_results, test_results):
            results.setdefault(f"start_{j}", {})

        if isinstance(method, SingleTaskMethod) and 'task_id' not in settings:
            # trained sequentially: start j corresponds to task j
            method.task = j

        method.model.load_state_dict(torch.load(c, map_location=device))

        for split, loader, results in (
                ('val', val_loader, val_results),
                ('test', test_loader, test_results),
                ('train', train_loader, train_results)):
            evaluate(j, e, method, scores, loader, logdir,
                reference_point=settings['reference_point'],
                split=split,
                result_dict=results,
                text=settings['text'])


if __name__ == "__main__":
    # --checkpoint_dir is specific to this script, everything else is parsed like in main.py
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint_dir', '-c', required=True, help="Directory containing the c_*.pth checkpoints (searched recursively).")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    settings = parse_args()
    eval(settings, args.checkpoint_dir)
