import torch
import random
import numpy as np
import copy
import yaml

# seed now to be save and overwrite later
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.mps.manual_seed(1)


import argparse
import os
import pathlib
import time
import tracemalloc
import json
from torch.utils import data

import settings as s
import utils
from objectives import from_name
from hv import HyperVolume


from methods import HypernetMethod, ParetoMTLMethod, SingleTaskMethod, COSMOSMethod, MGDAMethod, UniformScalingMethod, MOSELMethod, FairClassifierMethod, PCGradMethod, GradNormMethod, NashMTLMethod, FAMOMethod
from scores import from_objectives, CrossEntropy, BinaryCrossEntropy


def method_from_name(method, **kwargs):
    if method == 'ParetoMTL':
        return ParetoMTLMethod(**kwargs)
    elif method == 'mosel':
        return MOSELMethod(**kwargs)
    elif 'cosmos' in method:
        return COSMOSMethod(**kwargs)
    elif method == 'SingleTask':
        return SingleTaskMethod(**kwargs)
    elif 'hyper' in method:
        return HypernetMethod(**kwargs)
    elif method == 'mgda':
        return MGDAMethod(**kwargs)
    elif method == 'uniform':
        return UniformScalingMethod(**kwargs)
    elif method == 'fair_classifier':
        return FairClassifierMethod(**kwargs)
    elif method == 'pcgrad':
        return PCGradMethod(**kwargs)
    elif method == 'nashmtl':
        return NashMTLMethod(**kwargs)
    elif method == 'gradnorm':
        return GradNormMethod(**kwargs)
    elif method == 'famo':
        return FAMOMethod(**kwargs)
    else:
        raise ValueError("Unknown method {}".format(method))


epoch_max = -1
volume_max = -1
elapsed_time = 0
peak_cpu_memory = 0  
peak_gpu_memory = 0  


def evaluate(j, e, method, scores, data_loader, logdir, reference_point, split, result_dict, text):
    assert split in ['train', 'val', 'test']
    global volume_max
    global epoch_max

    if next(iter(data_loader))['data'][0].dim() == 1 and not text:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    score_values = np.array([])

    mcr_indices = [i for i, s in enumerate(scores) if isinstance(s, (CrossEntropy, BinaryCrossEntropy))]
    mcr_mode = len(mcr_indices) > 0

    is_fair_classifier = isinstance(method, FairClassifierMethod)
    if is_fair_classifier:
        method.begin_eval(data_loader, device)

    for batch in data_loader:
        batch = utils.dict_to_device(batch, device)

        s = []  
        m = [] 

        for l in method.eval_step(batch):
            batch.update(l)
            raw_results = [s(**batch) for s in scores]
            loss_values = []
            mcr_values_step = []
            for i, res in enumerate(raw_results):
                if i in mcr_indices:
                    loss, mcr = res  
                    loss_values.append(loss)
                    mcr_values_step.append(mcr)
                else:
                    loss_values.append(res)  

            s.append(loss_values)
            if mcr_mode:
                m.append(mcr_values_step)

        if score_values.size == 0:
            score_values = np.array(s)
            if mcr_mode:
                mcr_values = np.array(m)
        else:
            score_values += np.array(s)
            if mcr_mode:
                mcr_values += np.array(m)

    score_values /= len(data_loader)

    if is_fair_classifier:
        method.end_eval()

    if mcr_mode:
        mcr_values /= len(data_loader)

    def filter_dominated(score_values):
        points = score_values
        dominated = utils.is_dominated(points)
        if dominated.any():
            non_dom_points = points[~dominated]
            for i in np.where(dominated)[0]:
                nearest = np.argmin(np.linalg.norm(non_dom_points - points[i], axis=1))
                points[i] = non_dom_points[nearest]
        return points, dominated

    score_values, dominated = filter_dominated(score_values)

    hv = HyperVolume(reference_point)
    volume = hv.compute(score_values) if score_values.shape[1] < 3 else -1

    if mcr_mode:
        full_mcr_values = score_values.copy()  
        for col, i in enumerate(mcr_indices):
            full_mcr_values[:, i] = mcr_values[:, col]

        volume_mcr = hv.compute(full_mcr_values) if full_mcr_values.shape[1] < 3 else -1

        mcr_labels = [
            f"MCR" if i in mcr_indices else scores[i].__class__.__name__
            for i in range(len(scores))
        ]
        pareto_front_mcr = utils.ParetoFront(mcr_labels, logdir, "{}_{:03d}".format('mcr', e))
        pareto_front_mcr.append(full_mcr_values)
        pareto_front_mcr.plot()

        result = {
            "scores": score_values.tolist(),
            "hv": volume,
            "scores_mcr": full_mcr_values.tolist(),
            "mcr": volume_mcr,
        }
    else:
        result = {
            "scores": score_values.tolist(),
            "hv": volume,
        }

    if len(scores) == 2:
        pareto_front = utils.ParetoFront([s.__class__.__name__ for s in scores], logdir, "{}_{:03d}".format(split, e))
        pareto_front.append(score_values)
        pareto_front.plot()

    if split == 'val':
        if volume > volume_max:
            volume_max = volume
            epoch_max = e
        result.update({
            "max_epoch_so_far": epoch_max,
            "max_volume_so_far": volume_max,
            "training_time_so_far": elapsed_time,
            "peak_cpu_memory_mb_so_far": peak_cpu_memory,
            "peak_gpu_memory_mb_so_far": peak_gpu_memory,
        })
    elif split == 'test':
        result.update({
            "training_time_so_far": elapsed_time,
            "peak_cpu_memory_mb_so_far": peak_cpu_memory,
            "peak_gpu_memory_mb_so_far": peak_gpu_memory,
        })

    result.update(method.log())

    if f"epoch_{e}" in result_dict[f"start_{j}"]:
        result_dict[f"start_{j}"][f"epoch_{e}"].update(result)
    else:
        result_dict[f"start_{j}"][f"epoch_{e}"] = result

    with open(pathlib.Path(logdir) / f"{split}_results.json", "w") as file:
        json.dump(result_dict, file)
    return result_dict


def main(settings):
    print("start processig with settings", settings)
    utils.set_seed(settings['seed'])

    global elapsed_time
    global peak_cpu_memory
    global peak_gpu_memory

    # create the experiment folders
    logdir = os.path.join(settings['logdir'], settings['method'], settings['dataset'], utils.get_runname(settings))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)


    # prepare
    train_set = utils.dataset_from_name(split='train', **settings)
    val_set = utils.dataset_from_name(split='val', **settings)
    test_set = utils.dataset_from_name(split='test', **settings)

    train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=True, num_workers=settings['num_workers'])
    val_loader = data.DataLoader(val_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    test_loader = data.DataLoader(test_set, settings['batch_size'], settings['num_workers'])

    objectives = from_name(settings.pop('objectives'), train_set.task_names(), settings)
    scores = from_objectives(objectives, settings)

    rm1 = utils.RunningMean(400)
    rm2 = utils.RunningMean(400)

    method = method_from_name(objectives=objectives, **settings)

    train_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    val_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    test_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))

    if next(iter(train_loader))['data'][0].dim() == 1 and not settings['text']:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    with open(pathlib.Path(logdir) / "settings.json", "w") as file:
        json.dump(train_results, file)

    tracemalloc.start()

    # main
    for j in range(settings['num_starts']):
        train_results[f"start_{j}"] = {}
        val_results[f"start_{j}"] = {}
        test_results[f"start_{j}"] = {}

        # for MOSEL we need two optimizer
        if settings['method'] == 'mosel':

            # leader: preference alignment network
            if settings['dataset'] in ['multi_mnist', 'multi_fashion_mnist', 'multi_fashion']:
                optimizer_leader = torch.optim.Adam(method.model.params_preference_alignment(), settings['lr'][0], weight_decay=1e-3)
            else:
                optimizer_leader = torch.optim.Adam(method.model.params_preference_alignment(), settings['lr'][0])
            # follower: representation learning network
            optimizer_follower = torch.optim.Adam(method.model.params_representation(), settings['lr'][1])

            if settings['use_scheduler']:
                scheduler_leader = torch.optim.lr_scheduler.MultiStepLR(optimizer_leader, settings['scheduler_milestones'], gamma=settings['scheduler_gamma'])
                scheduler_follower = torch.optim.lr_scheduler.MultiStepLR(optimizer_follower, settings['scheduler_milestones'], gamma=settings['scheduler_gamma'])

            for e in range(settings['epochs']):
                print(f"Epoch {e}")
                tick = time.time()
                method.new_epoch(e)

                for b, batch in enumerate(train_loader):
                    batch = utils.dict_to_device(batch, device)
                    optimizer_leader.zero_grad()
                    optimizer_follower.zero_grad()
                    
                    stats = method.step(batch) 
                    
                    optimizer_leader.step()
                    optimizer_follower.step()
                    
                    loss, sim  = stats if isinstance(stats, tuple) else (stats, 0)   
                    print("Epoch {:03d}, batch {:03d}, train_loss {:.4f}, sim {:.4f}, rm train_loss {:.3f}, rm sim {:.3f}".format(e, b, loss, sim, rm1(loss), rm2(sim)))
                
                tock = time.time()
                elapsed_time += (tock - tick)

                current_cpu, peak_cpu = tracemalloc.get_traced_memory()
                peak_cpu_memory = max(peak_cpu_memory, peak_cpu / 1e6)
                tracemalloc.reset_peak()

                if torch.cuda.is_available():
                    peak_gpu = torch.cuda.max_memory_allocated() / 1e6
                    peak_gpu_memory = max(peak_gpu_memory, peak_gpu)
                    torch.cuda.reset_peak_memory_stats()

                print(f"Epoch {e} | Time: {tock - tick:.2f}s | "
                      f"CPU mem current: {current_cpu / 1e6:.1f} MB, peak: {peak_cpu_memory:.1f} MB | "
                      f"GPU mem peak: {peak_gpu_memory:.1f} MB")

                if settings['use_scheduler']:
                    val_results[f"start_{j}"][f"epoch_{e}"] = {'lr': scheduler_leader.get_last_lr()[0]}
                    val_results[f"start_{j}"][f"epoch_{e}"] = {'lr': scheduler_follower.get_last_lr()[0]}
                    scheduler_leader.step()
                    scheduler_follower.step()

                # run eval on train set (mainly for debugging)
                if settings['train_eval_every'] > 0 and (e+1) % settings['train_eval_every'] == 0:
                    train_results = evaluate(j, e, method, scores, train_loader, logdir, 
                        reference_point=settings['reference_point'],
                        split='train',
                        result_dict=train_results,
                        text=settings['text'])

                
                if settings['eval_every'] > 0 and (e+1) % settings['eval_every'] == 0:
                    # Validation results
                    val_results = evaluate(j, e, method, scores, val_loader, logdir, 
                        reference_point=settings['reference_point'],
                        split='val',
                        result_dict=val_results,
                        text=settings['text'])

                    # Test results
                    test_results = evaluate(j, e, method, scores, test_loader, logdir, 
                        reference_point=settings['reference_point'],
                        split='test',
                        result_dict=test_results,
                        text=settings['text'])

                # Checkpoints
                if settings['checkpoint_every'] > 0 and (e+1) % settings['checkpoint_every'] == 0:
                    pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
                    torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, e)))

        else: 
            optimizer = torch.optim.Adam(method.model_params(), settings['lr'])
            if settings['use_scheduler']:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, settings['scheduler_milestones'], gamma=settings['scheduler_gamma'])
            
            for e in range(settings['epochs']):
                print(f"Epoch {e}")
                tick = time.time()
                method.new_epoch(e)

                _ext_opt = not getattr(method, 'uses_internal_optimizer', False)
                for b, batch in enumerate(train_loader):
                    batch = utils.dict_to_device(batch, device)
                    if _ext_opt:
                        optimizer.zero_grad()
                    stats = method.step(batch)
                    if _ext_opt:
                        optimizer.step()

                    loss, sim  = stats if isinstance(stats, tuple) else (stats, 0)
                    print("Epoch {:03d}, batch {:03d}, train_loss {:.4f}, sim {:.4f}, rm train_loss {:.3f}, rm sim {:.3f}".format(e, b, loss, sim, rm1(loss), rm2(sim)))
                
                tock = time.time()
                elapsed_time += (tock - tick)

                current_cpu, peak_cpu = tracemalloc.get_traced_memory()
                peak_cpu_memory = max(peak_cpu_memory, peak_cpu / 1e6)
                tracemalloc.reset_peak()

                if torch.cuda.is_available():
                    peak_gpu = torch.cuda.max_memory_allocated() / 1e6
                    peak_gpu_memory = max(peak_gpu_memory, peak_gpu)
                    torch.cuda.reset_peak_memory_stats()

                print(f"Epoch {e} | Time: {tock - tick:.2f}s | "
                      f"CPU mem current: {current_cpu / 1e6:.1f} MB, peak: {peak_cpu_memory:.1f} MB | "
                      f"GPU mem peak: {peak_gpu_memory:.1f} MB")

                if settings['use_scheduler']:
                    val_results[f"start_{j}"][f"epoch_{e}"] = {'lr': scheduler.get_last_lr()[0]}
                    scheduler.step()

                # run eval on train set (mainly for debugging)
                if settings['train_eval_every'] > 0 and (e+1) % settings['train_eval_every'] == 0:
                    train_results = evaluate(j, e, method, scores, train_loader, logdir, 
                        reference_point=settings['reference_point'],
                        split='train',
                        result_dict=train_results,
                        text=settings['text'])

                
                if settings['eval_every'] > 0 and (e+1) % settings['eval_every'] == 0:
                    # Validation results
                    val_results = evaluate(j, e, method, scores, val_loader, logdir, 
                        reference_point=settings['reference_point'],
                        split='val',
                        result_dict=val_results,
                        text=settings['text'])

                    # Test results
                    test_results = evaluate(j, e, method, scores, test_loader, logdir, 
                        reference_point=settings['reference_point'],
                        split='test',
                        result_dict=test_results,
                        text=settings['text'])

                # Checkpoints
                if settings['checkpoint_every'] > 0 and (e+1) % settings['checkpoint_every'] == 0:
                    pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
                    torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, e)))

        print("epoch_max={}, val_volume_max={}, time={}, peak_cpu_memory_mb={:.1f}, peak_gpu_memory_mb={:.1f}".format(
            epoch_max, volume_max, elapsed_time, peak_cpu_memory, peak_gpu_memory))
        pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, 999999)))

    tracemalloc.stop()
    return volume_max


METHOD_SETTINGS = {
    'SingleTask': s.SingleTaskSolver,
    'cosmos': s.cosmos,
    'mosel': s.mosel,
    'hyper_ln': s.hyperSolver_ln,
    'hyper_epo': s.hyperSolver_epo,
    'pmtl': s.paretoMTL,
    'mgda': s.mgda,
    'uniform': s.uniform_scaling,
    'fair_classifier': s.fair_classifier,
    'pcgrad': s.pcgrad,
    'gradnorm': s.gradnorm,
    'nashmtl': s.nashmtl,
    'famo': s.famo,
}

DATASET_SETTINGS = {
    'mm': s.multi_mnist,
    'mf': s.multi_fashion,
    'mfm': s.multi_fashion_mnist,
    'adult': s.adult,
    'credit': s.credit,
    'compass': s.compass,
    'celeba': s.celeba,
    'fairface': s.fairface,
    'civil': s.civil_comments,
}


def parse_value(value):
    """Parse a command-line value into int, float, bool, None, str or a list (comma-separated)."""
    value = value.strip()
    if value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
    if ',' in value:
        return [parse_value(v) for v in value.split(',') if v.strip()]
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    if value.lower() in ('none', 'null'):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_args():
    """
    Build the settings. Later sources override earlier ones:
    generic < method < dataset < config file (--config) < command-line options.

    Every key in the resulting settings can be overridden with --<key> <value>,
    lists are passed comma-separated, e.g. --lr 0.0001,0.001
    """
    # first pass: dataset, method and config determine the available settings and their defaults
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument('--dataset', '-d', default='mm', choices=DATASET_SETTINGS.keys(), help="The dataset to run on.")
    base.add_argument('--method', '-m', default='cosmos', choices=METHOD_SETTINGS.keys(), help="The method to generate the Pareto front.")
    base.add_argument('--config', '-f', type=str, help='Path to YAML config file. Overrides the defaults in settings.py.')
    args, _ = base.parse_known_args()

    settings = copy.deepcopy(s.generic)
    settings.update(copy.deepcopy(METHOD_SETTINGS[args.method]))
    settings.update(copy.deepcopy(DATASET_SETTINGS[args.dataset]))

    # config overrides settings
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
        # dataset and method are selected with --dataset and --method
        config.pop('dataset', None)
        config.pop('method', None)
        if 'split' in config:
            config['split_network'] = config.pop('split')
        settings.update(config)

    # second pass: command-line options override config and settings
    parser = argparse.ArgumentParser(parents=[base], description=parse_args.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', '-s', type=int, default=argparse.SUPPRESS, help=f"Seed (default: {settings['seed']})")
    parser.add_argument('--task_id', '-t', type=int, default=argparse.SUPPRESS, help='Task id to run single task in parallel. If not set then sequentially.')
    parser.add_argument('--lamda', '-l', type=float, default=argparse.SUPPRESS, help=f"Lambda for COSMOS and ParetoMTL (default: {settings.get('lamda')})")
    parser.add_argument('--alpha', '-a', type=parse_value, default=argparse.SUPPRESS, help=f"Alpha (default: {settings.get('alpha')})")
    parser.add_argument('--split_network', '--split', type=int, default=argparse.SUPPRESS, help=f"Split network (default: {settings.get('split_network')})")
    parser.add_argument('--lr', '--learning_rate', type=parse_value, default=argparse.SUPPRESS, help=f"Learning rate(s) (default: {settings['lr']})")

    defined = {a.dest for a in parser._actions}
    for key in sorted(settings):
        if key not in defined:
            parser.add_argument(f'--{key}', type=parse_value, default=argparse.SUPPRESS, help=f"(default: {settings[key]})")

    overrides = vars(parser.parse_args())
    for key in ('dataset', 'method', 'config'):
        overrides.pop(key)

    for key, value in overrides.items():
        # keep tuples (e.g. dim) as tuples
        if isinstance(settings.get(key), tuple):
            value = tuple(value) if isinstance(value, list) else (value,)
        settings[key] = value

    if 'task_id' in settings:
        settings['task_id'] = int(settings['task_id'])
        if args.method == 'SingleTask':
            settings['num_starts'] = 1

    if 'lr' in overrides:
        lr = overrides['lr']
        settings['_grid_suffix'] = '_lr' + (','.join(str(x) for x in lr) if isinstance(lr, list) else str(lr))

    return settings


if __name__ == "__main__":
    
    settings = parse_args()
    main(settings)