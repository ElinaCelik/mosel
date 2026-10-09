import torch
import numpy as np
from sklearn.metrics import roc_curve

from utils import num_parameters, model_from_dataset
from ..base import BaseMethod
from objectives import DEOHyperbolicTangentRelaxation

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_1d(logits):
    if isinstance(logits, torch.Tensor):
        out = logits.squeeze(-1) if logits.ndim == 2 else logits
    else:
        out = np.squeeze(logits, axis=-1) if logits.ndim == 2 else logits
    if out.ndim != 1:
        raise ValueError(
            "Expected 1-D binary logits, got shape {}".format(
                tuple(logits.shape)))
    return out

def _compute_roc_convex_hull(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)

    finite_mask = np.isfinite(thr)
    if not finite_mask.all():
        max_finite = thr[finite_mask].max() if finite_mask.any() else 0.0
        thr = np.where(finite_mask, thr, max_finite + 1.0)

    points = sorted(
        set(zip(fpr.tolist(), tpr.tolist(), thr.tolist())),
        key=lambda x: (x[0], -x[1]),
    )

    hull = []
    for pt in points:
        while len(hull) >= 2:
            o, a, b = hull[-2], hull[-1], pt
            cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(pt)
    return hull


def _interp_randomized_threshold(hull, target_fpr, target_tpr):

    for i in range(len(hull) - 1):
        f0, t0, th0 = hull[i]
        f1, t1, th1 = hull[i + 1]
        df, dt = f1 - f0, t1 - t0
        if abs(df) < 1e-12 and abs(dt) < 1e-12:
            continue
        alpha = ((target_fpr - f0) / df) if abs(df) > 1e-9 else ((target_tpr - t0) / dt)
        if -1e-6 <= alpha <= 1 + 1e-6:
            alpha = float(np.clip(alpha, 0.0, 1.0))
            if not np.isfinite(th0):
                return 0.0, th1, th1
            if not np.isfinite(th1):
                return 1.0, th0, th0
            return alpha, th0, th1

    dists = [(abs(f - target_fpr) + abs(t - target_tpr), th)
             for f, t, th in hull if np.isfinite(th)]
    best_th = min(dists, key=lambda x: x[0])[1]
    return 1.0, best_th, best_th

def _deterministic_threshold(params):
    return params['alpha'] * params['thr_lo'] + (1 - params['alpha']) * params['thr_hi']


def _deterministic_predict(logits_np, sens_np, group_params):
    preds = np.zeros(len(logits_np))
    for g, params in group_params.items():
        m = sens_np == g
        preds[m] = (logits_np[m] >= _deterministic_threshold(params)).astype(float)
    return preds


def _deterministic_accuracy(logits_np, labels_np, sens_np, group_params):
    preds = _deterministic_predict(logits_np, sens_np, group_params)
    return float(np.mean(preds == labels_np))


def _deterministic_eo_gap(logits_np, labels_np, sens_np, group_params):
    preds = _deterministic_predict(logits_np, sens_np, group_params)
    tprs = {}
    for g in group_params:
        m   = sens_np == g
        pos = labels_np[m] == 1
        tprs[g] = float(preds[m][pos].mean()) if pos.sum() > 0 else 0.0
    gs = list(group_params.keys())
    return max(
        abs(tprs[gs[i]] - tprs[gs[j]])
        for i in range(len(gs)) for j in range(i + 1, len(gs))
    )

def _build_pareto_front(logits, labels, sensible_attribute, n_points):

    logits_np = _to_numpy(logits).ravel()
    labels_np = _to_numpy(labels).ravel()
    sens_np   = _to_numpy(sensible_attribute).ravel()

    groups = np.unique(sens_np)

    hulls = {}
    for g in groups:
        m = sens_np == g
        if labels_np[m].sum() == 0 or (labels_np[m] == 0).sum() == 0:
            hulls[g] = None
            continue
        hulls[g] = _compute_roc_convex_hull(logits_np[m], labels_np[m])

    baseline_params = {
        g: {'alpha': 1.0, 'thr_lo': 0.0, 'thr_hi': 0.0} for g in groups
    }
    baseline_acc = _deterministic_accuracy(logits_np, labels_np, sens_np, baseline_params)
    baseline_gap = _deterministic_eo_gap(logits_np, labels_np, sens_np, baseline_params)

    points = [{
        'target_tpr':   None,
        'group_params': baseline_params,
        'fairness_gap': baseline_gap,
        'accuracy':     baseline_acc,
        'is_baseline':  True,
    }]

    candidate_tprs = np.linspace(0.0, 1.0, n_points)

    for target_tpr in candidate_tprs:
        gp = {}
        for g in groups:
            hull = hulls[g]
            if hull is None:
                gp[g] = {'alpha': 1.0, 'thr_lo': 0.0, 'thr_hi': 0.0}
                continue
            hull_tprs = [t for _, t, _ in hull]
            hull_fprs = [f for f, _, _ in hull]
            group_fpr = float(np.interp(target_tpr, hull_tprs, hull_fprs))
            alpha, th_lo, th_hi = _interp_randomized_threshold(
                hull, group_fpr, target_tpr)
            gp[g] = {'alpha': alpha, 'thr_lo': th_lo, 'thr_hi': th_hi}

        acc = _deterministic_accuracy(logits_np, labels_np, sens_np, gp)
        gap = _deterministic_eo_gap(logits_np, labels_np, sens_np, gp)

        points.append({
            'target_tpr':   float(target_tpr),
            'group_params': gp,
            'fairness_gap': gap,
            'accuracy':     acc,
            'is_baseline':  False,
        })

    return points


class FairClassifierMethod(BaseMethod):
    
    def __init__(self, objectives, lmda, lr, n_test_rays, dim, baseline, **kwargs):
        self.objectives = [obj for obj in objectives if not isinstance(obj, DEOHyperbolicTangentRelaxation)]
        self.lamda       = lmda
        self.n_test_rays = n_test_rays
        self.lr          = lr
        self.baseline    = baseline
        self._pareto     = None
        self._full_data  = None

        dim = list(dim)
        self.device = (
            torch.device("cpu") if len(dim) == 1
            else torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.model = model_from_dataset(method='fair_classifier', dim=dim, **kwargs).to(self.device)
        self.n_params = num_parameters(self.model)

    def step(self, batch):
        self.model.zero_grad()
        out = self.model(dict(data=batch['data']))
        batch.update(out)

        for obj in self.objectives:
            loss = obj(**batch)
            loss.backward()
        return loss.item()

    def begin_eval(self, data_loader, device):
        all_logits, all_labels, all_sens = [], [], []

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = self.model(batch)

                logits_key = next(k for k in out if k.startswith('logits'))
                labels_key = next(k for k in batch if k.startswith('labels'))

                all_logits.append(out[logits_key].cpu())
                all_labels.append(batch[labels_key].cpu())
                all_sens.append(batch['sensible_attribute'].cpu())

        self._full_data = {
            'logits': torch.cat(all_logits, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'sens':   torch.cat(all_sens, dim=0),
            'logits_key': logits_key,
            'labels_key': labels_key,
        }

        logits_1d = _ensure_1d(self._full_data['logits'])
        self._pareto = _build_pareto_front(
            logits_1d,
            self._full_data['labels'],
            self._full_data['sens'],
            n_points=self.n_test_rays,
        )

    def eval_step(self, batch, test_rays=None):
        if self.baseline:
            self.model.eval()
            return [self.model(batch)]

        if self._full_data is not None:
            return self._eval_step_full_data(batch)
        return self._eval_step_per_batch(batch)

    def _apply_pareto_thresholds(self, logits, sens_np, logits_key):
        logits_1d = _ensure_1d(logits)
        logits_np = _to_numpy(logits_1d)

        results = []
        for point in self._pareto:
            thr = np.empty(len(logits_np))
            for g, params in point['group_params'].items():
                m = sens_np == g
                thr[m] = _deterministic_threshold(params)
            shifted = torch.tensor(
                logits_np - thr, dtype=logits.dtype, device=logits.device)
            results.append({logits_key: shifted})
        return results

    def _eval_step_full_data(self, batch):
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch)
        logits_key = self._full_data['logits_key']
        sens_np = _to_numpy(batch['sensible_attribute'])
        return self._apply_pareto_thresholds(out[logits_key], sens_np, logits_key)

    def _eval_step_per_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch)

        logits_key = next(
            (k for k in ('logits',) if k in out),
            next(k for k in out if k.startswith('logits')),
        )
        labels_key = next(
            (k for k in ('labels',) if k in batch),
            next(k for k in batch if k.startswith('labels')),
        )

        logits_1d = _ensure_1d(out[logits_key])
        sens_np = _to_numpy(batch['sensible_attribute'])

        self._pareto = _build_pareto_front(
            logits_1d,
            batch[labels_key],
            batch['sensible_attribute'],
            n_points=self.n_test_rays,
        )
        return self._apply_pareto_thresholds(out[logits_key], sens_np, logits_key)

    def end_eval(self):
        self._full_data = None
        self._pareto    = None