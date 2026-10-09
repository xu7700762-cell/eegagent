# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal, linalg
from scipy.special import logit
from sklearn.base import BaseEstimator, TransformerMixin
FS=256
BANDS=((.5,4),(4,8),(8,13),(13,30))
REGIONS=((0,1),tuple(range(2,9)),tuple(range(9,14)),tuple(range(14,19)),tuple(range(19,27)),(27,28,29))

def spectral_features(x):
    f, psd = signal.welch(x, fs=FS, nperseg=512, noverlap=256, axis=1)
    power = np.stack([np.trapezoid(psd[:, (f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=1)
                      for lo, hi in BANDS], axis=1)
    power = np.maximum(power, 1e-10)
    relative = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-10)
    return np.concatenate((np.log(power).ravel(), relative.ravel())).astype(np.float32), power


def covariance(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    c = x @ x.T / (x.shape[1] - 1)
    return 0.9 * c + 0.1 * max(float(np.trace(c)) / len(c), 1e-10) * np.eye(len(c))


def predict_score(model, values):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(values), float)
    return logit(np.clip(model.predict_proba(values)[:, 1], 1e-6, 1 - 1e-6))


def compact_spectrum(absolute, reference=None):
    a = np.asarray(absolute, float)
    logpower, relative = a[:, :120].reshape(-1,30,4), a[:, 120:].reshape(-1,30,4)
    def regional(values):
        return np.stack([values[:, group, :].mean(axis=1) for group in REGIONS], axis=1)
    logs, rel = regional(logpower), regional(relative)
    n = max(1,len(a)//3)
    changes = np.zeros(24) if reference is None else regional(np.asarray(reference).reshape(-1,30,4)).mean(axis=0).ravel()
    # Ratios reduce overall amplitude differences across subjects, while
    # preserving initial-reference change as a separately learned feature.
    centered_logs = logs-logs.mean(axis=2,keepdims=True)
    return np.r_[rel.mean(axis=0).ravel(), rel.std(axis=0).ravel(), centered_logs.mean(axis=0).ravel(),
                 centered_logs[-n:].mean(axis=0).ravel()-centered_logs[:n].mean(axis=0).ravel(),
                 changes, float(reference is not None)]


def filterbank_covariance(windows):
    result = []
    for low,high in BANDS:
        sos = signal.butter(3, [low,high], btype="bandpass",fs=256,output="sos")
        values = signal.sosfilt(sos, np.asarray(windows,float), axis=-1)
        values -= values.mean(axis=-1,keepdims=True)
        cov = np.einsum("wct,wdt->wcd",values,values)/(values.shape[-1]-1)
        traces = np.trace(cov,axis1=1,axis2=2)
        cov = cov / np.maximum(traces[:,None,None],1e-10)
        result.append(cov.mean(axis=0))
    return np.asarray(result).ravel()


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Supervised spatial filters refitted inside every subject-heldout head."""
    def __init__(self, components=4, shrinkage=.2):
        self.components, self.shrinkage = components, shrinkage

    def fit(self, x, y):
        values=np.asarray(x).reshape(-1,4,30,30)
        if set(np.unique(y)) != {0,1}:
            raise ValueError("CSP requires both training classes")
        self.filters_=[]
        for band in range(4):
            matrices=[]
            for label in (0,1):
                c=values[np.asarray(y)==label,band].mean(axis=0)
                c=(1-self.shrinkage)*c+self.shrinkage*np.trace(c)/30*np.eye(30)
                matrices.append(c)
            _,vectors=linalg.eigh(matrices[1],matrices[0]+matrices[1])
            n=self.components//2
            self.filters_.append(vectors[:,list(range(n))+list(range(30-n,30))].T)
        self.filters_=np.asarray(self.filters_)
        return self

    def transform(self, x):
        values=np.asarray(x).reshape(-1,4,30,30)
        variances=np.einsum("bkc,nbcd,bkd->nbk",self.filters_,values,self.filters_)
        variances=np.maximum(variances,1e-10)
        return np.log(variances/variances.sum(axis=2,keepdims=True)).reshape(len(values),-1)
