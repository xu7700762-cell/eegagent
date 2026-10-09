"""Export real LOSO risk curves and model comparison diagnostics without fitting."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .data import digest, read_csv, write_json
from .experiment import save_csv
from .risk_coverage_v3 import paired_subject_bootstrap, risk_curve, score_metrics

DEFAULT_RELIABILITY = Path('outputs/vrms_reliability/v3_20261008')
DEFAULT_MODELS = Path('outputs/vrms_model_ablation/v3_20261008')


def export(reliability=DEFAULT_RELIABILITY,models=DEFAULT_MODELS):
    reliability,models=Path(reliability),Path(models)
    out=reliability/'analysis'
    if out.exists():
        raise FileExistsError('Preserve the finished analysis; use a new run directory')
    out.mkdir()
    summary=json.loads((reliability/'results/summary.json').read_text(encoding='utf-8'))
    baseline=json.loads((reliability/'baseline_diagnostics/summary.json').read_text(encoding='utf-8'))
    rows=read_csv(reliability/'results/path_oof.csv')
    y=np.asarray([int(r['label']) for r in rows])
    groups=np.asarray([int(r['subject_key']) for r in rows])
    model_rows={int(r['path_index']):r for r in read_csv(models/'model_results/path_oof.csv')}
    series={key:np.asarray([float(r[key]) if r[key] else np.nan for r in rows])
            for key in ('p_raw_v2','p_raw','p_cal')}
    for method in ('cnn_frozen_mil','femba_frozen_mil'):
        series[method]=np.asarray([float(model_rows[int(r['path_index'])][method+'_probability'])
                   if model_rows[int(r['path_index'])][method+'_probability'] else np.nan for r in rows])
    curves={key:risk_curve(y,p,groups) for key,p in series.items()}
    points=[dict(method=key,target_coverage=point['target_coverage'],achieved=point['achieved'],
                  actual_coverage=point['actual_point']['coverage'] if point['achieved'] else None,
                  accepted_paths=point['actual_point']['accepted_paths'] if point['achieved'] else None,
                  accepted_subjects=point['actual_point']['accepted_subjects'] if point['achieved'] else None,
                  risk=point['actual_point']['risk'] if point['achieved'] else None,
                  implied_risk=point['actual_point']['mean_implied_risk'] if point['achieved'] else None,
                  subject_bootstrap_95ci=json.dumps(point['actual_point']['subject_bootstrap_95ci']) if point['achieved'] else None)
            for key,curve in curves.items() for point in curve['target_points']]
    save_csv(out/'risk_at_20_40_60.csv',points)
    save_csv(out/'risk_curve_source.csv',[dict(method=key,**row) for key,curve in curves.items() for row in curve['rows']])
    pairing=paired_subject_bootstrap(y,series['cnn_frozen_mil'],series['femba_frozen_mil'],groups)
    write_json(out/'numerical_analysis.json',dict(status='completed',risk_coverage=curves,
        model_metrics={key:score_metrics(y,p) for key,p in series.items()},paired_encoder_mil=pairing,
        no_fitting=True,no_threshold_selection=True,
        risk_ci='1000 resamples of subjects, fixed observed score threshold, percentile95; exploratory',
        confidence='absolute probability margin; no truth used for ranking'))
    colors=['#D55E00','#0072B2','#009E73','#666666','#CC79A7']
    styles=['-','--','-.',':','-']
    labels=['V2 CNN readout','V3 CNN readout','V3 calibrated CNN','Frozen CNN + MIL','Frozen FEMBA + MIL']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.titlesize':11}):
        fig,axes=plt.subplots(1,3,figsize=(14.0,4.7),layout='constrained')
        counts=baseline['reasons']['uncertain']['primary_upstream_failure_counts']
        names=['calibration_unavailable','policy_unvalidated_with_calibration','ood_exceeded','margin_failed']
        ax=axes[0]
        values=[counts[name] for name in names]
        ax.barh(['Calibration unavailable','Policy unvalidated','OOD exceeded','Margin failed'],values,color='#0072B2')
        ax.invert_yaxis()
        for i,value in enumerate(values):
            ax.text(value+1,i,str(value),va='center')
        ax.set(xlim=(0,max(values)*1.18),xlabel='Paths (exclusive upstream reason)',title='A  Why 131 V2 paths abstained')
        ax=axes[1]
        for (key,curve),color,style,label in zip(curves.items(),colors,styles,labels):
            xs=[r['coverage'] for r in curve['rows']]
            ys=[r['risk'] for r in curve['rows']]
            if xs:
                ax.step(xs,ys,where='post',color=color,ls=style,lw=1.55,label=f'{label} (n={len(series[key][np.isfinite(series[key])])})')
        for value in (.2,.4,.6):
            ax.axvline(value,color='#dddddd',lw=.8,zorder=0)
        ax.axhline(.2,color='black',ls=':',lw=1,label='Declared risk target 0.20')
        ax.plot(14/146,3/14,'o',color='#D55E00',ms=5,label='V2 released subset')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Coverage / all 146 eligible paths',
               ylabel='Observed error among accepted paths',title='B  Risk–Coverage (descriptive)')
        ax.legend(fontsize=7.1,loc='upper right',frameon=False)
        ax=axes[2]
        valid=[f for f in summary['per_fold_calibration'] if f['raw']['auroc'] is not None and f['calibrated']['auroc'] is not None]
        ax.plot([0,1],[0,1],color='#666666',ls='--',lw=1)
        ax.scatter([f['raw']['auroc'] for f in valid],[f['calibrated']['auroc'] for f in valid],
                   marker='o',facecolors='none',edgecolors='#0072B2',s=38)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Within-fold raw AUROC',ylabel='Within-fold calibrated AUROC',
               title=f'C  Monotone calibration ({len(valid)} dual-class folds)')
        ax.text(.04,.95,'Single-class / missing folds omitted\nPooled AUROC is reported separately',transform=ax.transAxes,
                fontsize=8,va='top')
        for extension in ('png','pdf','svg'):
            fig.savefig(out/f'reliability_v3_risk_coverage.{extension}',dpi=220,facecolor='white')
        plt.close(fig)
    write_json(out/'figure_manifest.json',dict(source_sha256={str(path.resolve()):digest(path) for path in
        (reliability/'results/path_oof.csv',reliability/'results/summary.json',reliability/'baseline_diagnostics/summary.json',
         models/'model_results/path_oof.csv')},
        script_sha256=digest(Path(__file__)),figure_width_inches=14.0,figure_height_inches=4.7,dpi=220,
        curve_smoothing=False,ties_kept_together=True,outer_label_threshold_selection=False,
        zero_coverage_risk='undefined; no zero-risk point invented',
        alt_text='V2 has 39 calibration failures,84 unvalidated policies,2 OOD and6 margin failures. Observed LOSO risk is plotted against accepted coverage. Per-fold monotone calibration AUROC stays on the diagonal.',
        output_sha256={p.name:digest(p) for p in out.glob('reliability_v3_risk_coverage.*')}))
    print(json.dumps(dict(figure=str(out/'reliability_v3_risk_coverage.png'),paired=pairing),indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reliability',type=Path,default=DEFAULT_RELIABILITY)
    parser.add_argument('--models',type=Path,default=DEFAULT_MODELS)
    args=parser.parse_args()
    export(args.reliability,args.models)


if __name__=='__main__':
    main()
