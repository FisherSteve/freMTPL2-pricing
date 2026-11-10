"""Build the Beamer talk from existing aggregates, without fitting or evaluating models.

python scripts/build_slides.py --compile
python scripts/build_slides.py --check
Only Python's standard library, Git and pdfLaTeX are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides"
REF = "a44c7e91c481fab9312af66bdc08cb96ffd1ab85"
INPUTS = {}
VALUES = {}
TABLES = {}


def read(path):
    pinned = path.startswith("reports/s3/")
    content = (subprocess.run(["git", "show", f"{REF}:{path}"], cwd=ROOT,
                             capture_output=True, check=True).stdout
               if pinned else (ROOT / path).read_bytes())
    INPUTS[path] = {"sha256": hashlib.sha256(content).hexdigest(),
                    "origin": f"git:{REF}" if pinned else "checkout"}
    return json.loads(content.decode("utf-8"))


def num(x, digits=3):
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", "{,}").replace("X", r"\,")


def macro(name, x, origin, digits=3):
    VALUES[name] = {"value": x, "origin": origin, "digits": digits, "latex": num(x, digits)}


def pick(rows, key, value):
    return next(r for r in rows if r[key] == value)


def table(name, headers, rows, origin, fmt):
    TABLES[name] = {"origin": origin, "rows": rows}
    return ("\\newcommand{\\" + name + "}{%\n\\begin{tabular}{" + fmt + "}\\toprule\n"
            + " & ".join(headers) + r"\\\midrule" + "\n"
            + "\n".join(" & ".join(row) + r"\\" for row in rows)
            + "\n" + r"\bottomrule\end{tabular}" + "\n}\n")


def generate():
    INPUTS.clear(); VALUES.clear(); TABLES.clear()
    core = read("reports/s6/results.json")
    original_freq = read("reports/s4/scores.json")
    eval_loss = read("reports/s5/evaluation_loss_scores.json")
    dev_loss = read("reports/s5/development_loss_scores.json")
    for r in core["development_frequency"]:
        assert abs(r["poisson_deviance_per_exposure"] - pick(original_freq,"model_id",r["model_id"])["poisson_deviance_per_exposure"]) < 1e-12
    for r in core["loss_comparison"]:
        for rows, field in [(eval_loss,"evaluation_deviance"),(dev_loss,"development_deviance")]:
            assert abs(r[field] - pick(rows,"model_id",r["model_id"])["tweedie_deviance_per_exposure"]) < 1e-10
    gain = 1-core["loss_comparison"][1]["evaluation_deviance"]/core["loss_comparison"][0]["evaluation_deviance"]
    assert abs(gain-core["relative_loss_deviance_reduction"])<1e-12
    macro("EvalGain",100*gain,"reports/s5/evaluation_loss_scores.json:1-D_selected/D_intercept",2)
    for name,field,d in [("EvalObserved","observed_annual_loss_cost",2),("EvalPredicted","predicted_annual_loss_cost",2),("EvalAP","actual_over_predicted",3),("EvalPolicies","policies",0)]:
        macro(name,core["selected_evaluation"][field],f"reports/s6/results.json:selected_evaluation.{field}",d)
    read("reports/s0/source_comparison.json")
    reduced = read("reports/s6a/scores.json")
    abl = read("reports/s6a/ablation/scores.json")
    full = pick(reduced,"model_id","pp_hgb_mean")["tweedie_deviance_per_exposure"]
    best = pick(reduced,"model_id","two_BonusMalus_gam")["tweedie_deviance_per_exposure"]
    macro("ReducedGap",100*(best/full-1),"reports/s6a/scores.json:100*(two_BonusMalus_gam/pp_hgb_mean-1)",4)
    dep = read("reports/s6a/ablation/dependence.json")
    macro("AgeBMRho",dep["exposure_weighted_pearson"],"reports/s6a/ablation/dependence.json:exposure_weighted_pearson")
    macro("AgeBMRank",dep["spearman"],"reports/s6a/ablation/dependence.json:spearman")
    for name,field in [("AreaAgeEta","Area_DrivAge_weighted_eta_squared"),("AreaBMEta","Area_BonusMalus_weighted_eta_squared")]:
        macro(name,dep[field],"reports/s6a/ablation/dependence.json:"+field,5)
    tail = read("reports/s3/tail_concentration.json")["rows"][0]
    macro("LargestShare",100*tail["loss_share"],"reports/s3/tail_concentration.json:rows[0].loss_share",2)
    macro("LargestClaim",tail["loss_eur"],"reports/s3/tail_concentration.json:rows[0].loss_eur",2)
    age = read("reports/s3/age_bootstrap.json")["rows"]
    bm = read("reports/s3/bm_standardized.json")["rows"]
    profiles = read("reports/s6a/existing_age_profiles.json")
    geo = read("reports/s6a/geo.json")
    std = read("reports/s6a/standardized_with_duration.json")
    std0 = read("reports/s6a/standardized_without_duration.json")
    aug = read("reports/o2/report.json")
    assert aug["exploratory"] and not aug["evaluation_used"]
    real = pick(aug["scores"],"method","real")
    ct = [r for r in aug["scores"] if r["method"]=="ctgan"]
    changes=[100*r["vs_real"]["difference"]/real["poisson_deviance_per_exposure"] for r in ct]
    macro("AugMin",min(changes),"reports/o2/report.json:100*vs_real.difference/real.deviance, min",2)
    macro("AugMax",max(changes),"reports/o2/report.json:100*vs_real.difference/real.deviance, max",2)
    macro("AugBaseline",real["poisson_deviance_per_exposure"],"reports/o2/report.json:scores.real",6)
    support=read("reports/o2/joint_support.json")
    outside=[100*r["outside_training_Area_density_fraction"] for r in support["rows"]]
    macro("SupportMin",min(outside),"reports/o2/joint_support.json:outside_training_Area_density_fraction, min",2)
    macro("SupportMax",max(outside),"reports/o2/joint_support.json:outside_training_Area_density_fraction, max",2)
    # These inputs also substantiate textual scope/assumptions, not just tables.
    extra=["reports/s6a/report.json","reports/s6a/ablation/report.json","reports/s6a/release.json",
           "reports/s6a/bootstrap_best.json","reports/s6a/ablation/bootstrap_two_BonusMalus_gam.json",
           "reports/s6a/existing_age_contrasts.json","reports/s6a/support_with_duration.json",
           "reports/s6a/support_without_duration.json","reports/s6a/standardized_intervals_with_duration.json",
           "reports/s6a/standardized_intervals_without_duration.json","reports/s6a/costs.json",
           "reports/s6a/sources.json","reports/o2/sources.json","reports/s3/eda_report.json",
           "reports/s3/age_contrasts.json","reports/s3/bm_common_cells.json","reports/s3/sensitivities.json",
           "reports/s4/paired_bootstrap.json","reports/s4/segments.json","reports/s5/evaluation_loss_segments.json",
           "reports/s5/evaluation_bootstrap.json","config/frequency_protocol.json",
           "config/interpretation_protocol.json","config/feature_ablation_protocol.json",
           "config/severity_protocol.json","config/augmentation_protocol.json","config/validation_protocol.json",
           "paper/evidence.json","paper/sources.json","slides/sources.json"]
    data={p:read(p) for p in extra}
    blocks=[]
    blocks.append(table("FrequencyTable",["Regel",r"$D_N$",r"Gewinn\,\%"],
        [[label,num(r["poisson_deviance_per_exposure"],6),num(100*r["relative_deviance_reduction_vs_intercept"],2)]
         for label,r in zip(["Intercept","GLM","GAM","HGB"],core["development_frequency"])],
        "reports/s6/results.json:development_frequency; checked against reports/s4/scores.json","lrr"))
    blocks.append(table("LossTable",["Schadenhöhe",r"OOF $D_L$",r"Eval. $D_L$"],
        [[label,num(r["development_deviance"],3),num(r["evaluation_deviance"],3)] for label,r in
         zip(["Mittel (gewählt)","Gamma-GLM","Log-Smearing"],core["loss_comparison"][1:])],
        "reports/s6/results.json:loss_comparison; independently checked S5 dev/eval scores","lrr"))
    feature_rows=[]
    for label,rows,model in [("Alle Merkmale (HGB)",reduced,"pp_hgb_mean"),("Alter + BM (GAM)",reduced,"two_BonusMalus_gam"),("BM allein (GAM)",abl,"abl_BonusMalus_gam"),("BM + Area (GAM)",abl,"abl_BonusMalus_Area_gam")]:
        feature_rows.append([label,num(pick(rows,"model_id",model)["tweedie_deviance_per_exposure"],3)])
    blocks.append(table("FeatureTable",["Regel",r"OOF $D_L$"],feature_rows,"reports/s6a/{scores.json,ablation/scores.json}","lr"))
    all_rows=[]
    for features,label in [("DrivAge","Alter"),("BonusMalus","BM"),("Area","Area"),("BonusMalus_Area","BM + Area")]:
        all_rows.append([label,*[num(pick(abl,"model_id",f"abl_{features}_{family}")["tweedie_deviance_per_exposure"],3) for family in ("glm","gam")]])
    for feature,label in [("Area","Alter + Area"),("Density",r"Alter + log(Density)"),("Region","Alter + Region"),("BonusMalus","Alter + BM")]:
        all_rows.append([label,*[num(pick(reduced,"model_id",f"two_{feature}_{family}")["tweedie_deviance_per_exposure"],3) for family in ("glm","gam")]])
    blocks.append(table("AllFeatureTable",["Merkmale",r"GLM $D_L$",r"GAM $D_L$"],all_rows,"reports/s6a/{scores.json,ablation/scores.json}","lrr"))
    blocks.append(table("AgeProfileTable",["Refit", "35 Jahre","45 Jahre","55 Jahre"],
        [[label,*[num(r["standardized_annual_rate"],5) for r in sorted([r for r in profiles if r["model_id"]==model],key=lambda r:r["assigned_age"])]]
         for label,model in [("GLM","glm_age_bm_a0001"),("GAM","gam_b20_l10"),("HGB","hgb_l15")]],"reports/s6a/existing_age_profiles.json","lrrr"))
    blocks.append(table("GeoTable",["Area",r"Rate/Jahr",r"EUR/Schaden"],[[r["geo"],num(r["frequency"],4),num(r["severity_eur"],0)] for r in geo],"reports/s6a/geo.json","lrr"))
    blocks.append(table("StandardTable",["Mix", "Area",r"Rate/Jahr",r"EUR/Jahr"],
        [[label,r["geo"],num(r["recorded_positive_count"],5),num(r["recorded_loss_eur"],2)] for label,rows in [("ohne Dauer",std0),("mit Dauer",std)] for r in rows],"reports/s6a/standardized_{without,with}_duration.json","llrr"))
    # Preserve all duration groups meeting the displayed exposure threshold.
    fseg=[r for r in data['reports/s4/segments.json'] if r['model_id']=='hgb_l15' and 'exposure' in r['dimension']]
    lseg=[r for r in data['reports/s5/evaluation_loss_segments.json'] if r['model_id']=='pp_hgb_mean' and 'exposure' in r['dimension']]
    for name,rows,path in [('FrequencyDuration',fseg,'reports/s4/segments.json'),('LossDuration',lseg,'reports/s5/evaluation_loss_segments.json')]:
        order=['7 Tage–<0,1 J.','0,1–<0,5 J.','0,5–<1 J.','1+ J.']
        wanted=sorted([r for r in rows if r['exposure_years']>=500],key=lambda r:order.index(r['level']))
        assert len(wanted)==4, f'Unexpected duration support in {path}'
        # Full supported duration table is small; no unsupported cells are hidden behind a score.
        blocks.append(table(name,["Vertragsdauer",r"A/P"],[[r['level'].replace('–','--'),num(r['actual_over_predicted'],3)] for r in wanted],path,"lr"))
    # Native PGFPlots tables retain every plotted value and asymmetric interval.
    observed=[r for r in age if r['age_band']!='Gesamt']
    labels=[r['age_band'].replace('–','--') for r in observed]
    age_dat="x rate low high exposure\n"+"\n".join(f"{i+1} {r['frequency']:.15g} {r['frequency']-r['frequency_lower']:.15g} {r['frequency_upper']-r['frequency']:.15g} {r['exposure']/1000:.15g}" for i,r in enumerate(observed))+"\n"
    bm_dat="x raw standard low high\n"+"\n".join(f"{i+1} {r['frequency']:.15g} {r['standardized_frequency']:.15g} {r['standardized_frequency']-r['lower']:.15g} {r['upper']-r['standardized_frequency']:.15g}" for i,r in enumerate(bm))+"\n"
    aug_dat="x real low high bootstrap\n"+"\n".join(f"{i+1} {r['vs_real']['difference']:.15g} {r['vs_real']['difference']-r['vs_real']['conditional_95_interval'][0]:.15g} {r['vs_real']['conditional_95_interval'][1]-r['vs_real']['difference']:.15g} {r['vs_bootstrap']['difference']:.15g}" for i,r in enumerate(ct))+"\n"
    TABLES['AgeChart']={'origin':'reports/s3/age_bootstrap.json','rows':observed,'unit':'claims / policy-year; exposure in thousands of policy-years','uncertainty':'pointwise 95% whole-policy percentile bootstrap, 500 draws; not simultaneous'}
    TABLES['BMChart']={'origin':'reports/s3/bm_standardized.json','rows':bm,'unit':'claims / policy-year','uncertainty':'pointwise 95% bootstrap with fixed common BM mixture; selected support'}
    TABLES['AugChart']={'origin':'reports/o2/report.json:scores.ctgan','rows':ct,'unit':'Poisson deviance / policy-year, candidate minus control','uncertainty':'fixed-prediction paired asymptotic 95% ratio intervals vs real-only; bootstrap controls shown as points'}
    blocks.append(r"\newcommand{\AgeLabels}{"+",".join(labels)+"}\n")
    generated="% Generated from aggregate reports; no model calls.\n"+"\n".join("\\newcommand{\\"+k+"}{"+v['latex']+"}" for k,v in VALUES.items())+"\n"+"".join(blocks)
    slides={
        '1':{'sources':['reports/s6/results.json'],'population':'existing common evaluation; selected HGB x mean vs intercept x mean','unit':'relative loss deviance reduction','uncertainty':'point estimate; evaluation has historical prior exploration'},
        '2':{'sources':['reports/s0/source_comparison.json','paper/sources.json'],'population':'full source datasets','unit':'contracts; positive claim rows; policy-years; historical EUR','uncertainty':'source reconciliation, not sampling inference'},
        '3':{'sources':['reports/s3/age_bootstrap.json','reports/s3/bm_standardized.json','reports/s3/bm_common_cells.json','reports/s3/age_contrasts.json'],'population':'development; BM plot only common support, distinct from full age bins','unit':'claims/policy-year','uncertainty':'500 pointwise contract bootstrap draws; BM reference held fixed'},
        '4':{'sources':['reports/s6a/geo.json','reports/s6a/standardized_with_duration.json','reports/s6a/standardized_intervals_with_duration.json','reports/s6a/existing_age_profiles.json','paper/sources.json'],'population':'development; supported subset; external road users distinct','unit':'claims/year; EUR/claim; external physical injury shares','uncertainty':'observational; support/tail-dependent; no source linkage'},
        '5':{'sources':['config/frequency_protocol.json'],'population':'training folds only','unit':'rate/year; exposure in years','uncertainty':'model specifications, no effect interval claim'},
        '6':{'sources':['reports/s6/results.json','reports/s4/scores.json','reports/s4/segments.json','reports/s4/paired_bootstrap.json','config/validation_protocol.json'],'population':'development OOF','unit':'Poisson deviance/policy-year; count A/P','uncertainty':'selection estimates; 500 fixed-prediction paired draws'},
        '7':{'sources':['reports/s6/results.json','reports/s5/development_loss_scores.json','reports/s5/evaluation_loss_scores.json','reports/s5/evaluation_bootstrap.json','reports/s3/tail_concentration.json'],'population':'development OOF vs existing evaluation; tail development only','unit':'annual-rate Tweedie deviance p=1.5; EUR/policy-year; loss share','uncertainty':'fixed-prediction intervals; no post-evaluation reselection'},
        '8':{'sources':['reports/s6a/scores.json','reports/s6a/ablation/scores.json','reports/s6a/ablation/dependence.json','reports/s6a/ablation/bootstrap_two_BonusMalus_gam.json'],'population':'exploratory development extension on existing folds','unit':'annual-rate Tweedie deviance p=1.5; weighted correlation','uncertainty':'fixed-fit 500-draw intervals; no effect-stability evidence'},
        '9':{'sources':['reports/o2/report.json','reports/o2/joint_support.json','config/augmentation_protocol.json'],'population':'542392 real OOF policies; 20000 training policies/fold','unit':'Poisson deviance/policy-year; ratio of additional rows','uncertainty':'paired fixed-prediction asymptotic intervals; seed/fold diagnostics separate'},
        '10':{'sources':['reports/s6/results.json','reports/s6a/sources.json','reports/s6a/ablation/report.json'],'population':'historical French insurer; no German transfer validation','unit':'qualitative implications','uncertainty':'external/temporal/German validation and refit stability missing'},
    }
    for path in ['scripts/build_slides.py','slides/fachvortrag.tex','slides/sprechernotizen.md',
                 'src/frequency_models.py','src/severity_models.py','src/interpretation.py',
                 'src/metrics.py','src/severity.py','src/augmentation.py','paper/begleitpaper.tex','paper/literatur.bib']:
        INPUTS[path]={'sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),'origin':'checkout'}
    backups={
        'B1':{'sources':['src/frequency_models.py','paper/begleitpaper.tex'],'unit':'count; rate/year; exposure years','uncertainty':'likelihood identity; balance only at an appropriate training optimum'},
        'B2':{'sources':['src/metrics.py','src/severity.py','paper/begleitpaper.tex'],'unit':'Poisson deviance/year; annual-rate Tweedie deviance p=1.5','uncertainty':'score identities and explicitly stated mean/moment assumptions'},
        'B3':{'sources':['config/frequency_protocol.json','config/interpretation_protocol.json','src/frequency_models.py','src/interpretation.py'],'unit':'objective normalization; basis dimension; conditional EDF','uncertainty':'fixed smoothing, no effect-stability inference'},
        'B4':{'sources':['config/severity_protocol.json','src/severity_models.py'],'unit':'EUR/positive claim row; unit row weights','uncertainty':'global smearing need not be conditionally correct'},
        'B5':{'sources':['paper/begleitpaper.tex','reports/s3/tail_concentration.json'],'unit':'EUR/policy-year; period EUR; EUR/claim','uncertainty':'finite conditional means and product assumption; capping changes estimand'},
        'B6':{'sources':['reports/s6a/existing_age_profiles.json','reports/s3/bm_common_cells.json','reports/s3/bm_standardized.json'],'unit':'claims/year; contracts; policy-years','population':'291234 supported reference contracts vs separate BM support','uncertainty':'no model-refit bands; selected coarse common cells'},
        'B7':{'sources':['reports/s6a/support_with_duration.json','reports/s6a/support_without_duration.json','reports/s6a/standardized_with_duration.json','reports/s6a/standardized_without_duration.json','reports/s6a/standardized_intervals_with_duration.json','reports/s6a/standardized_intervals_without_duration.json'],'unit':'claims/year; EUR/year; contracts/cells','population':'different support populations with vs without duration','uncertainty':'500 whole-policy draws; support and reference weights fixed'},
        'B8':{'sources':['slides/sources.json','reports/s0/source_comparison.json'],'unit':'source populations, periods, geography and physical injury vs payments','uncertainty':'read scope limited to actual original documentation/pages; no linked insurance causal effect'},
        'B9':{'sources':['reports/s6a/scores.json','reports/s6a/bootstrap_best.json','reports/s6a/ablation/scores.json','reports/s6a/ablation/dependence.json'],'unit':'annual-rate Tweedie deviance; weighted Pearson, unweighted Spearman, weighted eta squared','population':'exploratory development only','uncertainty':'500 fixed-fit draws; no full refit or concurvity stability study'},
        'B10':{'sources':['reports/s4/paired_bootstrap.json','reports/s5/evaluation_bootstrap.json','reports/s6a/bootstrap_best.json','src/augmentation.py','paper/begleitpaper.tex'],'unit':'paired score difference on whole contracts','uncertainty':'fixed predictions; O2 ratio delta method vs percentile resampling for core'},
        'B11':{'sources':['reports/o2/report.json','reports/o2/joint_support.json','reports/o2/sources.json','config/augmentation_protocol.json','src/augmentation.py'],'unit':'epochs; candidate/accepted rows; outside empirical conditional range share','population':'15 generators, 20000 real training contracts per fold','uncertainty':'joint-support diagnostic added after registration; no privacy or convergence guarantee'},
        'B12':{'sources':['reports/s6a/costs.json','paper/begleitpaper.tex'],'unit':'hypothetical EUR/year costs and premium share','population':'scenarios, no estimated channel effect','uncertainty':'conditional on stipulated costs; no profitability or German transfer proof'},
        'B13':{'sources':['paper/sources.json','paper/literatur.bib','slides/sources.json'],'unit':'bibliographic identifiers','uncertainty':'methodological references versus actual implementation distinguished'},
    }
    evidence={'schema':1,'build':'python scripts/build_slides.py --compile','scope':'aggregate-only authoring, no model fitting, prediction generation or resampling',
              'baseline_commit':'ce3f18243ac508152d3eeae8099b03c22b447fe1','inputs':INPUTS,'numeric_macros':VALUES,'tables_and_figures':TABLES,'main_slides':slides,
              'backup_slides':backups,'reference_notes':'All S3 inputs explicitly read from published a44c7e91; foreign replay reports never overwritten. S4 replay metadata discrepancy is not used to regenerate S6. S6 numeric fields are checked directly against S4/S5 score artifacts.',
              'bibliography':'../paper/literatur.bib','external_read_scope':'sources.json lists original pages read for S8; no full ONISR report reading claimed.'}
    return {OUT/'generated.tex':generated,OUT/'age.dat':age_dat,OUT/'bm.dat':bm_dat,OUT/'augmentation.dat':aug_dat,
            OUT/'evidence.json':json.dumps(evidence,ensure_ascii=False,indent=2)+"\n"}


def compile_pdfs():
    build=OUT/'build'; build.mkdir(exist_ok=True)
    env={**os.environ,'SOURCE_DATE_EPOCH':'1793736544','FORCE_SOURCE_DATE':'1'}
    for job,definition in [('fachvortrag',None),('hauptvortrag','MainOnly'),('backup','BackupOnly')]:
        arg=("\\def\\"+definition+"{1}" if definition else "")+r"\input{fachvortrag.tex}"
        cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','-no-shell-escape','-output-directory=build',f'-jobname={job}',arg]
        for n in range(2):
            r=subprocess.run(cmd,cwd=OUT,env=env,capture_output=True,text=True,errors='replace')
            (build/f'{job}-pass{n+1}.txt').write_text(r.stdout+r.stderr,encoding='utf-8')
            if r.returncode: raise RuntimeError(f'TeX failed: slides/build/{job}-pass{n+1}.txt')
        log=(build/f'{job}.log').read_text(encoding='utf-8',errors='replace')
        issues=re.findall(r'[^\n]*(?:Overfull|undefined|multiply defined|Missing character)[^\n]*',log)
        if issues: raise RuntimeError('; '.join(issues))
        built, destination = build/f'{job}.pdf', OUT/f'{job}.pdf'
        # Avoid replacing a byte-identical PDF open in a Windows viewer.
        if not destination.exists() or destination.read_bytes()!=built.read_bytes():
            shutil.copyfile(built,destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true'); parser.add_argument('--compile',action='store_true')
    args=parser.parse_args()
    for path,content in generate().items():
        if args.check:
            if not path.exists() or path.read_text(encoding='utf-8')!=content:
                raise ValueError(f'Stale slides input: {path}')
        else: path.write_text(content,encoding='utf-8',newline='\n')
    if args.compile: compile_pdfs()
    print(f'Slides evidence checked/generated: {len(INPUTS)} source hashes, {len(VALUES)} numeric macros; no model calls')


if __name__=='__main__': main()
