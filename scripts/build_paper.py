"""Build the LaTeX companion from existing aggregate evidence; never fit models.

Run from any directory: python scripts/build_paper.py [--check] [--compile].
Requires Python 3.12 (stdlib only), Git, pdfLaTeX and BibTeX for compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SOURCES: dict[str, str] = {}
VALUES: dict[str, dict] = {}
REFERENCE_COMMIT = "a44c7e91c481fab9312af66bdc08cb96ffd1ab85"
VERSIONED_INPUTS = {"reports/s3/tail_concentration.json", "reports/s3/eda_report.json"}


def read(path):
    if path in VERSIONED_INPUTS:
        # Local analysis replays can change run metadata. Use the explicit
        # published reference without modifying the user's working reports.
        result = subprocess.run(["git", "show", f"{REFERENCE_COMMIT}:{path}"],
                                cwd=ROOT, capture_output=True, check=True)
        content = result.stdout
    else:
        content = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(content).hexdigest()
    return json.loads(content.decode("utf-8"))


def number(value, digits=6):
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", "{,}").replace("X", r"\,")


def value(name, raw, origin, digits=6):
    rendered = number(raw, digits)
    VALUES[name] = {"value": raw, "origin": origin, "digits": digits, "latex": rendered}
    return "\\" + name


def row(table, key, wanted):
    return next(r for r in table if r[key] == wanted)


def generate():
    core = read("reports/s6/results.json")
    augmentation = read("reports/o2/report.json")
    joint_support = read("reports/o2/joint_support.json")
    read("reports/o2/sources.json")
    read("config/augmentation_protocol.json")
    if not augmentation["exploratory"] or augmentation["evaluation_used"]:
        raise ValueError("O2 must remain exploratory and development-only")
    real = row(augmentation["scores"], "method", "real")
    value("AugmentationBaseline", real["poisson_deviance_per_exposure"], "reports/o2/report.json:scores.real")
    value("AugmentationSeconds", augmentation["seconds"], "reports/o2/report.json:seconds", 1)
    ctgan = [r for r in augmentation["scores"] if r["method"] == "ctgan"]
    relative = [100*r["vs_real"]["difference"]/real["poisson_deviance_per_exposure"] for r in ctgan]
    value("AugmentationMinLoss", min(relative), "reports/o2/report.json:100*vs_real.difference/real.D_N, minimum", 2)
    value("AugmentationMaxLoss", max(relative), "reports/o2/report.json:100*vs_real.difference/real.D_N, maximum", 2)
    outside = [100*r["outside_training_Area_density_fraction"] for r in joint_support["rows"]]
    value("AugmentationSupportMin", min(outside), "reports/o2/joint_support.json:outside_training_Area_density_fraction, minimum percent", 2)
    value("AugmentationSupportMax", max(outside), "reports/o2/joint_support.json:outside_training_Area_density_fraction, maximum percent", 2)
    augment = [r"\begin{tabular}{rrrrrr}\toprule", r"Ratio & Seed & Bootstrap $D_N$ & CTGAN $D_N$ & $\Delta_{\rm CT-Boot}$ & CTGAN A/P\\\midrule"]
    for r in augmentation["scores"]:
        if r["method"] == "ctgan":
            control = next(b for b in augmentation["scores"] if b["method"] == "bootstrap" and
                           b["ratio"] == r["ratio"] and b["seed"] == r["seed"])
            augment.append(f"{number(r['ratio'],2)} & {r['seed']} & {number(control['poisson_deviance_per_exposure'])} & "
                f"{number(r['poisson_deviance_per_exposure'])} & {number(r['vs_bootstrap']['difference'])} & "
                f"{number(r['actual_over_predicted'],4)}"+r"\\")
    augment.append(r"\bottomrule\end{tabular}")
    conclusion = ("Die vorab festgelegte Bedingung für einen über beide Ratios und alle drei Seeds konsistenten Vorteil gegenüber real-only und Bootstrap wird nicht erfüllt. Die Augmentation wird für diesen Pilot verworfen."
        if augmentation["decision"] == "discard_for_this_pilot" else
        "Die vorab festgelegte Bedingung für eine weitere unabhängige Untersuchung ist erfüllt; ein bestätigter Vorteil im vollständigen Bestand ist damit nicht belegt.")
    reduced = read("reports/s6a/scores.json")
    read("reports/s6a/release.json")
    if any(r["run_id"] != "s6af633eebaf65aa470" for r in reduced):
        raise ValueError("Unexpected S6a evidence identity")
    # Include the exact implemented specifications in the evidence chain.
    for path in ["config/frequency_protocol.json", "config/interpretation_protocol.json",
                 "config/severity_protocol.json", "config/validation_protocol.json"]:
        read(path)
    freq = [r"\begin{tabular}{lrrr}\toprule",
            r"Frequenzregel & $D_N$ & Reduktion (\%) & A/P\\\midrule"]
    labels = ["Intercept", "GLM (Familienminimum)", "GAM (Familienminimum)", "Histogramm-Poisson"]
    for label, r in zip(labels, core["development_frequency"]):
        freq.append(f"{label} & {number(r['poisson_deviance_per_exposure'])} & "
                    f"{number(100*r['relative_deviance_reduction_vs_intercept'],2)} & {number(r['actual_over_predicted'],4)}" + r"\\")
    freq.append(r"\bottomrule\end{tabular}")
    loss = [r"\begin{tabular}{lrrr}\toprule",
            r"Produkt & Entwicklung $D_L$ & Evaluation $D_L$ & Eval. A/P\\\midrule"]
    labels = [r"Intercept $\times$ Mittel", r"HGB $\times$ Mittel", r"HGB $\times$ Gamma", r"HGB $\times$ Smearing"]
    for label, r in zip(labels, core["loss_comparison"]):
        loss.append(f"{label} & {number(r['development_deviance'])} & {number(r['evaluation_deviance'])} & {number(r['evaluation_actual_over_predicted'],4)}" + r"\\")
    loss.append(r"\bottomrule\end{tabular}")
    pairs = [r"\begin{tabular}{lrr}\toprule", r"Zweites Merkmal neben Alter & GLM $D_L$ & GAM $D_L$\\\midrule"]
    for feature, label in [("Area", "Area"), ("Density", r"$\log(\mathrm{Density})$"), ("Region", "Region"), ("BonusMalus", "Bonus-Malus")]:
        a = row(reduced, "model_id", f"two_{feature}_glm")
        b = row(reduced, "model_id", f"two_{feature}_gam")
        pairs.append(f"{label} & {number(a['tweedie_deviance_per_exposure'])} & {number(b['tweedie_deviance_per_exposure'])}" + r"\\")
    pairs.append(r"\bottomrule\end{tabular}")
    best = row(reduced, "model_id", "two_BonusMalus_gam")["tweedie_deviance_per_exposure"]
    full = row(reduced, "model_id", "pp_hgb_mean")["tweedie_deviance_per_exposure"]
    value("ReducedScore", best, "reports/s6a/scores.json:two_BonusMalus_gam", 10)
    value("FullScore", full, "reports/s6a/scores.json:pp_hgb_mean", 10)
    value("ReducedGap", 100*(best/full-1), "100*(two_BonusMalus_gam/pp_hgb_mean-1)", 4)
    for name, field, digits in [("EvalPolicies","policies",0),("EvalClaims","observed_claims",0),
                                 ("EvalObserved","observed_annual_loss_cost",2),("EvalPredicted","predicted_annual_loss_cost",2),
                                 ("EvalAP","actual_over_predicted",4)]:
        value(name,core["selected_evaluation"][field],f"reports/s6/results.json:selected_evaluation.{field}",digits)
    value("EvalGain",100*core["relative_loss_deviance_reduction"],"reports/s6/results.json:relative_loss_deviance_reduction",2)
    boot = read("reports/s6a/bootstrap_best.json")
    for prefix, model, sign in [("FullGap", "pp_hgb_mean", -1), ("BMGap", "two_BonusMalus_glm", 1)]:
        r = row(boot,"model_id",model)
        value(prefix,sign*r['deviance_difference'],f"reports/s6a/bootstrap_best.json:{model}, sign={sign}")
        lo, hi = (r['lower'],r['upper']) if sign==1 else (-r['upper'],-r['lower'])
        value(prefix+"Lo",lo,f"reports/s6a/bootstrap_best.json:{model}:oriented lower")
        value(prefix+"Hi",hi,f"reports/s6a/bootstrap_best.json:{model}:oriented upper")
    geo = read("reports/s6a/geo.json")
    geotab = [r"\begin{tabular}{lrrrr}\toprule",r"Area-Gruppe & Rate/Jahr & EUR/Schaden & EUR/Jahr & mittl. Dauer\\\midrule"]
    for r in geo:
        geotab.append(f"{r['geo']} & {number(r['frequency'])} & {number(r['severity_eur'],2)} & {number(r['annual_loss_eur'],2)} & {number(r['mean_duration'],3)}"+r"\\")
    geotab.append(r"\bottomrule\end{tabular}")
    value("RuralTailShare",100*geo[0]['loss_share_ge_100000'],"reports/s6a/geo.json:AB.loss_share_ge_100000",2)
    tail=read("reports/s3/tail_concentration.json")['rows'][0]
    value("LargestShare",100*tail['loss_share'],"reports/s3/tail_concentration.json:rows[0].loss_share",2)
    value("LargestClaim",tail['loss_eur'],"reports/s3/tail_concentration.json:rows[0].loss_eur",2)
    age = read("reports/s6a/existing_age_profiles.json")
    agetab = [r"\begin{tabular}{lrrr}\toprule",r"Modell & Alter 35 & Alter 45 & Alter 55\\\midrule"]
    for model,label in [("glm_age_bm_a0001","GLM"),("gam_b20_l10","GAM"),("hgb_l15","HGB")]:
        rows=sorted([r for r in age if r['model_id']==model],key=lambda r:r['assigned_age'])
        agetab.append(label+" & "+" & ".join(number(r['standardized_annual_rate']) for r in rows)+r"\\")
    agetab.append(r"\bottomrule\end{tabular}")
    value("AgePolicies",age[0]['reference_policies'],"reports/s6a/existing_age_profiles.json:reference_policies",0)
    value("AgeExposure",age[0]['reference_exposure_years'],"reports/s6a/existing_age_profiles.json:reference_exposure_years",2)
    stand = []
    for kind,label in [("without_duration","ohne Dauerzellen"),("with_duration","mit Dauerzellen")]:
        table=read(f"reports/s6a/standardized_{kind}.json")
        for r in table:
            stand.append(f"{label} & {r['geo']} & {number(r['recorded_positive_count'])} & {number(r['recorded_loss_eur'],2)}"+r"\\")
    standard = [r"\begin{tabular}{llrr}\toprule",r"Unterstützung & Gruppe & Rate/Jahr & EUR/Jahr\\\midrule",*stand,r"\bottomrule\end{tabular}"]
    for p in ["reports/s6a/geo_intervals.json","reports/s6a/standardized_intervals_without_duration.json",
              "reports/s6a/standardized_intervals_with_duration.json","reports/s6a/common_geo_with_duration.json",
              "reports/s5/evaluation_bootstrap.json","reports/s6a/geography_calibration.json",
              "reports/s0/source_comparison.json", "reports/s3/eda_report.json",
              "reports/s6a/existing_age_contrasts.json", "reports/s6a/costs.json",
              "reports/s5/evaluation_loss_segments.json", "reports/s5/evaluation_severity_scores.json",
              "paper/sources.json"]:
        read(p)
    for path in ["src/frequency_models.py", "src/severity_models.py", "src/interpretation.py",
                 "src/metrics.py", "src/severity.py", "src/augmentation.py", "requirements.lock.txt"]:
        SOURCES[path] = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    macro = ["% Generated by scripts/build_paper.py; do not edit."]
    macro += [f"\\newcommand{{\\{name}}}{{{v['latex']}}}" for name,v in VALUES.items()]
    macro.append("\\newcommand{\\AugmentationConclusion}{"+conclusion+"}")
    for name,lines in [("FrequencyTable",freq),("LossTable",loss),("ReducedTable",pairs),("GeoTable",geotab),("AgeTable",agetab),("StandardTable",standard),("AugmentationTable",augment)]:
        macro += [f"\\newcommand{{\\{name}}}{{%",*lines,"}"]
    manifest={"scope":"Existing aggregate evidence only; no fitting or evaluation rerun", "s6a_run":"s6af633eebaf65aa470",
              "o2_run":augmentation["run_id"],
              "sources_sha256":SOURCES,
              "versioned_inputs":{p:REFERENCE_COMMIT for p in sorted(VERSIONED_INPUTS)},
              "numeric_macros":VALUES,
              "table_origins":{"FrequencyTable":"reports/s6/results.json:development_frequency", "LossTable":"reports/s6/results.json:loss_comparison",
                               "ReducedTable":"reports/s6a/scores.json", "GeoTable":"reports/s6a/geo.json",
                               "AgeTable":"reports/s6a/existing_age_profiles.json", "StandardTable":"reports/s6a/standardized_{without,with}_duration.json",
                               "AugmentationTable":"reports/o2/report.json:scores.ctgan and matching scores.bootstrap; paired CTGAN-minus-bootstrap"}}
    return {PAPER/"generated.tex":"\n".join(macro)+"\n",PAPER/"evidence.json":json.dumps(manifest,ensure_ascii=False,indent=2)+"\n"}


def compile_pdf():
    for exe in ("pdflatex","bibtex"):
        if not shutil.which(exe):
            raise RuntimeError(f"Required TeX program missing: {exe}")
    out=PAPER/"build"
    out.mkdir(exist_ok=True)
    commands=[["pdflatex","-interaction=nonstopmode","-halt-on-error","-file-line-error","-no-shell-escape","-output-directory=build","begleitpaper.tex"],
              ["bibtex","begleitpaper"],
              ["pdflatex","-interaction=nonstopmode","-halt-on-error","-file-line-error","-no-shell-escape","-output-directory=build","begleitpaper.tex"],
              ["pdflatex","-interaction=nonstopmode","-halt-on-error","-file-line-error","-no-shell-escape","-output-directory=build","begleitpaper.tex"]]
    # BibTeX runs in build; copy only its local bibliography input.
    shutil.copyfile(PAPER/"literatur.bib",out/"literatur.bib")
    for i,command in enumerate(commands):
        r=subprocess.run(command,cwd=out if command[0]=="bibtex" else PAPER,capture_output=True,text=True,errors="replace")
        (out/f"pass-{i+1}.txt").write_text(r.stdout+r.stderr,encoding="utf-8")
        if r.returncode:
            raise RuntimeError(f"TeX command failed; see paper/build/pass-{i+1}.txt")
    log=(out/"begleitpaper.log").read_text(encoding="utf-8",errors="replace")
    issues=re.findall(r"[^\n]*(?:Overfull|undefined|multiply defined|Citation.*Warning)[^\n]*",log)
    if issues:
        raise RuntimeError("Unresolved TeX diagnostics: "+"; ".join(issues))
    shutil.copyfile(out/"begleitpaper.pdf",PAPER/"begleitpaper.pdf")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--compile",action="store_true")
    args=parser.parse_args()
    for path,content in generate().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=content:
                raise ValueError(f"Stale generated paper evidence: {path}")
        else:
            path.write_text(content,encoding="utf-8",newline="\n")
    if args.compile:
        compile_pdf()
    print(f"Paper evidence {'checked' if args.check else 'generated'}: {len(SOURCES)} sources, {len(VALUES)} numeric macros; no model calls")


if __name__=="__main__":
    main()
