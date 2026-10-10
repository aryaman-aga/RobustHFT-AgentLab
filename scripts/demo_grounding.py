#!/usr/bin/env python
"""Demo runner: Focus exclusively on Temporal Grounding.

Compares:
  1. Undefended Agent (Grounding=False, Risk=False, Micro=False)
  2. Grounded Agent   (Grounding=True,  Risk=False, Micro=False)

Usage:
  .venv/bin/python scripts/demo_grounding.py
  .venv/bin/python scripts/demo_grounding.py --seed 42 --kappa 0.7
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repo root is on python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rhal.config import load_config
from rhal.evaluation.experiment import RunSpec, Simulation, attach_net_asr, prepare_dataset
from rhal.misgen import DetectabilityModel


def main():
    parser = argparse.ArgumentParser(description="Demonstrate Temporal Grounding Defense")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="Config YAML path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--kappa", type=float, default=0.7, help="Grounding strength kappa (default: 0.7)")
    parser.add_argument("--show-flips", type=int, default=10, help="Number of flip examples to display")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    kappa = args.kappa

    print("=" * 80)
    print("      ROBUSTHFT-AGENTLAB: TEMPORAL GROUNDING DEFENSE EVALUATION")
    print("=" * 80)
    print(f"Configuration : seed={seed}, grounding_strength(kappa)={kappa}")
    print(f"Data Source   : {cfg.data.n_days} days history ({cfg.data.eval_days} evaluation days)")
    print(f"Retrieval     : {cfg.retrieval.method.upper()} top_k={cfg.retrieval.top_k}")
    print("=" * 80)

    # 1. Load or train detector
    det_path = Path(cfg.output_dir) / "models" / "detector.pkl"
    if det_path.exists():
        det = DetectabilityModel.load(det_path)
    else:
        print("[!] Trained detector not found, generating sample detector...")
        from rhal.evaluation.experiment import train_detector
        det = train_detector(cfg, seed)

    # 2. Prepare dataset
    print("\n[+] Preparing dataset and historical market features...")
    ds = prepare_dataset(cfg, seed)
    sim = Simulation(cfg, ds, det, seed)

    # 3. Define specs
    # Baseline (Undefended)
    spec_undef_clean = RunSpec("clean__none", injection="none", grounded=False, risk=False, micro=False)
    spec_undef_inj   = RunSpec("injected__none", injection="misgen", grounded=False, risk=False, micro=False)
    spec_undef_plac  = RunSpec("placebo__none", injection="placebo", grounded=False, risk=False, micro=False)

    # Grounding Only (Temporal Grounding Active, No Risk, No Micro)
    ground_ov = {"grounding.strength": kappa}
    spec_grd_clean   = RunSpec("clean__grounding_only", injection="none", grounded=True, risk=False, micro=False, overrides=ground_ov)
    spec_grd_inj     = RunSpec("injected__grounding_only", injection="misgen", grounded=True, risk=False, micro=False, overrides=ground_ov)
    spec_grd_plac    = RunSpec("placebo__grounding_only", injection="placebo", grounded=True, risk=False, micro=False, overrides=ground_ov)

    print("[+] Simulating Undefended Agent...")
    res_undef_clean = sim.run(spec_undef_clean)
    res_undef_inj   = sim.run(spec_undef_inj)
    res_undef_plac  = sim.run(spec_undef_plac)

    print("[+] Simulating Grounded Agent (with strict e + H <= t admissibility)...")
    res_grd_clean   = sim.run(spec_grd_clean)
    res_grd_inj     = sim.run(spec_grd_inj)
    res_grd_plac    = sim.run(spec_grd_plac)

    # Calculate net ASR
    rows = [
        res_undef_clean.to_row(), res_undef_inj.to_row(), res_undef_plac.to_row(),
        res_grd_clean.to_row(),   res_grd_inj.to_row(),   res_grd_plac.to_row()
    ]
    rows = attach_net_asr(rows)
    lookup = {r["arm"]: r for r in rows}

    u_inj = lookup["injected__none"]
    u_plc = lookup["placebo__none"]
    g_inj = lookup["injected__grounding_only"]
    g_plc = lookup["placebo__grounding_only"]

    print("\n" + "=" * 80)
    print("                      HEAD-TO-HEAD PERFORMANCE")
    print("=" * 80)
    header = f"{'Metric':<28} | {'Undefended (None)':<18} | {'Temporal Grounding Only':<22}"
    print(header)
    print("-" * len(header))
    print(f"{'Naive ASR (Total Flips)':<28} | {u_inj['ASR']*100:>15.1f}%   | {g_inj['ASR']*100:>19.1f}%")
    print(f"{'Placebo ASR (Noise Floor)':<28} | {u_plc['ASR']*100:>15.1f}%   | {g_plc['ASR']*100:>19.1f}%")
    u_net = u_inj.get('ASR_net', u_inj['ASR'] - u_plc['ASR']) * 100
    g_net = g_inj.get('ASR_net', g_inj['ASR'] - g_plc['ASR']) * 100
    print(f"{'Net ASR (Misinfo Attributable)':<28} | {u_net:>15.1f}%   | {g_net:>19.1f}%")
    print(f"{'Misinfo Exposure Rate (MER)':<28} | {u_inj['MER']*100:>15.1f}%   | {g_inj['MER']*100:>19.1f}%")
    print(f"{'Signal Deviation':<28} | {u_inj['signal_deviation']:>16.3f}   | {g_inj['signal_deviation']:>20.3f}")
    print(f"{'Clean Return (No Attack)':<28} | {u_inj['CR_clean_paired']*100:>+15.2f}%   | {g_inj['CR_clean_paired']*100:>+19.2f}%")
    print(f"{'Attacked Return (Injected)':<28} | {u_inj['CR']*100:>+15.2f}%   | {g_inj['CR']*100:>+19.2f}%")
    print(f"{'Max Drawdown Under Attack':<28} | {u_inj['max_drawdown']*100:>15.2f}%   | {g_inj['max_drawdown']*100:>19.2f}%")
    print("=" * 80)

    # 4. Step-by-step inspection: Days where grounding successfully intervened
    print("\n" + "=" * 80)
    print("    STEP-BY-STEP TRACE: HOW TEMPORAL GROUNDING INTERVENES AGAINST ATTACKS")
    print("=" * 80)

    interventions = []
    for t_idx in range(len(res_undef_clean.treated_steps)):
        s_step = res_undef_clean.treated_steps[t_idx] # Clean shadow
        u_step = res_undef_inj.treated_steps[t_idx]   # Undefended injected
        g_step = res_grd_inj.treated_steps[t_idx]     # Grounded injected

        clean_act = s_step["action"]
        undef_act = u_step["action"]
        grd_act   = g_step["action"]

        # Was there an attack flip in undefended?
        if undef_act != clean_act:
            protected = (grd_act == clean_act)
            interventions.append({
                "date": s_step["date"],
                "clean_act": clean_act,
                "undef_act": undef_act,
                "grd_act": grd_act,
                "protected": protected,
                "text_sig": g_step["text_signal"],
                "mkt_sig": g_step["market_signal"],
                "consistency": g_step["consistency"],
                "rationale": g_step["rationale"],
            })

    print(f"Total days analyzed: {len(res_undef_clean.treated_steps)}")
    print(f"Days Undefended Agent was flipped by attack: {len(interventions)}")
    saved_count = sum(1 for item in interventions if item['protected'])
    print(f"Days Temporal Grounding neutralized the flip: {saved_count} ({saved_count/max(len(interventions), 1)*100:.1f}%)")
    print("\nSample Days showing attack intervention:\n")

    sample_items = interventions[:args.show_flips]
    for idx, item in enumerate(sample_items, 1):
        status = "[DEFENDED]" if item["protected"] else "[FAILED]"
        print(f"Day {idx:02d} ({item['date']}): {status}")
        print(f"  * Clean Decision      : {item['clean_act']:<4}")
        print(f"  * Undefended Decision : {item['undef_act']:<4}  <-- FLIPPED by adversarial text!")
        print(f"  * Grounded Decision   : {item['grd_act']:<4}  <-- {'Corrected back to clean!' if item['protected'] else 'Remained flipped'}")
        print(f"  * Signals             : text_signal={item['text_sig']:+.2f}, market_evidence={item['mkt_sig']:+.2f}")
        print(f"  * Consistency (c)     : {item['consistency']:.2f} (coupling factor kappa={kappa})")
        print(f"  * Policy Rationale    : {item['rationale']}")
        print()

    print("=" * 80)
    print("KEY TAKEAWAYS FOR PRESENTATION / EVALUATION:")
    print("1. In undefended mode, the agent relies heavily on textual sentiment, so adversarial")
    print("   news flips decisions readily.")
    print("2. Temporal Grounding retrieves historical analogs (e + H <= t) and evaluates consistency c.")
    print("3. When adversarial text contradicts the market (c is low), shrinkage s_g pulls the signal")
    print("   back towards quantitative reality, successfully preventing the decision flip.")
    print("=" * 80)


if __name__ == "__main__":
    main()
