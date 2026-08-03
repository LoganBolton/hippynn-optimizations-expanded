#!/usr/bin/env python3
"""Build a self-contained interactive gallery of robust L3N5 winners."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def tetrahedral_asymmetry(vectors: np.ndarray, distances: np.ndarray) -> np.ndarray:
    normalized = vectors / distances.mean(axis=1)[:, None, None]
    ideal = np.asarray(
        ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)), dtype=np.float64
    ) / np.sqrt(3)
    observed_norm = np.sum(np.square(normalized), axis=(1, 2))
    ideal_norm = float(np.sum(np.square(ideal)))
    best_squared = np.full(len(normalized), np.inf)
    for permutation in itertools.permutations(range(4)):
        covariance = np.einsum("nki,kj->nij", normalized, ideal[list(permutation)])
        singular_values = np.linalg.svd(covariance, compute_uv=False)
        squared = (observed_norm + ideal_norm - 2 * singular_values.sum(axis=1)) / 4
        best_squared = np.minimum(best_squared, squared)
    return np.sqrt(np.maximum(best_squared, 0))


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    input_dir = here / "per_point" / "random_common_80k"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=input_dir)
    parser.add_argument("--analysis-dir", type=Path, default=input_dir / "l3n4_l3n5_subset_analysis")
    parser.add_argument("--output", type=Path, default=input_dir / "l3n4_l3n5_subset_analysis" / "top200_molecule_gallery.html")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--normal", type=int, default=20)
    parser.add_argument("--remove-source", type=int, default=3937730)
    return parser.parse_args()


def descriptors(positions: np.ndarray, forces: np.ndarray) -> dict[str, np.ndarray]:
    vectors = positions[:, 1:] - positions[:, :1]
    distances = np.linalg.norm(vectors, axis=2)
    units = vectors / distances[:, :, None]
    i, j = np.triu_indices(4, 1)
    cosines = np.einsum("nia,nja->nij", units, units)[:, i, j]
    angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
    chh_by_center = []
    hhh_by_center = []
    for center_index in range(1, 5):
        center = positions[:, center_index]
        carbon_vector = positions[:, 0] - center
        carbon_unit = carbon_vector / np.linalg.norm(carbon_vector, axis=1, keepdims=True)
        other_hydrogens = [index for index in range(1, 5) if index != center_index]
        hydrogen_vectors = positions[:, other_hydrogens] - center[:, None, :]
        hydrogen_units = hydrogen_vectors / np.linalg.norm(hydrogen_vectors, axis=2, keepdims=True)
        chh_by_center.append(
            np.degrees(np.arccos(np.clip(np.einsum("na,nha->nh", carbon_unit, hydrogen_units), -1, 1)))
        )
        h_i, h_j = np.triu_indices(3, 1)
        hhh_cosines = np.einsum("nha,nka->nhk", hydrogen_units, hydrogen_units)[:, h_i, h_j]
        hhh_by_center.append(np.degrees(np.arccos(np.clip(hhh_cosines, -1, 1))))
    hydrogen_center_means = np.stack(
        [np.concatenate((chh, hhh), axis=1).mean(axis=1) for chh, hhh in zip(chh_by_center, hhh_by_center)],
        axis=1,
    )
    chh_angles = np.concatenate(chh_by_center, axis=1)
    hhh_angles = np.concatenate(hhh_by_center, axis=1)
    atom_center_means = np.column_stack((angles.mean(axis=1), hydrogen_center_means))
    return {
        "radial_cv": distances.std(axis=1) / distances.mean(axis=1),
        "angular_rmse": np.sqrt(np.mean(np.square(angles - 109.4712206), axis=1)),
        "hch_mean": angles.mean(axis=1),
        "chh_mean": chh_angles.mean(axis=1),
        "hhh_mean": hhh_angles.mean(axis=1),
        "hydrogen_center_mean": np.concatenate((chh_angles, hhh_angles), axis=1).mean(axis=1),
        "all_angle_mean": np.concatenate((angles, chh_angles, hhh_angles), axis=1).mean(axis=1),
        "atom_center_mean_std": atom_center_means.std(axis=1),
        "tetrahedral_asymmetry": tetrahedral_asymmetry(vectors, distances),
        "ch_mean": distances.mean(axis=1),
        "ch_min": distances.min(axis=1),
        "ch_max": distances.max(axis=1),
        "max_force": np.linalg.norm(forces, axis=2).max(axis=1),
    }


def main() -> int:
    args = parse_args()
    winner_path = args.analysis_dir / "robust_l3n5_winners.csv"
    comparison_path = args.analysis_dir / "molecule_level_architecture_comparison.csv"
    winners = pd.read_csv(winner_path)
    comparison = pd.read_csv(comparison_path).set_index("source_index")

    npz_path = sorted(args.input_dir.glob("l3_n4_*_per_point.npz"))[0]
    with np.load(npz_path) as data:
        source = data["source_indices"].astype(np.int64)
        positions = data["positions"].astype(np.float64)
        energies = data["true_energy"].astype(np.float64)
        forces = data["true_forces"].astype(np.float64)
    feature = descriptors(positions, forces)
    source_to_row = {int(value): i for i, value in enumerate(source)}

    top_by_metric = {
        metric: winners[winners.metric == metric].nsmallest(args.top, "rank").copy()
        for metric in ("energy_abs_error", "force_rmse")
    }
    top_ids = {
        metric: set(frame.source_index.astype(int))
        for metric, frame in top_by_metric.items()
    }
    all_top_ids = top_ids["energy_abs_error"] | top_ids["force_rmse"]

    n_structures = len(source)
    normal_score = (
        rankdata(feature["tetrahedral_asymmetry"]) / n_structures
        + 0.35 * rankdata(feature["max_force"]) / n_structures
    )
    normal_eligible = (
        (source != args.remove_source)
        & ~np.isin(source, np.fromiter(all_top_ids, dtype=np.int64))
    )
    eligible_rows = np.flatnonzero(normal_eligible)
    normal_rows = eligible_rows[np.argsort(normal_score[eligible_rows])[: args.normal]]

    def base_entry(source_id: int) -> dict:
        row = source_to_row[source_id]
        centered = positions[row] - positions[row, :1]
        errors = comparison.loc[source_id]
        return {
            "source": source_id,
            "positions": np.round(centered, 6).tolist(),
            "targetEnergy": round(float(energies[row]), 6),
            "maxForce": round(float(feature["max_force"][row]), 6),
            "tetrahedralAsymmetry": round(float(feature["tetrahedral_asymmetry"][row]), 6),
            "radialCV": round(float(feature["radial_cv"][row]), 6),
            "angularRMSE": round(float(feature["angular_rmse"][row]), 6),
            "hchMean": round(float(feature["hch_mean"][row]), 6),
            "chhMean": round(float(feature["chh_mean"][row]), 6),
            "hhhMean": round(float(feature["hhh_mean"][row]), 6),
            "hydrogenCenterMean": round(float(feature["hydrogen_center_mean"][row]), 6),
            "allAngleMean": round(float(feature["all_angle_mean"][row]), 6),
            "atomCenterMeanStd": round(float(feature["atom_center_mean_std"][row]), 6),
            "chMean": round(float(feature["ch_mean"][row]), 6),
            "chMin": round(float(feature["ch_min"][row]), 6),
            "chMax": round(float(feature["ch_max"][row]), 6),
            "energyL3N4": round(float(errors.l3n4_median_energy_abs_error), 6),
            "energyL3N5": round(float(errors.l3n5_median_energy_abs_error), 6),
            "forceL3N4": round(float(errors.l3n4_median_force_rmse), 6),
            "forceL3N5": round(float(errors.l3n5_median_force_rmse), 6),
            "inEnergyTop": source_id in top_ids["energy_abs_error"],
            "inForceTop": source_id in top_ids["force_rmse"],
        }

    entries = []
    for metric, group in (("energy_abs_error", "energy"), ("force_rmse", "force")):
        for winner in top_by_metric[metric].sort_values("rank").itertuples(index=False):
            entry = base_entry(int(winner.source_index))
            entry.update(
                {
                    "key": f"{group}-{winner.source_index}",
                    "group": group,
                    "rank": int(winner.rank),
                    "absoluteImprovement": round(float(winner.absolute_improvement), 6),
                    "percentImprovement": round(float(winner.percent_improvement), 3),
                }
            )
            entries.append(entry)

    for normal_rank, row in enumerate(normal_rows, start=1):
        source_id = int(source[row])
        entry = base_entry(source_id)
        entry.update(
            {
                "key": f"normal-{source_id}",
                "group": "normal",
                "rank": normal_rank,
                "absoluteImprovement": None,
                "percentImprovement": None,
                "normalScore": round(float(normal_score[row]), 6),
            }
        )
        entries.append(entry)

    payload = json.dumps(entries, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__MOLECULE_DATA__", payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {len(entries)} cards ({args.top} energy, {args.top} force, {args.normal} normal) to {args.output}")
    print(f"Unique top-winner structures: {len(all_top_ids)}; shared by both rankings: {len(top_ids['energy_abs_error'] & top_ids['force_rmse'])}")
    return 0


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>L3N5 vs L3N4: top molecule gallery</title>
<style>
:root { --ink:#18202a; --muted:#647184; --line:#dce2e9; --paper:#f4f6f8; --card:#fff; --energy:#315fa8; --force:#d06427; --normal:#27866a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; }
header { position:sticky; top:0; z-index:20; padding:18px 24px 14px; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }
h1 { margin:0 0 4px; font-size:22px; }
.sub { color:var(--muted); margin-bottom:13px; }
.controls { display:flex; flex-wrap:wrap; align-items:center; gap:9px; }
button,select,input { border:1px solid #bbc5d0; border-radius:7px; background:#fff; color:var(--ink); padding:8px 10px; font:inherit; }
button { cursor:pointer; }
button.active { color:#fff; border-color:#27384c; background:#27384c; }
input[type=search] { width:190px; }
.check { display:flex; align-items:center; gap:5px; color:var(--muted); }
.check input { width:auto; }
main { padding:18px 24px 40px; }
#summary { margin:0 0 14px; color:var(--muted); }
#gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(285px,1fr)); gap:14px; }
.card { overflow:hidden; background:var(--card); border:1px solid var(--line); border-radius:11px; box-shadow:0 2px 7px rgba(19,30,42,.06); }
.card.energy { border-top:4px solid var(--energy); }
.card.force { border-top:4px solid var(--force); }
.card.normal { border-top:4px solid var(--normal); }
.heading { display:flex; justify-content:space-between; gap:8px; align-items:flex-start; padding:11px 12px 5px; }
.source { font-size:16px; font-weight:700; }
.rank { color:var(--muted); font-weight:600; }
.badges { display:flex; gap:5px; flex-wrap:wrap; padding:0 12px 6px; min-height:25px; }
.badge { padding:2px 7px; border-radius:999px; color:#fff; font-size:11px; font-weight:700; }
.badge.energy { background:var(--energy); }.badge.force { background:var(--force); }.badge.normal { background:var(--normal); }
canvas { display:block; width:100%; height:230px; cursor:grab; background:radial-gradient(circle at 50% 48%,#fff 0,#f8fafc 68%,#edf1f5 100%); border-block:1px solid #edf0f3; }
canvas:active { cursor:grabbing; }
.stats { display:grid; grid-template-columns:1fr 1fr; gap:5px 12px; padding:10px 12px 12px; font-variant-numeric:tabular-nums; }
.stat { display:flex; justify-content:space-between; gap:5px; border-bottom:1px dotted #e2e6eb; }
.stat span:first-child { color:var(--muted); }
.wide { grid-column:1 / -1; }
.help { margin-top:17px; padding:13px 15px; background:#fff; border:1px solid var(--line); border-radius:9px; color:var(--muted); }
.dot { display:inline-block; width:11px; height:11px; border-radius:50%; vertical-align:-1px; margin-right:3px; border:1px solid #888; }
@media(max-width:650px){ header{position:relative;padding:15px} main{padding:14px} #gallery{grid-template-columns:1fr} }
</style>
</head>
<body>
<header>
  <h1>L3N5’s strongest robust wins over L3N4</h1>
  <div class="sub">100 ranked by energy-error improvement + 100 ranked by force-RMSE improvement + 20 normal-looking references. Drag any molecule to rotate; double-click to reset.</div>
  <div class="controls">
    <button class="filter active" data-group="all">All 220</button>
    <button class="filter" data-group="energy">Energy top 100</button>
    <button class="filter" data-group="force">Force top 100</button>
    <button class="filter" data-group="normal">Normal references</button>
    <input id="search" type="search" inputmode="numeric" placeholder="Find source ID">
    <select id="sort"><option value="rank">Sort by rank</option><option value="force-desc">Target force, high to low</option><option value="asymmetry-desc">Combined asymmetry, high to low</option><option value="radial-desc">Radial CV, high to low</option><option value="angle-desc">Angular distortion, high to low</option><option value="source">Source ID</option></select>
    <label class="check"><input id="sameScale" type="checkbox"> same spatial scale</label>
  </div>
</header>
<main>
  <div id="summary"></div>
  <div id="gallery"></div>
  <div class="help"><strong>Rendering:</strong> <span class="dot" style="background:#333"></span>carbon, <span class="dot" style="background:#fff"></span>hydrogen. Each card is normalized to its own maximum C–H radius unless “same spatial scale” is enabled. Radial CV measures inequality among the four C–H distances; angular distortion is the six-angle RMSE from 109.47°. “Normal” references minimize a combined rank of radial CV, angular distortion, and target force and are excluded from both winner lists. The four structures shared by both top-100 rankings appear once in each ranking tab.</div>
</main>
<script>
const molecules=__MOLECULE_DATA__;
const gallery=document.getElementById('gallery'), summary=document.getElementById('summary');
let group='all', query='', sort='rank', sameScale=false;
const rotations=new Map();
const fmt=(x,d=3)=>x==null?'—':Number(x).toFixed(d);
function badges(m){let b=''; if(m.group==='normal') return '<span class="badge normal">normal reference</span>'; if(m.inEnergyTop)b+='<span class="badge energy">energy top 100</span>'; if(m.inForceTop)b+='<span class="badge force">force top 100</span>'; return b;}
function card(m){const metric=m.group==='energy'?'Energy |error|':m.group==='force'?'Force RMSE':'Reference'; const err=m.group==='energy'?`${fmt(m.energyL3N4)} → ${fmt(m.energyL3N5)}`:m.group==='force'?`${fmt(m.forceL3N4)} → ${fmt(m.forceL3N5)}`:'—'; return `<article class="card ${m.group}" data-key="${m.key}"><div class="heading"><div class="source">Source ${m.source}</div><div class="rank">${m.group==='normal'?'normal #':'rank #'}${m.rank}</div></div><div class="badges">${badges(m)}</div><canvas aria-label="Interactive methane geometry for source ${m.source}"></canvas><div class="stats"><div class="stat wide"><span>${metric}</span><b>${err}</b></div>${m.group==='normal'?'':`<div class="stat wide"><span>Improvement</span><b>${fmt(m.absoluteImprovement)} (${fmt(m.percentImprovement,1)}%)</b></div>`}<div class="stat wide"><span>Combined tetrahedral asymmetry</span><b>${fmt(m.tetrahedralAsymmetry,3)}</b></div><div class="stat"><span>Radial CV</span><b>${fmt(m.radialCV)}</b></div><div class="stat"><span>H–C–H RMSE</span><b>${fmt(m.angularRMSE,1)}°</b></div><div class="stat"><span>Mean H–C–H</span><b>${fmt(m.hchMean,1)}°</b></div><div class="stat"><span>Mean C–H–H</span><b>${fmt(m.chhMean,1)}°</b></div><div class="stat"><span>All H-centered</span><b>${fmt(m.hydrogenCenterMean,1)}°</b></div><div class="stat"><span>Center-mean variation</span><b>${fmt(m.atomCenterMeanStd,1)}°</b></div><div class="stat"><span>All-angle mean</span><b>${fmt(m.allAngleMean,1)}°*</b></div><div class="stat"><span>Mean H–H–H</span><b>${fmt(m.hhhMean,1)}°*</b></div><div class="stat"><span>C–H mean</span><b>${fmt(m.chMean)}</b></div><div class="stat"><span>C–H min–max</span><b>${fmt(m.chMin)}–${fmt(m.chMax)}</b></div><div class="stat"><span>Max target force</span><b>${fmt(m.maxForce,1)}</b></div><div class="stat"><span>Target energy</span><b>${fmt(m.targetEnergy,1)}</b></div><div class="wide" style="color:#7b8490;font-size:11px">*Always 60° by triangle-angle identity; shown for completeness.</div></div></article>`;}
function visible(){let a=molecules.filter(m=>(group==='all'||m.group===group)&&(!query||String(m.source).includes(query))); const dir=sort.endsWith('-desc')?-1:1; a.sort((x,y)=>{if(sort==='source')return x.source-y.source;if(sort==='force-desc')return dir*(x.maxForce-y.maxForce);if(sort==='asymmetry-desc')return dir*(x.tetrahedralAsymmetry-y.tetrahedralAsymmetry);if(sort==='radial-desc')return dir*(x.radialCV-y.radialCV);if(sort==='angle-desc')return dir*(x.angularRMSE-y.angularRMSE);const go={energy:0,force:1,normal:2};return (go[x.group]-go[y.group])||x.rank-y.rank;});return a;}
function render(){const data=visible();gallery.innerHTML=data.map(card).join('');summary.textContent=`Showing ${data.length} cards${group==='all'?' (216 unique winner/reference structures; four winners are intentionally shown in both rankings)':''}.`; gallery.querySelectorAll('.card').forEach((el,i)=>attach(el,data[i]));}
function rotatePoint(p,yaw,pitch){let x=p[0],y=p[1],z=p[2];let cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch);let x1=cy*x+sy*z,z1=-sy*x+cy*z;return [x1,cx*y-sx*z1,sx*y+cx*z1];}
function draw(canvas,m){const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);const rot=rotations.get(m.key)||{yaw:-.65,pitch:.48};const pts=m.positions.map(p=>rotatePoint(p,rot.yaw,rot.pitch));const maxR=Math.max(...m.positions.slice(1).map(p=>Math.hypot(...p)));const scale=sameScale?42:Math.min(70,Math.min(w,h)*.39/maxR);const project=p=>[w/2+p[0]*scale,h/2-p[1]*scale,p[2]];const q=pts.map(project);ctx.lineCap='round';for(let i=1;i<5;i++){let g=ctx.createLinearGradient(q[0][0],q[0][1],q[i][0],q[i][1]);g.addColorStop(0,'#48515b');g.addColorStop(1,'#cbd1d8');ctx.strokeStyle=g;ctx.lineWidth=7;ctx.beginPath();ctx.moveTo(q[0][0],q[0][1]);ctx.lineTo(q[i][0],q[i][1]);ctx.stroke();}const atoms=q.map((p,i)=>({p,i})).sort((a,b)=>a.p[2]-b.p[2]);for(const a of atoms){const isC=a.i===0,r=(isC?15:10)*(1+.035*a.p[2]);const grad=ctx.createRadialGradient(a.p[0]-r*.35,a.p[1]-r*.4,r*.1,a.p[0],a.p[1],r);if(isC){grad.addColorStop(0,'#78828d');grad.addColorStop(.45,'#333b44');grad.addColorStop(1,'#11161b');}else{grad.addColorStop(0,'#fff');grad.addColorStop(.65,'#eef1f4');grad.addColorStop(1,'#aeb6bf');}ctx.fillStyle=grad;ctx.strokeStyle=isC?'#080b0e':'#89939d';ctx.lineWidth=1.2;ctx.beginPath();ctx.arc(a.p[0],a.p[1],r,0,Math.PI*2);ctx.fill();ctx.stroke();}}
function attach(el,m){const c=el.querySelector('canvas');draw(c,m);let drag=false,last=null;c.onpointerdown=e=>{drag=true;last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(!drag)return;let r=rotations.get(m.key)||{yaw:-.65,pitch:.48};r={yaw:r.yaw+(e.clientX-last[0])*.012,pitch:Math.max(-1.5,Math.min(1.5,r.pitch+(e.clientY-last[1])*.012))};rotations.set(m.key,r);last=[e.clientX,e.clientY];draw(c,m)};c.onpointerup=()=>drag=false;c.ondblclick=()=>{rotations.delete(m.key);draw(c,m)}}
document.querySelectorAll('.filter').forEach(b=>b.onclick=()=>{document.querySelectorAll('.filter').forEach(x=>x.classList.remove('active'));b.classList.add('active');group=b.dataset.group;render()});
document.getElementById('search').oninput=e=>{query=e.target.value.trim();render()};document.getElementById('sort').onchange=e=>{sort=e.target.value;render()};document.getElementById('sameScale').onchange=e=>{sameScale=e.target.checked;render()};window.onresize=()=>{clearTimeout(window._rt);window._rt=setTimeout(render,150)};render();
</script>
</body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
