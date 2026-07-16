#!/usr/bin/env python3
"""Create a dependency-free interactive 3D HTML viewer for culprit structures."""

import argparse
import json
import random
from pathlib import Path

import numpy as np


ELEMENTS = {1: "H", 6: "C"}
RADII = {"H": 0.16, "C": 0.26}
COLORS = {"H": "#f4f4f4", "C": "#222222"}
EDGE_CUTOFF = 10.3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug_dir",
        type=Path,
        default=Path("examples/TEST_METHANE_MODEL_l4_n3_d100000_seed42-b256_fresh/validation_debug"),
    )
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--source_indices", nargs="+", type=int, default=[32801, 1136, 60963])
    parser.add_argument("--normal_count", type=int, default=6)
    parser.add_argument("--normal_mode", choices=("median", "best", "random"), default="median")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def load_epoch(debug_dir, epoch):
    path = debug_dir / f"validation_errors_epoch_{epoch}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def pair_bonds(numbers, positions):
    """Return the unique undirected edges used by the model's pair graph."""
    bonds = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance < EDGE_CUTOFF:
                bonds.append({"i": i, "j": j, "distance": distance})
    return bonds


def structure_payload(epoch_data, source_index, label=None):
    matches = np.where(epoch_data["source_indices"].astype(int) == int(source_index))[0]
    if len(matches) != 1:
        raise ValueError(f"source_index {source_index} matched {len(matches)} validation rows")
    row = int(matches[0])
    numbers = epoch_data["numbers"][row].astype(int)
    positions = epoch_data["positions"][row].astype(float)
    forces = epoch_data["forces"][row].astype(float)
    force_error = epoch_data["force_error"][row].astype(float)
    atoms = []
    for idx, (number, position) in enumerate(zip(numbers, positions)):
        symbol = ELEMENTS.get(int(number), str(int(number)))
        atoms.append(
            {
                "index": idx,
                "symbol": symbol,
                "position": position.tolist(),
                "radius": RADII.get(symbol, 0.18),
                "color": COLORS.get(symbol, "#999999"),
                "force": forces[idx].tolist(),
                "force_error": force_error[idx].tolist(),
                "force_mag": float(np.linalg.norm(forces[idx])),
                "force_error_mag": float(np.linalg.norm(force_error[idx])),
            }
        )
    return {
        "label": label or f"source {int(source_index)}",
        "source_index": int(source_index),
        "valid_row": row,
        "atoms": atoms,
        "bonds": pair_bonds(numbers, positions),
        "force_rmse": float(np.sqrt(np.mean(force_error**2))),
        "force_mae": float(np.mean(np.abs(force_error))),
        "energy_error": float(epoch_data["energy_error"][row]),
    }


def source_indices_by_normal_mode(epoch_data, excluded, count, mode, random_seed):
    if count <= 0:
        return []
    source_indices = epoch_data["source_indices"].astype(int)
    force_error = epoch_data["force_error"]
    per_config_rmse = np.sqrt(np.mean(force_error**2, axis=(1, 2)))
    candidate_rows = [idx for idx, source in enumerate(source_indices) if int(source) not in excluded]
    if mode == "best":
        chosen_rows = sorted(candidate_rows, key=lambda idx: per_config_rmse[idx])[:count]
    elif mode == "random":
        rng = random.Random(random_seed)
        chosen_rows = candidate_rows[:]
        rng.shuffle(chosen_rows)
        chosen_rows = chosen_rows[:count]
    else:
        median = float(np.median(per_config_rmse[candidate_rows]))
        chosen_rows = sorted(candidate_rows, key=lambda idx: abs(float(per_config_rmse[idx]) - median))[:count]
    return [int(source_indices[row]) for row in chosen_rows]


def html_template(payload):
    data = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Methane Culprit 3D Viewer</title>
<style>
  body {{ margin: 0; font-family: system-ui, sans-serif; background: #f7f7f4; color: #1f2328; }}
  header {{ padding: 14px 18px; border-bottom: 1px solid #d8d8d0; background: white; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  label {{ font-size: 14px; }}
  select, button {{ font: inherit; }}
  canvas {{ display: block; width: 100vw; height: calc(100vh - 62px); cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
  .metric {{ font-size: 13px; color: #4b5563; }}
</style>
</head>
<body>
<header>
  <label>Structure <select id="structure"></select></label>
  <label><input id="showForce" type="checkbox"> target forces</label>
  <label><input id="showError" type="checkbox" checked> force errors</label>
  <button id="reset">reset view</button>
  <span id="metrics" class="metric"></span>
</header>
<canvas id="canvas"></canvas>
<script>
const structures = {data};
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const selector = document.getElementById("structure");
const showForce = document.getElementById("showForce");
const showError = document.getElementById("showError");
const metrics = document.getElementById("metrics");
for (const s of structures) {{
  const opt = document.createElement("option");
  opt.value = s.source_index;
  opt.textContent = `${{s.label}} | source ${{s.source_index}} | F-RMSE ${{s.force_rmse.toFixed(3)}}`;
  selector.appendChild(opt);
}}

let rx = -0.55, ry = 0.65, zoom = 145, dragging = false, lastX = 0, lastY = 0;

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function rotate(p) {{
  let [x, y, z] = p;
  let cy = Math.cos(ry), sy = Math.sin(ry);
  let cx = Math.cos(rx), sx = Math.sin(rx);
  let x1 = cy * x + sy * z;
  let z1 = -sy * x + cy * z;
  let y1 = cx * y - sx * z1;
  let z2 = sx * y + cx * z1;
  return [x1, y1, z2];
}}

function project(p) {{
  const [x, y, z] = rotate(p);
  return [canvas.clientWidth / 2 + x * zoom, canvas.clientHeight / 2 - y * zoom, z];
}}

function currentStructure() {{
  return structures.find(s => String(s.source_index) === selector.value) || structures[0];
}}

function vectorEnd(atom, kind) {{
  const scale = kind === "force_error" ? 0.012 : 0.018;
  return atom.position.map((v, i) => v + atom[kind][i] * scale);
}}

function line(a, b, color, width=2) {{
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
}}

function arrow(atom, kind, color) {{
  const a = project(atom.position);
  const b = project(vectorEnd(atom, kind));
  line(a, b, color, 3);
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len;
  const left = [b[0] - 9 * ux - 5 * uy, b[1] - 9 * uy + 5 * ux];
  const right = [b[0] - 9 * ux + 5 * uy, b[1] - 9 * uy - 5 * ux];
  line(b, left, color, 3);
  line(b, right, color, 3);
}}

function draw() {{
  const s = currentStructure();
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  metrics.textContent = `${{s.label}}  valid row=${{s.valid_row}}  F-RMSE=${{s.force_rmse.toFixed(3)}}  F-MAE=${{s.force_mae.toFixed(3)}}  energy error=${{s.energy_error.toFixed(3)}}`;

  const items = [];
  for (const b of s.bonds) {{
    const a0 = s.atoms[b.i], a1 = s.atoms[b.j];
    const p0 = project(a0.position), p1 = project(a1.position);
    items.push({{z: (p0[2] + p1[2]) / 2, type: "bond", p0, p1, distance: b.distance}});
  }}
  for (const atom of s.atoms) {{
    const p = project(atom.position);
    items.push({{z: p[2], type: "atom", atom, p}});
  }}
  items.sort((a, b) => a.z - b.z);

  for (const item of items) {{
    if (item.type === "bond") {{
      line(item.p0, item.p1, "#777", 7);
      line(item.p0, item.p1, "#bbb", 3);
      const mx = (item.p0[0] + item.p1[0]) / 2, my = (item.p0[1] + item.p1[1]) / 2;
      ctx.fillStyle = "#454545";
      ctx.font = "12px system-ui";
      ctx.fillText(item.distance.toFixed(2), mx + 4, my - 4);
    }} else {{
      const r = item.atom.radius * zoom;
      ctx.fillStyle = item.atom.color;
      ctx.strokeStyle = "black";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(item.p[0], item.p[1], r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = item.atom.symbol === "C" ? "white" : "black";
      ctx.font = "13px system-ui";
      ctx.fillText(`${{item.atom.symbol}}${{item.atom.index}}`, item.p[0] - r * 0.45, item.p[1] + 4);
    }}
  }}
  if (showForce.checked) for (const atom of s.atoms) arrow(atom, "force", "#1f77b4");
  if (showError.checked) for (const atom of s.atoms) arrow(atom, "force_error", "#d62728");

  ctx.fillStyle = "#1f2328";
  ctx.font = "13px system-ui";
  ctx.fillText("Drag to rotate, wheel to zoom. Red = force error, blue = target force. Lines are model pair edges; labels are Angstrom.", 18, canvas.clientHeight - 18);
}}

canvas.addEventListener("mousedown", e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  ry += (e.clientX - lastX) * 0.01;
  rx += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
  draw();
}});
canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  zoom *= e.deltaY < 0 ? 1.08 : 0.92;
  draw();
}}, {{passive: false}});
document.getElementById("reset").onclick = () => {{ rx = -0.55; ry = 0.65; zoom = 145; draw(); }};
selector.onchange = draw;
showForce.onchange = draw;
showError.onchange = draw;
window.onresize = resize;
resize();
</script>
</body>
</html>
"""


def main():
    args = parse_args()
    epoch_data = load_epoch(args.debug_dir, args.epoch)
    culprit_payload = [
        structure_payload(epoch_data, source_index, label=f"culprit {idx + 1}")
        for idx, source_index in enumerate(args.source_indices)
    ]
    normal_sources = source_indices_by_normal_mode(
        epoch_data,
        excluded=set(args.source_indices),
        count=args.normal_count,
        mode=args.normal_mode,
        random_seed=args.random_seed,
    )
    normal_payload = [
        structure_payload(epoch_data, source_index, label=f"{args.normal_mode} normal {idx + 1}")
        for idx, source_index in enumerate(normal_sources)
    ]
    payload = culprit_payload + normal_payload
    out = args.out or args.debug_dir / "culprit_structures" / f"culprits_epoch_{args.epoch}_interactive.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_template(payload))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
