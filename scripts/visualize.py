from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image, ImageDraw

from stage1.utils.config import load_config
from stage1.utils.checkpointing import sha256_file
from stage1.data.dataset_adapter import (
    SafeStage1Dataset,
    Stage1Dataset,
    move_batch_to_device,
    projection_collate,
)
from stage1.data.geometry import build_stage1_targets, make_query_grid
from stage1.models.confidence import (
    Stage1ConfidenceModel,
    build_confidence_depth_inputs,
)
from stage1.models.model import Stage1CorrespondenceModel, Stage1Output


def _to_uint8_rgb(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).astype(np.uint8))


def _draw_cross(draw: ImageDraw.ImageDraw, xy: tuple[float, float], color: tuple[int, int, int], r: int = 3) -> None:
    x, y = xy
    draw.line((x - r, y, x + r, y), fill=color, width=2)
    draw.line((x, y - r, x, y + r), fill=color, width=2)


def save_correspondence_visualization(
    batch,
    output: Stage1Output,
    output_path: str | Path,
    max_points: int = 80,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    front = _to_uint8_rgb(batch.front_rgb[0])
    sat = _to_uint8_rgb(batch.satellite_rgb[0])
    gap = 16
    canvas = Image.new("RGB", (front.width + gap + sat.width, max(front.height, sat.height)), (255, 255, 255))
    canvas.paste(front, (0, 0))
    canvas.paste(sat, (front.width + gap, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")

    valid = output.targets.valid_gt_mask[0].detach().cpu()
    indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if indices.numel() == 0:
        indices = torch.arange(min(output.targets.query_uv.shape[1], max_points))
    if indices.numel() > max_points:
        stride = max(indices.numel() // max_points, 1)
        indices = indices[::stride][:max_points]

    query = output.targets.query_uv[0, indices].detach().cpu()
    gt = output.targets.gt_sat_uv[0, indices].detach().cpu()
    coarse = output.matcher.coarse_uv[0, indices].detach().cpu()
    refined = output.refined_uv[0, indices].detach().cpu()
    certainty = output.certainty[0, indices].detach().cpu()
    sat_x0 = front.width + gap
    for q, g, c, r, p in zip(query, gt, coarse, refined, certainty):
        alpha = int(60 + 180 * float(p.clamp(0, 1)))
        qxy = (float(q[0]), float(q[1]))
        gxy = (sat_x0 + float(g[0]), float(g[1]))
        cxy = (sat_x0 + float(c[0]), float(c[1]))
        rxy = (sat_x0 + float(r[0]), float(r[1]))
        draw.line((qxy[0], qxy[1], rxy[0], rxy[1]), fill=(0, 160, 255, alpha), width=1)
        _draw_cross(draw, gxy, (30, 220, 30, 255), r=3)
        _draw_cross(draw, cxy, (255, 170, 0, 190), r=2)
        _draw_cross(draw, rxy, (255, 30, 30, 255), r=3)
    canvas.save(output_path)


def save_similarity_heatmap(
    batch,
    output: Stage1Output,
    output_path: str | Path,
    query_index: int | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = output.targets.valid_gt_mask[0]
    if query_index is None:
        candidates = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if candidates.numel() == 0:
            return
        query_index = int(candidates[0].item())
    sat = _to_uint8_rgb(batch.satellite_rgb[0]).convert("RGBA")
    logits = output.matcher.ranking_scores[0, query_index].detach()
    h, w = output.satellite_features.shape[-2:]
    heat = logits.reshape(h, w)
    finite = torch.isfinite(heat)
    if finite.any():
        low = heat[finite].min()
        high = heat[finite].max()
        heat = torch.where(finite, heat, low)
        heat = (heat - low) / (high - low).clamp_min(1e-6)
    else:
        heat = torch.zeros_like(heat)
    heat_img = Image.fromarray((heat.cpu().numpy() * 255).astype(np.uint8)).resize(sat.size, Image.BILINEAR)
    rgba = np.zeros((sat.height, sat.width, 4), dtype=np.uint8)
    rgba[..., 0] = np.asarray(heat_img)
    rgba[..., 3] = 120
    overlay = Image.alpha_composite(sat, Image.fromarray(rgba, mode="RGBA"))
    draw = ImageDraw.Draw(overlay, "RGBA")
    gt = output.targets.gt_sat_uv[0, query_index].detach().cpu()
    pred = output.refined_uv[0, query_index].detach().cpu()
    _draw_cross(draw, (float(gt[0]), float(gt[1])), (30, 240, 30, 255), r=5)
    _draw_cross(draw, (float(pred[0]), float(pred[1])), (255, 30, 30, 255), r=5)
    overlay.convert("RGB").save(output_path)


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_interactive_correspondence_visualization(
    batch,
    output: Stage1Output,
    output_path: str | Path,
    sample_index: int = 0,
) -> None:
    """Write a self-contained HTML viewer for one query at a time."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    front = _to_uint8_rgb(batch.front_rgb[sample_index])
    satellite = _to_uint8_rgb(batch.satellite_rgb[sample_index])
    valid = output.targets.valid_gt_mask[sample_index].detach().cpu()
    indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if indices.numel() == 0:
        indices = torch.arange(output.targets.query_uv.shape[1])

    query = output.targets.query_uv[sample_index, indices].detach().cpu()
    gt = output.targets.gt_sat_uv[sample_index, indices].detach().cpu()
    coarse = output.matcher.coarse_uv[sample_index, indices].detach().cpu()
    refined = output.refined_uv[sample_index, indices].detach().cpu()
    certainty = output.certainty[sample_index, indices].detach().cpu()
    matching_score = output.matcher.selected_matching_scores[sample_index, indices].detach().cpu()
    pixel_error = torch.linalg.norm(refined - gt, dim=-1)
    meters_per_pixel = float(output.satellite_meters_per_pixel[sample_index].detach().cpu().item())
    metric_error = pixel_error * meters_per_pixel
    heatmap_height, heatmap_width = output.satellite_features.shape[-2:]
    logits = output.matcher.ranking_scores[sample_index, indices].detach().float().reshape(-1, heatmap_height, heatmap_width)
    finite = torch.isfinite(logits)
    any_finite = finite.any(dim=(-2, -1), keepdim=True)
    heatmap_min = logits.masked_fill(~finite, torch.inf).amin(dim=(-2, -1), keepdim=True)
    heatmap_max = logits.masked_fill(~finite, -torch.inf).amax(dim=(-2, -1), keepdim=True)
    heatmap_min = torch.where(any_finite, heatmap_min, torch.zeros_like(heatmap_min))
    heatmap_max = torch.where(any_finite, heatmap_max, heatmap_min)
    safe_logits = torch.where(finite, logits, heatmap_min)
    normalized = (safe_logits - heatmap_min) / (heatmap_max - heatmap_min).clamp_min(1e-6)
    normalized = torch.where(any_finite, normalized, torch.zeros_like(normalized))
    heatmaps = (normalized * 255.0).round().to(torch.uint8)

    points = []
    for point_index, q, g, c, r, score, cert, px_error, m_error in zip(
        indices.tolist(), query, gt, coarse, refined, matching_score, certainty, pixel_error, metric_error
    ):
        points.append(
            {
                "id": int(point_index),
                "x": float(q[0]),
                "y": float(q[1]),
                "gt_x": float(g[0]),
                "gt_y": float(g[1]),
                "coarse_x": float(c[0]),
                "coarse_y": float(c[1]),
                "pred_x": float(r[0]),
                "pred_y": float(r[1]),
                "matching_score": float(score),
                "certainty": float(cert),
                "pixel_error": float(px_error),
                "metric_error": float(m_error),
            }
        )

    sample_ids = getattr(batch, "sample_ids", [])
    sample_id = str(sample_ids[sample_index]) if sample_index < len(sample_ids) else str(sample_index)
    data = {
        "sampleId": sample_id,
        "scoreMode": output.matcher.score_mode,
        "frontSize": {"width": front.width, "height": front.height},
        "satelliteSize": {"width": satellite.width, "height": satellite.height},
        "frontImage": _image_to_data_url(front),
        "satelliteImage": _image_to_data_url(satellite),
        "metersPerPixel": meters_per_pixel,
        "heatmapSize": {"width": heatmap_width, "height": heatmap_height},
        "heatmaps": base64.b64encode(heatmaps.cpu().numpy().tobytes()).decode("ascii"),
        "points": points,
    }
    # Prevent an unusual sample id or image payload from closing the script tag.
    data_json = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage-1 Correspondence {sample_id}</title>
  <style>
    :root {{ --bg:#f5f7fb; --panel:#ffffff; --ink:#172033; --muted:#667085; --line:#d0d5dd; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; padding:20px; background:var(--bg); color:var(--ink); font-family:Arial, sans-serif; }}
    .page {{ max-width:1500px; margin:auto; display:grid; gap:16px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
    .toolbar {{ display:grid; grid-template-columns:minmax(240px, 1fr) auto auto; gap:12px; align-items:end; }}
    label {{ display:block; color:var(--muted); font-size:13px; margin-bottom:6px; }}
    select, button {{ height:36px; border:1px solid var(--line); border-radius:7px; background:#fff; padding:0 10px; color:var(--ink); }}
    button {{ cursor:pointer; }}
    button:hover {{ background:#f2f4f7; }}
    .viewer {{ display:grid; grid-template-columns:minmax(0, 1fr) minmax(0, 1fr); gap:16px; }}
    .panel-title {{ color:var(--muted); font-size:13px; margin:0 0 8px; }}
    .canvas {{ position:relative; overflow:hidden; border:1px solid var(--line); border-radius:8px; background:#fff; }}
    .canvas img {{ display:block; width:100%; height:auto; }}
    canvas.heatmap-overlay {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
    svg.overlay {{ position:absolute; inset:0; width:100%; height:100%; }}
    .combined {{ grid-column:1 / -1; }}
    .combined-images {{ display:grid; grid-template-columns:{front.width}fr {satellite.width}fr; }}
    .combined-images img {{ width:100%; display:block; }}
    .stats {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:8px; color:var(--muted); font-size:13px; }}
    .stat {{ background:#f8fafc; border-radius:7px; padding:8px; }}
    .stat strong {{ display:block; color:var(--ink); font-size:14px; margin-top:3px; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); font-size:13px; margin-top:10px; }}
    .legend span::before {{ content:""; display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }}
    .front::before {{ background:#d84315; }} .gt::before {{ background:#20c997; }}
    .coarse::before {{ background:#f59f00; }} .pred::before {{ background:#fa2b2b; }}
    .hint {{ color:var(--muted); font-size:12px; margin-top:10px; }}
    @media (max-width: 900px) {{ .viewer {{ grid-template-columns:1fr; }} .combined {{ grid-column:auto; }} .toolbar {{ grid-template-columns:1fr 1fr; }} .toolbar > :first-child {{ grid-column:1 / -1; }} .stats {{ grid-template-columns:repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h2 style="margin:0 0 4px">Stage-1 Correspondence</h2>
      <div style="color:var(--muted); font-size:13px">sample {sample_id} · valid queries {len(points)}</div>
    </div>
    <div class="card toolbar">
      <div>
        <label for="pointSelect">Select front query</label>
        <select id="pointSelect"></select>
      </div>
      <button id="prevButton" type="button">Previous</button>
      <button id="nextButton" type="button">Next</button>
    </div>
    <div class="card viewer">
      <section>
        <p class="panel-title">Front view — click a query point</p>
        <div class="canvas">
          <img src="{data["frontImage"]}" alt="front view">
          <svg class="overlay" id="frontSvg" viewBox="0 0 {front.width} {front.height}" preserveAspectRatio="none"></svg>
        </div>
      </section>
      <section>
        <p class="panel-title">Satellite similarity heatmap — selected query</p>
        <div class="canvas">
          <img src="{data["satelliteImage"]}" alt="satellite view">
          <canvas class="heatmap-overlay" id="satelliteHeatmapCanvas"></canvas>
          <svg class="overlay" id="satelliteSvg" viewBox="0 0 {satellite.width} {satellite.height}" preserveAspectRatio="none"></svg>
        </div>
      </section>
      <section class="combined">
        <p class="panel-title">Selected correspondence</p>
        <div class="canvas">
          <div class="combined-images">
            <img src="{data["frontImage"]}" alt="front view">
            <img src="{data["satelliteImage"]}" alt="satellite view">
          </div>
          <svg class="overlay" id="combinedSvg" viewBox="0 0 {front.width + satellite.width} {max(front.height, satellite.height)}" preserveAspectRatio="none"></svg>
        </div>
      </section>
    </div>
    <div class="card">
      <div class="stats" id="stats"></div>
      <div class="legend"><span class="front">front query</span><span class="gt">GT</span><span class="coarse">coarse</span><span class="pred">prediction</span></div>
      <div class="hint">Only the selected query's satellite GT/coarse/prediction are displayed. In coarse stage, coarse and prediction overlap.</div>
    </div>
  </div>
  <script>
    const data = {data_json};
    const frontSvg = document.getElementById("frontSvg");
    const satelliteSvg = document.getElementById("satelliteSvg");
    const satelliteHeatmapCanvas = document.getElementById("satelliteHeatmapCanvas");
    const combinedSvg = document.getElementById("combinedSvg");
    const pointSelect = document.getElementById("pointSelect");
    const stats = document.getElementById("stats");
    const frontOffset = data.frontSize.width;
    const heatmaps = decodeBase64(data.heatmaps);
    let selectedPosition = data.points.length ? 0 : -1;

    function decodeBase64(value) {{
      const binary = atob(value);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
      return bytes;
    }}

    function svgElement(name, attributes) {{
      const element = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
      return element;
    }}
    function clear(svg) {{ while (svg.firstChild) svg.removeChild(svg.firstChild); }}
    function circle(svg, x, y, radius, fill, stroke, width, onClick) {{
      const element = svgElement("circle", {{cx:x, cy:y, r:radius, fill, stroke, "stroke-width":width}});
      if (onClick) element.addEventListener("click", (event) => {{ event.stopPropagation(); onClick(); }});
      svg.appendChild(element);
    }}
    function cross(svg, x, y, color, radius, width) {{
      svg.appendChild(svgElement("path", {{d:`M ${{x-radius}} ${{y}} L ${{x+radius}} ${{y}} M ${{x}} ${{y-radius}} L ${{x}} ${{y+radius}}`, stroke:color, "stroke-width":width, "stroke-linecap":"round"}}));
    }}
    function line(svg, x1, y1, x2, y2, color, width, dash) {{
      const attributes = {{x1, y1, x2, y2, stroke:color, "stroke-width":width, "stroke-linecap":"round"}};
      if (dash) attributes["stroke-dasharray"] = dash;
      svg.appendChild(svgElement("line", attributes));
    }}
    function text(svg, x, y, value, color) {{
      const element = svgElement("text", {{x, y, fill:color, "font-size":12, "font-weight":"bold"}});
      element.textContent = value;
      svg.appendChild(element);
    }}
    function selectPosition(position) {{
      if (!data.points.length) return;
      selectedPosition = (position + data.points.length) % data.points.length;
      pointSelect.value = String(selectedPosition);
      render();
    }}
    function renderHeatmap(position) {{
      const width = data.heatmapSize.width;
      const height = data.heatmapSize.height;
      satelliteHeatmapCanvas.width = data.satelliteSize.width;
      satelliteHeatmapCanvas.height = data.satelliteSize.height;
      const context = satelliteHeatmapCanvas.getContext("2d");
      context.clearRect(0, 0, satelliteHeatmapCanvas.width, satelliteHeatmapCanvas.height);
      if (position < 0) return;
      const lowResolutionCanvas = document.createElement("canvas");
      lowResolutionCanvas.width = width;
      lowResolutionCanvas.height = height;
      const lowResolutionContext = lowResolutionCanvas.getContext("2d");
      const imageData = lowResolutionContext.createImageData(width, height);
      const offset = position * width * height;
      for (let index = 0; index < width * height; index += 1) {{
        const value = heatmaps[offset + index];
        imageData.data[index * 4] = value;
        imageData.data[index * 4 + 1] = 0;
        imageData.data[index * 4 + 2] = 0;
        imageData.data[index * 4 + 3] = 120;
      }}
      lowResolutionContext.putImageData(imageData, 0, 0);
      context.imageSmoothingEnabled = true;
      context.drawImage(lowResolutionCanvas, 0, 0, satelliteHeatmapCanvas.width, satelliteHeatmapCanvas.height);
    }}
    function render() {{
      clear(frontSvg); clear(satelliteSvg); clear(combinedSvg);
      if (selectedPosition < 0) {{ stats.innerHTML = "No valid query points"; return; }}
      const selected = data.points[selectedPosition];
      renderHeatmap(selectedPosition);

      for (const [position, point] of data.points.entries()) {{
        const active = position === selectedPosition;
        circle(frontSvg, point.x, point.y, active ? 6 : 3, active ? "#d84315" : "rgba(216,67,21,0.30)", "#fff", 1, () => selectPosition(position));
      }}
      text(frontSvg, selected.x + 8, selected.y - 8, `q${{selected.id}}`, "#172033");

      line(satelliteSvg, selected.gt_x, selected.gt_y, selected.pred_x, selected.pred_y, "rgba(52,152,219,0.9)", 2, "5 4");
      cross(satelliteSvg, selected.gt_x, selected.gt_y, "#20c997", 8, 3);
      cross(satelliteSvg, selected.coarse_x, selected.coarse_y, "#f59f00", 6, 2);
      cross(satelliteSvg, selected.pred_x, selected.pred_y, "#fa2b2b", 8, 3);
      text(satelliteSvg, selected.gt_x + 9, selected.gt_y - 9, "GT", "#087f5b");
      text(satelliteSvg, selected.pred_x + 9, selected.pred_y + 15, "Pred", "#c92a2a");

      line(combinedSvg, selected.x, selected.y, frontOffset + selected.pred_x, selected.pred_y, "rgba(52,152,219,0.9)", 2);
      circle(combinedSvg, selected.x, selected.y, 6, "#d84315", "#fff", 1, null);
      cross(combinedSvg, frontOffset + selected.gt_x, selected.gt_y, "#20c997", 8, 3);
      cross(combinedSvg, frontOffset + selected.coarse_x, selected.coarse_y, "#f59f00", 6, 2);
      cross(combinedSvg, frontOffset + selected.pred_x, selected.pred_y, "#fa2b2b", 8, 3);
      text(combinedSvg, selected.x + 8, selected.y - 8, `q${{selected.id}}`, "#172033");

      stats.innerHTML = `
        <div class="stat">query<strong>#${{selected.id}} (${{selected.x.toFixed(1)}}, ${{selected.y.toFixed(1)}})</strong></div>
        <div class="stat">GT / prediction<strong>(${{selected.gt_x.toFixed(1)}}, ${{selected.gt_y.toFixed(1)}}) → (${{selected.pred_x.toFixed(1)}}, ${{selected.pred_y.toFixed(1)}})</strong></div>
        <div class="stat">pixel error<strong>${{selected.pixel_error.toFixed(2)}} px</strong></div>
        <div class="stat">metric / certainty<strong>${{selected.metric_error.toFixed(2)}} m / ${{selected.certainty.toFixed(3)}}</strong></div>
        <div class="stat">matching score<strong>${{selected.matching_score.toExponential(3)}} (${{data.scoreMode}})</strong></div>`;
    }}
    for (const [position, point] of data.points.entries()) {{
      const option = document.createElement("option");
      option.value = String(position);
      option.textContent = `query #${{point.id}} @ (${{point.x.toFixed(1)}}, ${{point.y.toFixed(1)}})`;
      pointSelect.appendChild(option);
    }}
    pointSelect.addEventListener("change", (event) => selectPosition(Number(event.target.value)));
    document.getElementById("prevButton").addEventListener("click", () => selectPosition(selectedPosition - 1));
    document.getElementById("nextButton").addEventListener("click", () => selectPosition(selectedPosition + 1));
    frontSvg.addEventListener("click", (event) => {{
      if (!data.points.length) return;
      const rect = frontSvg.getBoundingClientRect();
      const x = (event.clientX - rect.left) * data.frontSize.width / rect.width;
      const y = (event.clientY - rect.top) * data.frontSize.height / rect.height;
      let best = 0; let bestDistance = Infinity;
      for (const [position, point] of data.points.entries()) {{
        const distance = (point.x - x) ** 2 + (point.y - y) ** 2;
        if (distance < bestDistance) {{ bestDistance = distance; best = position; }}
      }}
      selectPosition(best);
    }});
    render();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _load_confidence_model(
    config: dict,
    base_checkpoint: Path,
    confidence_checkpoint: Path,
    device: str,
) -> Stage1ConfidenceModel:
    """Load the frozen Stage-1 model and the separately trained confidence head."""

    model = Stage1ConfidenceModel(config).to(device)
    base_payload = torch.load(base_checkpoint, map_location=device, weights_only=False)
    base_state = base_payload.get("model", base_payload)
    missing, unexpected = model.stage1.load_compatible_checkpoint(base_state)
    if missing or unexpected:
        print(
            f"base checkpoint loaded partially: missing={len(missing)}, "
            f"unexpected={len(unexpected)}",
            flush=True,
        )

    confidence_payload = torch.load(confidence_checkpoint, map_location=device, weights_only=False)
    saved_base_hash = confidence_payload.get("base_checkpoint_sha256")
    if saved_base_hash is not None:
        actual_base_hash = sha256_file(base_checkpoint)
        if saved_base_hash != actual_base_hash:
            raise ValueError(
                "The confidence checkpoint was trained with a different Stage-1 base checkpoint: "
                f"saved SHA-256={saved_base_hash}, requested SHA-256={actual_base_hash}."
            )
    confidence_state = confidence_payload.get("confidence_head")
    if confidence_state is None:
        model_state = confidence_payload.get("model", {})
        confidence_state = {
            key.removeprefix("confidence_head."): value
            for key, value in model_state.items()
            if key.startswith("confidence_head.")
        }
    if not confidence_state:
        raise ValueError(
            f"Could not find confidence_head weights in {confidence_checkpoint}"
        )
    model.confidence_head.load_state_dict(confidence_state, strict=True)
    model.eval()
    return model


def save_interactive_confidence_visualization(
    batch,
    output,
    depth_inputs,
    output_path: str | Path,
    sample_index: int = 0,
) -> None:
    """Write a self-contained HTML heatmap for confidence-weighted anchors.

    The same color represents one anchor's normalized scale weight on the front
    query and on its predicted satellite match.  Anchor selection follows the
    confidence-stage geometric validity mask and does not use GT correspondences.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    front = _to_uint8_rgb(batch.front_rgb[sample_index])
    satellite = _to_uint8_rgb(batch.satellite_rgb[sample_index])
    query = output.stage1.targets.query_uv[sample_index].detach().cpu().float()
    refined = output.stage1.refined_uv[sample_index].detach().cpu().float()
    confidence = output.confidence[sample_index].detach().cpu().float()
    anchor_z = depth_inputs.anchor_z[sample_index].detach().cpu().float()
    mono_anchor_z = depth_inputs.mono_anchor_z[sample_index].detach().cpu().float()
    geometric_valid = depth_inputs.anchor_valid[sample_index].detach().cpu().bool()

    finite_coordinates = torch.isfinite(query).all(dim=-1) & torch.isfinite(refined).all(dim=-1)
    weight_valid = geometric_valid & torch.isfinite(confidence) & finite_coordinates
    indices = torch.nonzero(weight_valid, as_tuple=False).squeeze(1)
    if indices.numel() > 0:
        confidence_values = confidence[indices].clamp_min(1e-6)
        weights = confidence_values / confidence_values.sum().clamp_min(1e-12)
        relative_weights = weights / weights.max().clamp_min(1e-12)
    else:
        confidence_values = torch.empty(0, dtype=torch.float32)
        weights = torch.empty(0, dtype=torch.float32)
        relative_weights = torch.empty(0, dtype=torch.float32)

    points = []
    for point_index, c_value, weight, relative in zip(
        indices.tolist(), confidence_values, weights, relative_weights
    ):
        q = query[point_index]
        r = refined[point_index]
        z_value = anchor_z[point_index]
        mono_value = mono_anchor_z[point_index]
        ratio = z_value / mono_value.clamp_min(1e-6)
        sat_in_bounds = (
            0.0 <= float(r[0]) < satellite.width and 0.0 <= float(r[1]) < satellite.height
        )
        points.append(
            {
                "id": int(point_index),
                "x": float(q[0]),
                "y": float(q[1]),
                "pred_x": float(r[0]),
                "pred_y": float(r[1]),
                "confidence": float(c_value),
                "weight": float(weight),
                "relative_weight": float(relative),
                "anchor_z": float(z_value),
                "mono_anchor_z": float(mono_value),
                "ratio": float(ratio),
                "sat_in_bounds": bool(sat_in_bounds),
            }
        )

    sample_ids = getattr(batch, "sample_ids", [])
    sample_id = str(sample_ids[sample_index]) if sample_index < len(sample_ids) else str(sample_index)
    data = {
        "sampleId": sample_id,
        "frontSize": {"width": front.width, "height": front.height},
        "satelliteSize": {"width": satellite.width, "height": satellite.height},
        "frontImage": _image_to_data_url(front),
        "satelliteImage": _image_to_data_url(satellite),
        "points": points,
        "numQueries": int(query.shape[0]),
        "validAnchorCount": int(geometric_valid.sum().item()),
        "displayedAnchorCount": len(points),
        "invalidAnchorCount": int(query.shape[0] - geometric_valid.sum().item()),
    }
    data_json = json.dumps(data, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Confidence weights __SAMPLE_ID__</title>
  <style>
    :root { --bg:#f5f7fb; --panel:#ffffff; --ink:#172033; --muted:#667085; --line:#d0d5dd; }
    * { box-sizing:border-box; }
    body { margin:0; padding:20px; background:var(--bg); color:var(--ink); font-family:Arial, sans-serif; }
    .page { max-width:1600px; margin:auto; display:grid; gap:16px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }
    .toolbar { display:grid; grid-template-columns:minmax(240px, 1fr) auto auto; gap:12px; align-items:end; }
    label { display:block; color:var(--muted); font-size:13px; margin-bottom:6px; }
    select, button { height:36px; border:1px solid var(--line); border-radius:7px; background:#fff; padding:0 10px; color:var(--ink); }
    button { cursor:pointer; }
    button:hover { background:#f2f4f7; }
    .viewer { display:grid; grid-template-columns:minmax(0, 1fr) minmax(0, 1fr); gap:16px; }
    .panel-title { color:var(--muted); font-size:13px; margin:0 0 8px; }
    .canvas { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:8px; background:#fff; }
    .canvas img { display:block; width:100%; height:auto; }
    svg.overlay { position:absolute; inset:0; width:100%; height:100%; }
    .combined { grid-column:1 / -1; }
    .combined-images { display:grid; grid-template-columns:var(--front-ratio) var(--sat-ratio); }
    .combined-images img { width:100%; display:block; }
    .stats { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:8px; color:var(--muted); font-size:13px; }
    .stat { background:#f8fafc; border-radius:7px; padding:8px; }
    .stat strong { display:block; color:var(--ink); font-size:14px; margin-top:3px; }
    .legend { display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); font-size:13px; margin-top:12px; align-items:center; }
    .gradient { width:240px; height:12px; border-radius:6px; background:linear-gradient(90deg, hsl(240,90%,50%), hsl(180,90%,50%), hsl(120,90%,50%), hsl(60,90%,50%), hsl(0,90%,50%)); border:1px solid #98a2b3; }
    .hint { color:var(--muted); font-size:12px; margin-top:10px; line-height:1.5; }
    @media (max-width: 900px) { .viewer { grid-template-columns:1fr; } .combined { grid-column:auto; } .toolbar { grid-template-columns:1fr 1fr; } .toolbar > :first-child { grid-column:1 / -1; } .stats { grid-template-columns:repeat(2, 1fr); } }
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h2 style="margin:0 0 4px">Confidence-weighted scale anchors</h2>
      <div style="color:var(--muted); font-size:13px">sample __SAMPLE_ID__ · valid anchors <span id="validCount"></span></div>
    </div>
    <div class="card toolbar">
      <div>
        <label for="pointSelect">Select anchor</label>
        <select id="pointSelect"></select>
      </div>
      <button id="prevButton" type="button">Previous</button>
      <button id="nextButton" type="button">Next</button>
    </div>
    <div class="card viewer">
      <section>
        <p class="panel-title">Front view — confidence weight heatmap</p>
        <div class="canvas">
          <img id="frontImage" alt="front view">
          <svg class="overlay" id="frontSvg" preserveAspectRatio="none"></svg>
        </div>
      </section>
      <section>
        <p class="panel-title">Satellite view — corresponding predicted matches</p>
        <div class="canvas">
          <img id="satelliteImage" alt="satellite view">
          <svg class="overlay" id="satelliteSvg" preserveAspectRatio="none"></svg>
        </div>
      </section>
      <section class="combined">
        <p class="panel-title">Front anchors ↔ satellite matches (same color = same normalized weight)</p>
        <div class="canvas">
          <div class="combined-images">
            <img id="combinedFrontImage" alt="front view">
            <img id="combinedSatelliteImage" alt="satellite view">
          </div>
          <svg class="overlay" id="combinedSvg" preserveAspectRatio="none"></svg>
        </div>
      </section>
    </div>
    <div class="card">
      <div class="stats" id="stats"></div>
      <div class="legend"><span>low relative weight</span><div class="gradient"></div><span>high relative weight</span></div>
      <div class="hint">Color and marker size encode relative normalized weight wᵢ / max(w). The exact cᵢ and normalized wᵢ are shown for the selected anchor. Only the fixed geometric-validity rule is used; GT is not used to select anchors or compute the displayed weights. Out-of-bounds satellite predictions are still displayed and marked dashed.</div>
    </div>
  </div>
  <script>
    const data = __DATA__;
    const frontImage = document.getElementById("frontImage");
    const satelliteImage = document.getElementById("satelliteImage");
    const combinedFrontImage = document.getElementById("combinedFrontImage");
    const combinedSatelliteImage = document.getElementById("combinedSatelliteImage");
    const frontSvg = document.getElementById("frontSvg");
    const satelliteSvg = document.getElementById("satelliteSvg");
    const combinedSvg = document.getElementById("combinedSvg");
    const pointSelect = document.getElementById("pointSelect");
    const stats = document.getElementById("stats");
    document.getElementById("validCount").textContent = `${data.validAnchorCount} / ${data.numQueries} (displayed ${data.displayedAnchorCount})`;
    for (const image of [frontImage, combinedFrontImage]) image.src = data.frontImage;
    for (const image of [satelliteImage, combinedSatelliteImage]) image.src = data.satelliteImage;
    for (const [svg, size] of [[frontSvg, data.frontSize], [satelliteSvg, data.satelliteSize]]) {
      svg.setAttribute("viewBox", `0 0 ${size.width} ${size.height}`);
    }
    combinedSvg.setAttribute("viewBox", `0 0 ${data.frontSize.width + data.satelliteSize.width} ${Math.max(data.frontSize.height, data.satelliteSize.height)}`);
    document.querySelector(".combined-images").style.setProperty("--front-ratio", `${data.frontSize.width}fr`);
    document.querySelector(".combined-images").style.setProperty("--sat-ratio", `${data.satelliteSize.width}fr`);
    let selectedPosition = data.points.length ? 0 : -1;

    function svgElement(name, attributes) {
      const element = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
      return element;
    }
    function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
    function weightColor(value) {
      const bounded = Math.max(0, Math.min(1, value));
      return `hsl(${240 * (1 - bounded)}, 90%, 50%)`;
    }
    function markerRadius(point, active) { return (active ? 6 : 2.5) + 4 * Math.sqrt(Math.max(0, point.relative_weight)); }
    function circle(svg, x, y, radius, fill, stroke, width, onClick) {
      const element = svgElement("circle", {cx:x, cy:y, r:radius, fill, stroke, "stroke-width":width});
      if (onClick) element.addEventListener("click", (event) => { event.stopPropagation(); onClick(); });
      svg.appendChild(element);
    }
    function line(svg, x1, y1, x2, y2, color, width, dash, opacity) {
      const attributes = {x1, y1, x2, y2, stroke:color, "stroke-width":width, "stroke-linecap":"round", opacity};
      if (dash) attributes["stroke-dasharray"] = dash;
      svg.appendChild(svgElement("line", attributes));
    }
    function text(svg, x, y, value, color) {
      const element = svgElement("text", {x, y, fill:color, "font-size":12, "font-weight":"bold", "paint-order":"stroke", stroke:"white", "stroke-width":3});
      element.textContent = value;
      svg.appendChild(element);
    }
    function selectPosition(position) {
      if (!data.points.length) return;
      selectedPosition = (position + data.points.length) % data.points.length;
      pointSelect.value = String(selectedPosition);
      render();
    }
    function render() {
      clear(frontSvg); clear(satelliteSvg); clear(combinedSvg);
      if (selectedPosition < 0) {
        stats.innerHTML = '<div class="stat">No valid anchors<strong>Nothing to display</strong></div>';
        return;
      }
      const selected = data.points[selectedPosition];
      for (const [position, point] of data.points.entries()) {
        const active = position === selectedPosition;
        const color = weightColor(point.relative_weight);
        const radius = markerRadius(point, active);
        const opacity = active ? 1 : 0.70;
        circle(frontSvg, point.x, point.y, radius, color, active ? "#172033" : "white", active ? 2 : 0.7, () => selectPosition(position));
        circle(satelliteSvg, point.pred_x, point.pred_y, radius, color, active ? "#172033" : "white", active ? 2 : 0.7, () => selectPosition(position));
        line(combinedSvg, point.x, point.y, data.frontSize.width + point.pred_x, point.pred_y, color, active ? 3 : 1, point.sat_in_bounds ? "" : "6 4", opacity * (active ? 1 : 0.25));
        circle(combinedSvg, point.x, point.y, radius, color, active ? "#172033" : "white", active ? 2 : 0.7, () => selectPosition(position));
        circle(combinedSvg, data.frontSize.width + point.pred_x, point.pred_y, radius, color, active ? "#172033" : "white", active ? 2 : 0.7, () => selectPosition(position));
      }
      text(frontSvg, selected.x + 8, selected.y - 8, `anchor #${selected.id}`, "#172033");
      text(satelliteSvg, selected.pred_x + 8, selected.pred_y - 8, `match #${selected.id}`, "#172033");
      text(combinedSvg, selected.x + 8, selected.y - 8, `anchor #${selected.id}`, "#172033");
      stats.innerHTML = `
        <div class="stat">anchor<strong>#${selected.id} (${selected.x.toFixed(1)}, ${selected.y.toFixed(1)})</strong></div>
        <div class="stat">confidence cᵢ<strong>${selected.confidence.toFixed(5)}</strong></div>
        <div class="stat">normalized weight wᵢ<strong>${selected.weight.toExponential(4)}</strong></div>
        <div class="stat">relative weight<strong>${selected.relative_weight.toFixed(3)}</strong></div>
        <div class="stat">camera z / mono z<strong>${selected.anchor_z.toFixed(3)} / ${selected.mono_anchor_z.toFixed(3)} m</strong></div>
        <div class="stat">z ratio<strong>${selected.ratio.toFixed(4)}</strong></div>
        <div class="stat">predicted satellite<strong>(${selected.pred_x.toFixed(1)}, ${selected.pred_y.toFixed(1)})</strong></div>
        <div class="stat">satellite bounds<strong>${selected.sat_in_bounds ? "inside" : "out of bounds"}</strong></div>`;
    }
    for (const [position, point] of data.points.entries()) {
      const option = document.createElement("option");
      option.value = String(position);
      option.textContent = `anchor #${point.id} · w=${point.weight.toExponential(3)} · c=${point.confidence.toFixed(3)}`;
      pointSelect.appendChild(option);
    }
    pointSelect.addEventListener("change", (event) => selectPosition(Number(event.target.value)));
    document.getElementById("prevButton").addEventListener("click", () => selectPosition(selectedPosition - 1));
    document.getElementById("nextButton").addEventListener("click", () => selectPosition(selectedPosition + 1));
    frontSvg.addEventListener("click", (event) => {
      if (!data.points.length) return;
      const rect = frontSvg.getBoundingClientRect();
      const x = (event.clientX - rect.left) * data.frontSize.width / rect.width;
      const y = (event.clientY - rect.top) * data.frontSize.height / rect.height;
      let best = 0; let bestDistance = Infinity;
      for (const [position, point] of data.points.entries()) {
        const distance = (point.x - x) ** 2 + (point.y - y) ** 2;
        if (distance < bestDistance) { bestDistance = distance; best = position; }
      }
      selectPosition(best);
    });
    render();
  </script>
</body>
</html>
""".replace("__DATA__", data_json).replace("__SAMPLE_ID__", sample_id)
    output_path.write_text(html, encoding="utf-8")


def save_gt_geometry_visualization(batch, config: dict, output_path: str | Path, max_points: int = 120) -> None:
    front_image_hw = (batch.front_rgb.shape[-2], batch.front_rgb.shape[-1])
    # Use a coarse display grid for sanity visualization only.
    feat_hw = (max(1, front_image_hw[0] // int(config["geometry"].get("sanity_query_stride", 16))), max(1, front_image_hw[1] // int(config["geometry"].get("sanity_query_stride", 16))))
    query_uv = make_query_grid(batch.front_rgb.shape[0], feat_hw, front_image_hw, batch.front_rgb.device, batch.front_rgb.dtype)
    targets = build_stage1_targets(
        lidar_depth=batch.lidar_depth,
        query_uv=query_uv,
        intrinsics=batch.intrinsics,
        camera_to_world=batch.camera_to_world,
        satellite_meters_per_pixel=batch.satellite_meters_per_pixel,
        satellite_center_world_xy=batch.satellite_center_world_xy,
        satellite_image_hw=(batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1]),
        explicit_negative_mask=batch.explicit_negative_mask,
        lidar_query_radius_px=int(config["geometry"].get("lidar_query_radius_px", 4)),
    )
    dummy = type("DummyOutput", (), {})()
    dummy.targets = targets
    dummy.matcher = type("DummyMatcher", (), {})()
    dummy.matcher.coarse_uv = targets.gt_sat_uv
    dummy.refined_uv = targets.gt_sat_uv
    dummy.certainty = targets.valid_gt_mask.float()
    save_correspondence_visualization(batch, dummy, output_path, max_points=max_points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("models/stage1/configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--base_checkpoint",
        type=Path,
        default=None,
        help="Frozen coarse+refine checkpoint for --mode confidence.",
    )
    parser.add_argument(
        "--confidence_checkpoint",
        type=Path,
        default=None,
        help="Confidence-head checkpoint for --mode confidence.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest to visualize; overrides the manifest path in the config.",
    )
    parser.add_argument(
        "--sample_id",
        default=None,
        help="Sample ID to visualize; defaults to --sample_index.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Zero-based manifest index to visualize when --sample_id is not set.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/vis"))
    parser.add_argument("--mode", choices=("gt", "pred", "confidence"), default="gt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = (
        args.manifest
        or config["data"].get("test_manifest")
        or config["data"].get("val_manifest")
        or config["data"]["train_manifest"]
    )
    dataset = Stage1Dataset(manifest)
    if args.sample_id is not None:
        matching_indices = [
            index
            for index, record in enumerate(dataset.records)
            if (record.sample_id or record.front_rgb_path.stem) == args.sample_id
        ]
        if not matching_indices:
            raise ValueError(f"sample_id {args.sample_id!r} was not found in {manifest}")
        sample_index = matching_indices[0]
    else:
        sample_index = args.sample_index
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {sample_index} is outside manifest range [0, {len(dataset) - 1}]"
        )
    selected = SafeStage1Dataset(dataset)[sample_index]
    batch, load_errors = projection_collate([selected])
    if batch is None:
        error = load_errors[0] if load_errors else None
        detail = f": {error.error_type}: {error.error_message}" if error else ""
        raise RuntimeError(f"Could not load sample index {sample_index}{detail}")
    batch = move_batch_to_device(batch, args.device)
    print(
        f"visualizing sample_id={batch.sample_ids[0]} index={sample_index} manifest={manifest}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "gt":
        save_gt_geometry_visualization(batch, config, args.output_dir / "gt_geometry.png")
        return
    if args.mode == "confidence":
        if args.base_checkpoint is None or args.confidence_checkpoint is None:
            raise ValueError(
                "--mode confidence requires both --base_checkpoint and "
                "--confidence_checkpoint"
            )
        model = _load_confidence_model(
            config,
            args.base_checkpoint,
            args.confidence_checkpoint,
            args.device,
        )
        with torch.no_grad():
            output = model(batch)
        depth_inputs = build_confidence_depth_inputs(output, batch, config)
        save_interactive_confidence_visualization(
            batch,
            output,
            depth_inputs,
            args.output_dir / "confidence_weights_interactive.html",
            sample_index=0,
        )
        print(
            "saved confidence heatmap: "
            f"{args.output_dir / 'confidence_weights_interactive.html'} "
            f"(valid_anchors={int(depth_inputs.anchor_valid[0].sum().item())})",
            flush=True,
        )
        return
    model = Stage1CorrespondenceModel(config).to(args.device)
    model.set_training_stage(str(config.get("training", {}).get("stage", "joint")))
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_compatible_checkpoint(payload["model"])
    model.eval()
    with torch.no_grad():
        output = model(batch)
    save_correspondence_visualization(batch, output, args.output_dir / "correspondences.png")
    save_similarity_heatmap(batch, output, args.output_dir / "similarity_heatmap.png")
    save_interactive_correspondence_visualization(batch, output, args.output_dir / "correspondences_interactive.html")


if __name__ == "__main__":
    main()
