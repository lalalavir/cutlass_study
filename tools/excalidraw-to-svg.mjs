#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const DEFAULTS = {
  strokeColor: "#1e1e1e",
  backgroundColor: "transparent",
  strokeWidth: 2,
  opacity: 100,
  fontSize: 20,
};

function printUsageAndExit() {
  console.error(
    "Usage: node tools/excalidraw-to-svg.mjs <input.excalidraw|elements.json> [output.svg]"
  );
  process.exit(1);
}

function readJson(filePath) {
  const raw = fs.readFileSync(filePath, "utf8");
  return JSON.parse(raw);
}

function ensureElements(data) {
  if (Array.isArray(data)) {
    return data;
  }
  if (data && Array.isArray(data.elements)) {
    return data.elements;
  }
  throw new Error("Unsupported input JSON format.");
}

function escXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function colorOrNone(color) {
  if (!color || color === "transparent" || color === "none") return "none";
  return color;
}

function lineMetrics(fontSize) {
  const fs = Number(fontSize || DEFAULTS.fontSize);
  return {
    fontSize: fs,
    lineHeight: fs * 1.25,
  };
}

function textBlockMetrics(text, fontSize) {
  const lines = String(text ?? "").split("\n");
  const { lineHeight } = lineMetrics(fontSize);
  const maxChars = lines.reduce((m, line) => Math.max(m, line.length), 0);
  const width = maxChars * Number(fontSize || DEFAULTS.fontSize) * 0.56;
  const height = lines.length * lineHeight;
  return { lines, width, height, lineHeight };
}

function elementOpacity(el) {
  const o = Number(el.opacity ?? DEFAULTS.opacity);
  return Math.max(0, Math.min(1, o / 100));
}

function absoluteArrowPoints(el) {
  if (Array.isArray(el.points) && el.points.length > 0) {
    return el.points.map(([dx, dy]) => [Number(el.x) + dx, Number(el.y) + dy]);
  }
  return [
    [Number(el.x), Number(el.y)],
    [Number(el.x) + Number(el.width || 0), Number(el.y) + Number(el.height || 0)],
  ];
}

function shapeLabelBounds(el) {
  if (!el.label || !el.label.text) return null;
  const { width, height } = textBlockMetrics(
    el.label.text,
    el.label.fontSize || 16
  );
  const cx = Number(el.x) + Number(el.width) / 2;
  const cy = Number(el.y) + Number(el.height) / 2;
  return {
    minX: cx - width / 2,
    minY: cy - height / 2,
    maxX: cx + width / 2,
    maxY: cy + height / 2,
  };
}

function elementBounds(el) {
  const t = el.type;
  if (
    t === "cameraUpdate" ||
    t === "delete" ||
    t === "restoreCheckpoint" ||
    t === "selection"
  ) {
    return null;
  }

  if (t === "text") {
    const m = textBlockMetrics(el.text || "", el.fontSize || DEFAULTS.fontSize);
    return {
      minX: Number(el.x),
      minY: Number(el.y),
      maxX: Number(el.x) + m.width,
      maxY: Number(el.y) + m.height,
    };
  }

  if (t === "arrow") {
    const pts = absoluteArrowPoints(el);
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const halfStroke = Number(el.strokeWidth || DEFAULTS.strokeWidth) * 0.6;
    let minX = Math.min(...xs) - halfStroke;
    let minY = Math.min(...ys) - halfStroke;
    let maxX = Math.max(...xs) + halfStroke;
    let maxY = Math.max(...ys) + halfStroke;
    if (el.label?.text) {
      const m = textBlockMetrics(el.label.text, el.label.fontSize || 14);
      const mid = pts[Math.floor((pts.length - 1) / 2)];
      minX = Math.min(minX, mid[0] - m.width / 2);
      minY = Math.min(minY, mid[1] - m.height / 2);
      maxX = Math.max(maxX, mid[0] + m.width / 2);
      maxY = Math.max(maxY, mid[1] + m.height / 2);
    }
    return { minX, minY, maxX, maxY };
  }

  const minX = Number(el.x);
  const minY = Number(el.y);
  const maxX = Number(el.x) + Number(el.width || 0);
  const maxY = Number(el.y) + Number(el.height || 0);
  const label = shapeLabelBounds(el);
  if (!label) return { minX, minY, maxX, maxY };
  return {
    minX: Math.min(minX, label.minX),
    minY: Math.min(minY, label.minY),
    maxX: Math.max(maxX, label.maxX),
    maxY: Math.max(maxY, label.maxY),
  };
}

function mergeBounds(allBounds) {
  if (allBounds.length === 0) {
    return { minX: 0, minY: 0, maxX: 800, maxY: 600 };
  }
  return allBounds.reduce(
    (acc, b) => ({
      minX: Math.min(acc.minX, b.minX),
      minY: Math.min(acc.minY, b.minY),
      maxX: Math.max(acc.maxX, b.maxX),
      maxY: Math.max(acc.maxY, b.maxY),
    }),
    {
      minX: Number.POSITIVE_INFINITY,
      minY: Number.POSITIVE_INFINITY,
      maxX: Number.NEGATIVE_INFINITY,
      maxY: Number.NEGATIVE_INFINITY,
    }
  );
}

function buildText(x, y, text, fontSize, opts = {}) {
  const m = lineMetrics(fontSize);
  const lines = String(text ?? "").split("\n");
  const anchor = opts.anchor || "start";
  const baseline = opts.baseline || "hanging";
  const fill = opts.fill || "#1e1e1e";
  const extra = opts.extraAttrs ? ` ${opts.extraAttrs}` : "";
  const tspans = lines
    .map((line, i) => {
      const yy = Number(y) + i * m.lineHeight;
      return `<tspan x="${Number(x)}" y="${yy}">${escXml(line)}</tspan>`;
    })
    .join("");
  return `<text font-family="Segoe UI, Arial, sans-serif" font-size="${m.fontSize}" fill="${escXml(
    fill
  )}" text-anchor="${anchor}" dominant-baseline="${baseline}"${extra}>${tspans}</text>`;
}

function labelToSvg(el, color) {
  if (!el.label || !el.label.text) return "";
  const fs = Number(el.label.fontSize || 16);
  const m = textBlockMetrics(el.label.text, fs);
  const cx = Number(el.x) + Number(el.width) / 2;
  const cy = Number(el.y) + Number(el.height) / 2;
  const startY = cy - m.height / 2;
  return buildText(cx, startY, el.label.text, fs, {
    anchor: "middle",
    baseline: "hanging",
    fill: color,
  });
}

function markerId(color) {
  return `arrow-${color.replace(/[^a-zA-Z0-9]/g, "") || "default"}`;
}

function renderElement(el, markerDefs) {
  const stroke = colorOrNone(el.strokeColor || DEFAULTS.strokeColor);
  const fill = colorOrNone(el.backgroundColor || DEFAULTS.backgroundColor);
  const strokeWidth = Number(el.strokeWidth ?? DEFAULTS.strokeWidth);
  const opacity = elementOpacity(el);
  const dash = el.strokeStyle === "dashed" ? ' stroke-dasharray="8 6"' : "";
  const common = `stroke="${escXml(stroke)}" stroke-width="${strokeWidth}" fill="${escXml(
    fill
  )}" opacity="${opacity}"${dash}`;

  switch (el.type) {
    case "rectangle": {
      const x = Number(el.x);
      const y = Number(el.y);
      const w = Number(el.width);
      const h = Number(el.height);
      const rx = el.roundness ? Math.min(12, w / 6, h / 6) : 0;
      return [
        `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" ry="${rx}" ${common} />`,
        labelToSvg(el, stroke === "none" ? "#1e1e1e" : stroke),
      ]
        .filter(Boolean)
        .join("");
    }
    case "ellipse": {
      const cx = Number(el.x) + Number(el.width) / 2;
      const cy = Number(el.y) + Number(el.height) / 2;
      const rx = Number(el.width) / 2;
      const ry = Number(el.height) / 2;
      return [
        `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" ${common} />`,
        labelToSvg(el, stroke === "none" ? "#1e1e1e" : stroke),
      ]
        .filter(Boolean)
        .join("");
    }
    case "diamond": {
      const x = Number(el.x);
      const y = Number(el.y);
      const w = Number(el.width);
      const h = Number(el.height);
      const pts = [
        [x + w / 2, y],
        [x + w, y + h / 2],
        [x + w / 2, y + h],
        [x, y + h / 2],
      ]
        .map((p) => p.join(","))
        .join(" ");
      return [
        `<polygon points="${pts}" ${common} />`,
        labelToSvg(el, stroke === "none" ? "#1e1e1e" : stroke),
      ]
        .filter(Boolean)
        .join("");
    }
    case "text": {
      const color = stroke === "none" ? "#1e1e1e" : stroke;
      const body = buildText(el.x, el.y, el.text || "", el.fontSize || DEFAULTS.fontSize, {
        fill: color,
      });
      return `<g opacity="${opacity}">${body}</g>`;
    }
    case "arrow": {
      const pts = absoluteArrowPoints(el);
      const ptAttr = pts.map((p) => p.join(",")).join(" ");
      let startMarker = "";
      let endMarker = "";
      if (el.endArrowhead && el.endArrowhead !== null && el.endArrowhead !== "bar") {
        const id = markerId(stroke);
        markerDefs.add(stroke);
        endMarker = ` marker-end="url(#${id})"`;
      }
      if (el.startArrowhead && el.startArrowhead !== null && el.startArrowhead !== "bar") {
        const id = markerId(stroke);
        markerDefs.add(stroke);
        startMarker = ` marker-start="url(#${id})"`;
      }
      const poly = `<polyline points="${ptAttr}" fill="none" stroke="${escXml(
        stroke
      )}" stroke-width="${strokeWidth}" opacity="${opacity}"${dash}${startMarker}${endMarker} />`;
      if (!el.label?.text) return poly;
      const start = pts[0];
      const end = pts[pts.length - 1];
      const cx = (start[0] + end[0]) / 2;
      const cy = (start[1] + end[1]) / 2 - 10;
      const text = buildText(cx, cy, el.label.text, el.label.fontSize || 14, {
        anchor: "middle",
        baseline: "hanging",
        fill: stroke === "none" ? "#1e1e1e" : stroke,
      });
      return `${poly}${text}`;
    }
    default:
      return "";
  }
}

function buildMarkers(colors) {
  if (colors.size === 0) return "";
  const items = [...colors]
    .map((color) => {
      const id = markerId(color);
      return `<marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="${escXml(
        color
      )}" /></marker>`;
    })
    .join("");
  return `<defs>${items}</defs>`;
}

function renderSvg(elements) {
  const drawables = elements.filter(
    (el) =>
      el &&
      typeof el === "object" &&
      !["cameraUpdate", "delete", "restoreCheckpoint"].includes(el.type)
  );
  const bounds = mergeBounds(drawables.map(elementBounds).filter(Boolean));
  const pad = 24;
  const minX = Math.floor(bounds.minX - pad);
  const minY = Math.floor(bounds.minY - pad);
  const width = Math.ceil(bounds.maxX - bounds.minX + pad * 2);
  const height = Math.ceil(bounds.maxY - bounds.minY + pad * 2);
  const markerDefs = new Set();
  const body = drawables.map((el) => renderElement(el, markerDefs)).join("\n");
  const defs = buildMarkers(markerDefs);
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="${minX} ${minY} ${width} ${height}">
${defs}
<rect x="${minX}" y="${minY}" width="${width}" height="${height}" fill="#ffffff" />
${body}
</svg>
`;
}

function resolveOutputPath(inputPath, outputPathArg) {
  if (outputPathArg) {
    return path.resolve(outputPathArg);
  }
  const parsed = path.parse(inputPath);
  return path.join(parsed.dir, `${parsed.name}.svg`);
}

function main() {
  const [, , inputArg, outputArg] = process.argv;
  if (!inputArg) {
    printUsageAndExit();
  }
  const inputPath = path.resolve(inputArg);
  if (!fs.existsSync(inputPath)) {
    throw new Error(`Input file not found: ${inputPath}`);
  }
  const outputPath = resolveOutputPath(inputPath, outputArg);
  const data = readJson(inputPath);
  const elements = ensureElements(data);
  const svg = renderSvg(elements);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, svg, "utf8");
  console.log(`SVG exported: ${outputPath}`);
}

main();
