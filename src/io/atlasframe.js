/**
 * Reading recorded frames in the browser.
 *
 * `.atlasframe` is the compact format written by tools/export_frames.py: an
 * 8-byte magic, a uint32 header length, a UTF-8 JSON header, then the payload
 * blocks the header's `arrays` map points at. Positions are int16 quantized at
 * 5 mm, intensity is uint8 with a scale factor, so a 120k-point HDL-64E frame is
 * ~820 kB instead of the ~1.9 MB raw float32 would cost.
 *
 * User uploads are handled without any conversion step, because both source
 * formats are bare interleaved float32: KITTI `.bin` is stride 4
 * (x, y, z, intensity) and nuScenes `.pcd.bin` is stride 5 (plus ring index).
 */
import { PointFrame } from "../core/frame.js";

const MAGIC = "ATLFRAME";

function readMagic(view) {
  let s = "";
  for (let i = 0; i < 8; i++) s += String.fromCharCode(view.getUint8(i));
  return s;
}

export function parseAtlasFrame(arrayBuffer) {
  if (arrayBuffer.byteLength < 12) {
    throw new Error("buffer too short to be an atlasframe (bad magic)");
  }
  const view = new DataView(arrayBuffer);
  const magic = readMagic(view);
  if (magic !== MAGIC) {
    throw new Error('bad magic "' + magic + '": not an atlasframe file');
  }
  const headerLen = view.getUint32(8, true);
  const headerBytes = new Uint8Array(arrayBuffer, 12, headerLen);
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  const base = 12 + headerLen;

  const block = (name) => {
    const spec = header.arrays[name];
    if (!spec) throw new Error('atlasframe missing array "' + name + '"');
    const off = base + spec.offset;
    if (spec.dtype !== "int16") return new Uint8Array(arrayBuffer, off, spec.count);
    // The exporter pads the header so this offset is aligned, but a file written
    // by an older exporter can land on an odd byte; copy rather than throw.
    if (off % 2 === 0) return new Int16Array(arrayBuffer, off, spec.count);
    return new Int16Array(arrayBuffer.slice(off, off + spec.count * 2));
  };

  const n = header.nPoints;
  const q = header.posQuantum;
  const is = header.intensityScale;
  const qx = block("x");
  const qy = block("y");
  const qz = block("z");
  const qi = block("intensity");

  // Float64 for positions: the payload is int16 * quantum, and computing that in
  // float64 makes the coordinates bit-identical to what numpy produces from the
  // same bytes. Rounding to float32 shifts ranges in the last bits, which turns
  // an exact occlusion tie into a near-miss - a real off-by-one that
  // tools/crossvalidate.py caught against ATLAS's float64 path.
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const z = new Float64Array(n);
  const intensity = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = qx[i] * q;
    y[i] = qy[i] * q;
    z[i] = qz[i] * q;
    intensity[i] = qi[i] * is;
  }
  return { header, x, y, z, intensity };
}

export async function loadAtlasFrame(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("failed to fetch " + url + ": " + res.status);
  return parseAtlasFrame(await res.arrayBuffer());
}

export async function loadManifest(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("failed to fetch " + url + ": " + res.status);
  return res.json();
}

/**
 * Guess the interleave stride of a raw lidar dump. 4 is KITTI, 5 is nuScenes.
 * A length divisible by both is genuinely ambiguous, so say so rather than
 * silently picking one and mangling the cloud.
 */
export function detectStride(byteLength) {
  if (byteLength % 4 !== 0) {
    throw new Error("buffer length " + byteLength + " is not a whole number of float32 values");
  }
  const floats = byteLength / 4;
  const ok4 = floats % 4 === 0;
  const ok5 = floats % 5 === 0;
  if (ok4 && ok5) return { stride: 4, ambiguous: true, candidates: [4, 5] };
  if (ok4) return { stride: 4, ambiguous: false, candidates: [4] };
  if (ok5) return { stride: 5, ambiguous: false, candidates: [5] };
  throw new Error("buffer of " + floats + " float32 values divides by neither stride 4 nor 5");
}

/** Parse a bare interleaved float32 point dump (a user upload). */
export function parseRawLidar(arrayBuffer, { stride = null, name = "upload", sensor = null } = {}) {
  const s = stride ?? detectStride(arrayBuffer.byteLength).stride;
  const src = new Float32Array(arrayBuffer);
  if (src.length % s !== 0) {
    throw new Error("buffer of " + src.length + " floats is not a multiple of stride " + s);
  }
  const n = src.length / s;
  // Float64 for positions: the payload is int16 * quantum, and computing that in
  // float64 makes the coordinates bit-identical to what numpy produces from the
  // same bytes. Rounding to float32 shifts ranges in the last bits, which turns
  // an exact occlusion tie into a near-miss - a real off-by-one that
  // tools/crossvalidate.py caught against ATLAS's float64 path.
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const z = new Float64Array(n);
  const intensity = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * s;
    x[i] = src[o];
    y[i] = src[o + 1];
    z[i] = src[o + 2];
    intensity[i] = src[o + 3];
  }
  // Infer the dataset from the intensity scale: KITTI ships 0-1, nuScenes 0-255.
  let max = 0;
  for (let i = 0; i < n; i++) if (intensity[i] > max) max = intensity[i];
  const dataset = max <= 1.0 + 1e-6 ? "kitti" : "nuscenes";
  return {
    header: {
      version: 1,
      name,
      nPoints: n,
      dataset,
      sensor: sensor ?? (s === 5 ? "hdl32e" : "hdl64e"),
      posQuantum: 0,
      intensityScale: 1,
      boxes: [],
      uploaded: true,
      stride: s,
    },
    x,
    y,
    z,
    intensity,
  };
}

/** Index a loaded frame onto a sensor's scan lattice. */
export function toPointFrame(loaded, sensor) {
  return new PointFrame({
    x: loaded.x,
    y: loaded.y,
    z: loaded.z,
    intensity: loaded.intensity,
    sensor,
    meta: loaded.header,
  });
}
