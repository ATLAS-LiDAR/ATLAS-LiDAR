/**
 * Reading the ATLAS phantom trace libraries.
 *
 * `.atlastrace` is written by tools/export_traces.py: 8-byte magic, uint32
 * header length (space-padded so the payload is 8-byte aligned), UTF-8 JSON
 * header, then int16 x/y/z at 5 mm plus uint8 intensity, concatenated across
 * traces. Trace k occupies [prefix[k], prefix[k] + counts[k]).
 *
 * Points are stored RAW - absolute lidar-frame coordinates, exactly as the pkl
 * holds them - so `centerTrace` in atlas-inject.js can run ATLAS's own transform
 * rather than us pre-baking one.
 */

const MAGIC = "ATLTRACE";

export function parseTraceLibrary(arrayBuffer) {
  if (arrayBuffer.byteLength < 12) throw new Error("buffer too short: bad magic");
  const bytes = new Uint8Array(arrayBuffer);
  let magic = "";
  for (let i = 0; i < 8; i++) magic += String.fromCharCode(bytes[i]);
  if (magic !== MAGIC) throw new Error('bad magic "' + magic + '": not an .atlastrace library');

  const view = new DataView(arrayBuffer);
  const headerLen = view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuffer, 12, headerLen)));

  const base = header.payloadOffset;
  const block = (name) => {
    const spec = header.arrays[name];
    const off = base + spec.offset;
    if (spec.dtype !== "int16") return new Uint8Array(arrayBuffer, off, spec.count);
    return off % 2 === 0
      ? new Int16Array(arrayBuffer, off, spec.count)
      : new Int16Array(arrayBuffer.slice(off, off + spec.count * 2));
  };

  // prefix offsets so a trace can be sliced without scanning
  const prefix = new Int32Array(header.counts.length + 1);
  for (let i = 0; i < header.counts.length; i++) prefix[i + 1] = prefix[i] + header.counts[i];

  return {
    header,
    prefix,
    qx: block("x"),
    qy: block("y"),
    qz: block("z"),
    qi: block("intensity"),
  };
}

export async function loadTraceLibrary(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("failed to fetch " + url + ": " + res.status);
  return parseTraceLibrary(await res.arrayBuffer());
}

export async function loadTraceManifest(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("failed to fetch " + url + ": " + res.status);
  return res.json();
}

/** One trace, dequantized: {x, y, z, intensity, box, range}. */
export function getTrace(lib, index) {
  const { header, prefix, qx, qy, qz, qi } = lib;
  if (index < 0 || index >= header.nTraces) {
    throw new Error("trace index " + index + " out of range (0.." + (header.nTraces - 1) + ")");
  }
  const s = prefix[index];
  const n = header.counts[index];
  const q = header.posQuantum;
  const is = header.intensityScale;

  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const z = new Float64Array(n);
  const intensity = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = qx[s + i] * q;
    y[i] = qy[s + i] * q;
    z[i] = qz[s + i] * q;
    intensity[i] = qi[s + i] * is;
  }
  return {
    x,
    y,
    z,
    intensity,
    box: header.boxes[index].slice(),
    range: header.ranges[index],
    index,
  };
}

/** Every trace in a library, as an array (what placeTraceRelative expects). */
export function allTraces(lib) {
  const out = [];
  for (let i = 0; i < lib.header.nTraces; i++) out.push(getTrace(lib, i));
  return out;
}
