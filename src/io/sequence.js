/**
 * Reading driving sequences.
 *
 * `.atlasseq` is written by tools/export_sequence.py: 8-byte magic, uint32 header
 * length, UTF-8 JSON header, then int16 x/y/z at 5 mm and uint8 intensity,
 * concatenated across keyframes. Frame k occupies
 * [prefix[k], prefix[k] + counts[k]).
 *
 * Each header frame carries its ego->global pose and its ground-truth boxes
 * (with velocity, which the A-HFR tracker needs). `dtSec` is 0.5 for nuScenes
 * keyframes, and must be passed to match_with_velocity rather than Waymo's 0.1.
 *
 * `frameOfSequence` returns something shaped exactly like a `.atlasframe` load,
 * so the rest of the app treats a sequence keyframe and a static frame the same.
 */

const MAGIC = "ATLASSEQ";

export function parseSequence(arrayBuffer) {
  if (arrayBuffer.byteLength < 12) throw new Error("buffer too short: bad magic");
  const bytes = new Uint8Array(arrayBuffer);
  let magic = "";
  for (let i = 0; i < 8; i++) magic += String.fromCharCode(bytes[i]);
  if (magic !== MAGIC) throw new Error('bad magic "' + magic + '": not an .atlasseq sequence');

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

  const prefix = new Int32Array(header.counts.length + 1);
  for (let i = 0; i < header.counts.length; i++) prefix[i + 1] = prefix[i] + header.counts[i];

  return { header, prefix, qx: block("x"), qy: block("y"), qz: block("z"), qi: block("intensity") };
}

export async function loadSequence(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("failed to fetch " + url + ": " + res.status);
  return parseSequence(await res.arrayBuffer());
}

export async function loadSequenceManifest(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("failed to fetch " + url + ": " + res.status);
  return res.json();
}

/**
 * One keyframe, in the same shape a `.atlasframe` load produces so it can be fed
 * straight to toPointFrame().
 */
export function frameOfSequence(seq, index) {
  const { header, prefix, qx, qy, qz, qi } = seq;
  if (index < 0 || index >= header.nFrames) {
    throw new Error("frame index " + index + " out of range (0.." + (header.nFrames - 1) + ")");
  }
  const s = prefix[index];
  const n = header.counts[index];
  const q = header.posQuantum;
  const is = header.intensityScale;

  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const z = new Float32Array(n);
  const intensity = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = qx[s + i] * q;
    y[i] = qy[s + i] * q;
    z[i] = qz[s + i] * q;
    intensity[i] = qi[s + i] * is;
  }

  const meta = header.frames[index];
  return {
    header: {
      version: 1,
      name: header.slug + "#" + index,
      nPoints: n,
      dataset: header.dataset,
      sensor: header.sensor,
      posQuantum: q,
      intensityScale: is,
      boxes: meta.boxes,
      frameId: meta.frameId,
      pose: meta.pose,
      timestamp: meta.timestamp,
      frameIndex: index,
      sequenceSlug: header.slug,
      dtSec: header.dtSec,
    },
    x,
    y,
    z,
    intensity,
  };
}
