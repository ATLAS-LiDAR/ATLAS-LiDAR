/**
 * Round-trip export: hand a configured attack to the real detectors.
 *
 * The proxy detector in detect.js measures geometric detectability, not model
 * vulnerability, so it cannot produce ASR numbers comparable to the paper's.
 * This is the path that can: serialise the perturbed cloud plus the phantom box
 * into a bundle that tools/import_attack.py converts into the `{segment}_d.pkl`
 * format eval_matrix_kitti.py already consumes, unchanged.
 *
 * `.atlasattack` layout, chosen so numpy can read the payload with no parsing:
 *
 *   [0..8)              magic "ATLATTK1"
 *   [8..12)             uint32 LE header length (space-padded to align payload)
 *   [12..12+len)        UTF-8 JSON header
 *   [payloadOffset..]   float32 interleaved x, y, z, intensity  <- KITTI native
 *
 * The payload is exactly what KITTI's own velodyne .bin files contain, so
 * `np.frombuffer(buf, dtype=np.float32, offset=payloadOffset).reshape(-1, 4)`
 * is the whole reader, and `fr["points"][:, :4]` in the harness just works.
 */

export const ATTACK_MAGIC = "ATLATTK1";

const PROXY_NOTE =
  "Perturbed cloud exported from the ATLAS sandbox. Any ASR shown in the sandbox UI " +
  "is from a clustering PROXY detector and is not comparable to the paper's " +
  "numbers; run this bundle through the real detectors for those.";

/**
 * @param {object} cloud    composeCloud() result (the perturbed cloud)
 * @param {number[]|null} spoofGt  phantom box [x,y,z,dx,dy,dz,heading], or null
 *                                 for a removal attack (no phantom exists)
 */
export function buildAttackBundle({
  cloud,
  sensor,
  frameId,
  dataset,
  spoofGt = null,
  attack = {},
  spooferConfig = null,
  stats = {},
  scheduledAttack = true,
  sceneSlug = null,
  createdAt = null,
}) {
  const n = cloud.n;

  const header = {
    format: ATTACK_MAGIC,
    version: 1,
    frameId: String(frameId),
    sceneSlug,
    dataset,
    sensor: sensor.id,
    sensorSpec: {
      nBeams: sensor.nBeams,
      azResDeg: (sensor.azRes * 180) / Math.PI,
      hz: sensor.hz,
      returnMode: sensor.returnMode,
      maxRange: sensor.maxRange,
      minRange: sensor.minRange,
    },
    nPoints: n,
    stride: 4,
    dtype: "float32",
    spoofGt: spoofGt ? Array.from(spoofGt).slice(0, 7) : null,
    scheduledAttack: !!scheduledAttack,
    attack,
    spoofer: spooferConfig
      ? {
          id: spooferConfig.id,
          powerRatio: spooferConfig.powerRatio,
          divergenceMrad: spooferConfig.divergenceMrad,
          latencyS: spooferConfig.latencyS,
          prfHz: spooferConfig.prfHz,
          jitterS: spooferConfig.jitterS,
          syncMode: spooferConfig.syncMode,
          canPredict: spooferConfig.canPredict,
        }
      : null,
    stats,
    createdAt,
    note: PROXY_NOTE,
    payloadOffset: 0, // patched below once the header length is known
  };

  // The payload is a float32 view, which needs a 4-byte aligned offset. Pad the
  // JSON with spaces (which JSON ignores) until it is.
  let headerBytes = new TextEncoder().encode(JSON.stringify(header));
  let payloadOffset = 12 + headerBytes.length;
  let pad = (-payloadOffset) % 4;
  if (pad < 0) pad += 4;
  header.payloadOffset = payloadOffset + pad;
  // re-serialise with the real offset, then re-pad (length may have changed)
  headerBytes = new TextEncoder().encode(JSON.stringify(header));
  payloadOffset = 12 + headerBytes.length;
  pad = (-payloadOffset) % 4;
  if (pad < 0) pad += 4;
  if (header.payloadOffset !== payloadOffset + pad) {
    header.payloadOffset = payloadOffset + pad;
    headerBytes = new TextEncoder().encode(JSON.stringify(header).padEnd(headerBytes.length, " "));
    payloadOffset = 12 + headerBytes.length;
    pad = (-payloadOffset) % 4;
    if (pad < 0) pad += 4;
    header.payloadOffset = payloadOffset + pad;
  }

  const total = payloadOffset + pad + n * 4 * 4;
  const buffer = new ArrayBuffer(total);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  for (let i = 0; i < 8; i++) bytes[i] = ATTACK_MAGIC.charCodeAt(i);
  view.setUint32(8, headerBytes.length + pad, true);
  bytes.set(headerBytes, 12);
  for (let i = 0; i < pad; i++) bytes[12 + headerBytes.length + i] = 0x20; // space

  const pts = new Float32Array(buffer, header.payloadOffset, n * 4);
  for (let i = 0; i < n; i++) {
    pts[i * 4] = cloud.x[i];
    pts[i * 4 + 1] = cloud.y[i];
    pts[i * 4 + 2] = cloud.z[i];
    pts[i * 4 + 3] = cloud.intensity[i];
  }

  return { buffer, header, byteLength: total };
}

export function parseAttackBundle(arrayBuffer) {
  if (arrayBuffer.byteLength < 12) throw new Error("buffer too short: bad magic");
  const bytes = new Uint8Array(arrayBuffer);
  let magic = "";
  for (let i = 0; i < 8; i++) magic += String.fromCharCode(bytes[i]);
  if (magic !== ATTACK_MAGIC) throw new Error('bad magic "' + magic + '": not an attack bundle');

  const view = new DataView(arrayBuffer);
  const headerLen = view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuffer, 12, headerLen)));
  const points = new Float32Array(arrayBuffer, header.payloadOffset, header.nPoints * 4);
  return { header, points };
}

/** Safe, informative filename for a downloaded bundle. */
export function bundleFilename({ frameId, mode, slug = null }) {
  const safe = (s) => String(s).replace(/[^A-Za-z0-9._-]+/g, "-");
  const parts = [slug ? safe(slug) : "frame", safe(frameId), safe(mode)];
  return parts.filter(Boolean).join("_") + ".atlasattack";
}

/** Trigger a browser download of a bundle. */
export function downloadBundle(bundle, filename) {
  const blob = new Blob([bundle.buffer], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Base64 of a bundle, for handing it out of headless Chrome via the DOM. */
export function bundleToBase64(bundle) {
  const bytes = new Uint8Array(bundle.buffer);
  let s = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(s);
}
