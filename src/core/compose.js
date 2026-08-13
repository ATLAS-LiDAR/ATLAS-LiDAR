/**
 * Compose the cloud a victim would actually report after an attack.
 *
 * Attacks return a keepMask over the recorded frame plus any points they added,
 * rather than a new cloud, so that the original frame stays untouched and the
 * UI can render clean and perturbed side by side. Everything downstream of an
 * attack - the detector, defenses, export - wants the composed result, and the
 * `origin` tag is what lets a defense or a renderer tell real returns from
 * spoofed ones (a defense must never read `origin`: that would be cheating).
 */

export const ORIGIN = Object.freeze({ REAL: 0, SPOOF: 1, REPLACEMENT: 2 });

export function composeCloud({ frame, keepMask = null, spoof = null, replacement = null }) {
  let kept = frame.count;
  if (keepMask) {
    kept = 0;
    for (let i = 0; i < frame.count; i++) if (keepMask[i]) kept++;
  }
  const nSpoof = spoof ? spoof.x.length : 0;
  const nRep = replacement ? replacement.x.length : 0;
  const n = kept + nSpoof + nRep;

  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const z = new Float32Array(n);
  const intensity = new Float32Array(n);
  const origin = new Uint8Array(n);
  // index back into the source frame, or -1 for added points
  const sourceIndex = new Int32Array(n).fill(-1);

  let m = 0;
  for (let i = 0; i < frame.count; i++) {
    if (keepMask && !keepMask[i]) continue;
    x[m] = frame.x[i];
    y[m] = frame.y[i];
    z[m] = frame.z[i];
    intensity[m] = frame.intensity[i];
    origin[m] = ORIGIN.REAL;
    sourceIndex[m] = i;
    m++;
  }
  for (let i = 0; i < nSpoof; i++) {
    x[m] = spoof.x[i];
    y[m] = spoof.y[i];
    z[m] = spoof.z[i];
    intensity[m] = spoof.intensity ? spoof.intensity[i] : 0;
    origin[m] = ORIGIN.SPOOF;
    m++;
  }
  for (let i = 0; i < nRep; i++) {
    x[m] = replacement.x[i];
    y[m] = replacement.y[i];
    z[m] = replacement.z[i];
    intensity[m] = replacement.intensity ? replacement.intensity[i] : 0;
    origin[m] = ORIGIN.REPLACEMENT;
    m++;
  }

  return { x, y, z, intensity, origin, sourceIndex, n, sensor: frame.sensor };
}
