#!/usr/bin/env python3
"""Minimal Velodyne-over-pcap reader (no dpkt/scapy).

Reads a classic libpcap file, keeps UDP payloads of length 1206 (Velodyne data
packets) and 512/554 (position packets), and decodes enough of the data packet
to report sensor model, return mode, frame boundaries and per-frame geometry.

Data packet layout (all Velodyne spinning sensors):
    12 blocks x 100 bytes:
        u16 flag (0xEEFF)
        u16 azimuth (centi-degrees)
        32 x (u16 distance[2mm units], u8 reflectivity)
    u32 timestamp (us since top of hour)
    u8  return mode  (0x37 strongest, 0x38 last, 0x39 dual)
    u8  product id   (0x21 HDL-32E, 0x22 VLP-16, 0x28 VLP-32C, 0xA1 VLS-128)
"""
import struct
import sys

RETURN_MODE = {0x37: "strongest", 0x38: "last", 0x39: "dual"}
PRODUCT_ID = {0x21: "HDL-32E", 0x22: "VLP-16", 0x28: "VLP-32C",
              0x31: "VLS-128", 0xA1: "VLS-128"}

VLP16_INCL = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
VLP32C_INCL = [-25.0, -1.0, -1.667, -15.639, -11.31, 0.0, -0.667, -8.843,
               -7.254, 0.333, -0.333, -6.148, -5.333, 1.333, 0.667, -4.0,
               -4.667, 1.667, 1.0, -3.667, -3.333, 3.333, 2.333, -2.667,
               -3.0, 7.0, 4.667, -2.333, -2.0, 15.0, 10.333, -1.333]


def iter_packets(path):
    """Yield (ts_float, udp_dst_port, payload_bytes) for every UDP packet."""
    with open(path, "rb") as fh:
        gh = fh.read(24)
        magic = struct.unpack("<I", gh[:4])[0]
        if magic == 0xA1B2C3D4:
            endian, nano = "<", False
        elif magic == 0xD4C3B2A1:
            endian, nano = ">", False
        elif magic == 0xA1B23C4D:
            endian, nano = "<", True
        elif magic == 0x4D3CB2A1:
            endian, nano = ">", True
        else:
            raise SystemExit(f"not a classic pcap: magic={magic:#x}")
        while True:
            ph = fh.read(16)
            if len(ph) < 16:
                return
            tsec, tfrac, caplen, _ = struct.unpack(endian + "IIII", ph)
            data = fh.read(caplen)
            if len(data) < caplen:
                return
            ts = tsec + tfrac / (1e9 if nano else 1e6)
            # Ethernet II
            if len(data) < 42:
                continue
            ethertype = struct.unpack("!H", data[12:14])[0]
            if ethertype != 0x0800:
                continue
            ihl = (data[14] & 0x0F) * 4
            if data[14 + 9] != 17:  # UDP
                continue
            u = 14 + ihl
            dport = struct.unpack("!H", data[u + 2:u + 4])[0]
            yield ts, dport, data[u + 8:]


def decode_blocks(pl):
    """Return (list of (azimuth_deg, [(dist_m, refl)] * 32), timestamp_us,
    return_mode, product_id)."""
    blocks = []
    for b in range(12):
        off = b * 100
        flag, az = struct.unpack_from("<HH", pl, off)
        if flag != 0xEEFF:
            continue
        chans = []
        for c in range(32):
            d, r = struct.unpack_from("<HB", pl, off + 4 + c * 3)
            chans.append((d * 0.002, r))
        blocks.append((az / 100.0, chans))
    tstamp, rmode, pid = struct.unpack_from("<IBB", pl, 1200)
    return blocks, tstamp, rmode, pid


def analyse(path, max_packets=None, verbose=True):
    n_data = n_pos = 0
    rmodes, pids = {}, {}
    first_ts = last_ts = None
    frames = 0          # azimuth wrap count
    prev_az = None
    az_steps = []
    dist_hist = [0] * 12       # 0-5,5-10,...,55-60,60+ metres
    n_returns = n_empty = 0
    refl_sum = 0
    per_frame_pkts = []
    cur = 0
    for ts, dport, pl in iter_packets(path):
        if first_ts is None:
            first_ts = ts
        last_ts = ts
        if dport == 8308 or len(pl) in (512, 554):
            n_pos += 1
            continue
        if len(pl) != 1206:
            continue
        n_data += 1
        cur += 1
        blocks, tstamp, rmode, pid = decode_blocks(pl)
        rmodes[rmode] = rmodes.get(rmode, 0) + 1
        pids[pid] = pids.get(pid, 0) + 1
        for az, chans in blocks:
            if prev_az is not None:
                d = az - prev_az
                if d < -180:      # wrapped past 360
                    frames += 1
                    per_frame_pkts.append(cur)
                    cur = 0
                elif d >= 0:
                    az_steps.append(d)
            prev_az = az
            for dist, refl in chans:
                if dist <= 0.0:
                    n_empty += 1
                else:
                    n_returns += 1
                    refl_sum += refl
                    dist_hist[min(int(dist // 5), 11)] += 1
        if max_packets and n_data >= max_packets:
            break

    dur = (last_ts - first_ts) if first_ts is not None else 0
    nz = [s for s in az_steps if s > 0]
    nz.sort()
    med_step = nz[len(nz) // 2] if nz else float("nan")
    out = {
        "path": path, "data_packets": n_data, "position_packets": n_pos,
        "duration_s": dur, "frames": frames,
        "return_mode": {RETURN_MODE.get(k, hex(k)): v for k, v in rmodes.items()},
        "product": {PRODUCT_ID.get(k, hex(k)): v for k, v in pids.items()},
        "median_az_step_deg": med_step,
        "returns": n_returns, "empty": n_empty,
        "fill_rate": n_returns / max(1, n_returns + n_empty),
        "mean_refl": refl_sum / max(1, n_returns),
        "dist_hist": dist_hist,
        "pkts_per_frame_median": (sorted(per_frame_pkts)[len(per_frame_pkts) // 2]
                                  if per_frame_pkts else None),
    }
    if verbose:
        print(f"\n=== {path} ===")
        print(f"  data pkts      : {n_data}   position pkts: {n_pos}")
        print(f"  duration       : {dur:.2f} s   -> {n_data/max(dur,1e-9):.0f} pkt/s")
        print(f"  sensor         : {out['product']}")
        print(f"  return mode    : {out['return_mode']}")
        print(f"  azimuth frames : {frames}  ({frames/max(dur,1e-9):.2f} Hz), "
              f"median pkts/frame {out['pkts_per_frame_median']}")
        print(f"  median az step : {med_step:.3f} deg/block")
        print(f"  returns        : {n_returns}  empty: {n_empty}  "
              f"fill {out['fill_rate']*100:.1f}%  mean refl {out['mean_refl']:.1f}")
        labels = [f"{i*5}-{i*5+5}" for i in range(11)] + ["60+"]
        print("  range hist (m) : " + ", ".join(
            f"{l}:{c}" for l, c in zip(labels, dist_hist) if c))
    return out


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    files = sys.argv[2:] if lim else sys.argv[1:]
    for p in files:
        analyse(p, max_packets=lim)
