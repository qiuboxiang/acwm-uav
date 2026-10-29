"""Video gallery of every collected trajectory: FPV + top-down path + height bar + action label (visualisation only).

  python build_gallery_v5.py --data ../../../data/collected_10h_v5 --out ../exploration_v5/gallery_all
  python build_gallery_v5.py --data ../../../data/collected_10h_v5 --out ../exploration_v5/gallery_all --page_only
Previews are 10 fps at 700x274. Artifact asset stores take files up to 20 MiB, so trajectories are packed --pack per mp4 (pk/pack_NNN.mp4, played by time range in the
page) and posters are tiled into sprite sheets (ps/sheet_NN.jpg). manifest.json lists every trajectory with its pack, offset and
metrics; index.html inlines it together with every archived heatmap version (docs/exploration/heatmaps/<tag>/heatmap3d.json) and
each trajectory's path. Simulator poses are used for the top-down panel and paths (display only; the agent never sees them).
"""
import argparse, glob, json, os, re, subprocess
from multiprocessing import Pool
import numpy as np, cv2

FW, FH, M, STRIP = 448, 252, 252, 22  # FPV, map panel, label strip (px)
STEP = 1  # keep every STEP-th recorded frame (2 = 5 fps previews when space is tight)
W, H = FW + M, FH + STRIP
TW, TH, SC, SR = 280, 110, 5, 8  # poster tile size, sheet columns/rows
ARROWS = {"fly_forward": "FWD", "fly_backward": "BACK", "fly_left": "LEFT", "fly_right": "RIGHT", "fly_up": "UP", "fly_down": "DOWN", "turn_left": "TURN L", "turn_right": "TURN R"}
COLORS = {"fly": (90, 200, 80), "turn": (60, 170, 255), "up": (255, 160, 60), "down": (255, 160, 60)}
HERE = os.path.dirname(os.path.abspath(__file__))
CODEC = {"h264": lambda q: ["-c:v", "libx264", "-preset", "slow", "-crf", str(q)],
         "vp9": lambda q: ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(q), "-row-mt", "1", "-deadline", "good", "-cpu-used", "2", "-threads", "2"],
         "av1": lambda q: ["-c:v", "libsvtav1", "-rc", "0", "-qp", str(q), "-preset", "6"]}
EXT = {"h264": "mp4", "vp9": "webm", "av1": "mp4"}


def nat(s): return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]


def rows(data):
    out = []
    for f in sorted(glob.glob(f"{data}/w*/index.jsonl")):
        w = os.path.basename(os.path.dirname(f))
        for l in open(f):
            r = json.loads(l); r["dir"] = os.path.join(os.path.dirname(f), r["path"]); r["id"] = f"{w}/{os.path.basename(r['path'])}"; out.append(r)
    return sorted(out, key=lambda r: nat(r["id"]))


def groups_of(R, k=6, n_rand=12, seed=0):
    """Named highlight groups (tags), same rule as the first gallery: top/bottom k by several metrics plus a fixed random sample."""
    by = lambda key, rev: sorted(R, key=lambda r: r[key], reverse=rev)
    med = np.median([r["coverage"] for r in R]); rng = np.random.default_rng(seed)
    G = {"best": by("coverage", True)[:k], "typical": sorted(R, key=lambda r: abs(r["coverage"] - med))[:k], "worst": by("coverage", False)[:k],
         "collision": by("collisions", True)[:k], "vertical": by("vertical_frame_frac", True)[:k], "longest": by("flight_m", True)[:k],
         "random": [R[i] for i in rng.choice(len(R), n_rand, replace=False)]}
    tags = {}
    for g, rs in G.items():
        for r in rs: tags.setdefault(r["id"], []).append(g)
    return tags


class MapPanel:
    def __init__(self, geo):
        pts = geo["pts"]; lo = geo["bounds"][0][[0, 2]] - 0.5; hi = geo["bounds"][1][[0, 2]] + 0.5
        self.s = (M - 12) / float((hi - lo).max()); self.lo = lo; self.off = (M - (hi - lo) * self.s) / 2
        bg = np.full((M, M, 3), 250, np.uint8); P = self.px(pts[:, [0, 2]]); cnt = np.zeros((M, M))
        ok = (P >= 0).all(1) & (P < M).all(1); np.add.at(cnt, (P[ok, 1], P[ok, 0]), 1)
        shade = np.clip(np.log1p(cnt) / np.log1p(np.percentile(cnt[cnt > 0], 95)), 0, 1)
        self.bg = (bg - (shade[..., None] * np.array([150, 140, 130]))).astype(np.uint8)

    def px(self, xz): return np.round((xz - self.lo) * self.s + self.off).astype(int)


def label(fr):
    if not fr: return "START", (200, 200, 200)
    k, v = next(iter(fr.items())); kind = "turn" if k.startswith("turn") else k.split("_")[1] if k in ("fly_up", "fly_down") else "fly"
    return f"{ARROWS.get(k, k)} {v:.2f}{'deg' if kind == 'turn' else 'm'}", COLORS[kind]


def frames(r, mp):
    """Yield composite BGR frames of one trajectory; the poster is the frame at one third."""
    d = r["dir"]; Q = np.array(json.load(open(f"{d}/posture.txt")), float)
    acts = {a["frame"]: a["actions"] for a in json.load(open(f"{d}/real_action.json"))["frame"]}
    P = mp.px(Q[:, :2]); zlo, zhi = min(Q[:, 2].min(), -0.5), max(Q[:, 2].max(), 0.5)
    trail = mp.bg.copy(); cv2.polylines(trail, [P.reshape(-1, 1, 2)], False, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.drawMarker(trail, tuple(map(int, P[0])), (40, 40, 40), cv2.MARKER_STAR, 10, 1, cv2.LINE_AA)
    cap = cv2.VideoCapture(f"{d}/video.mp4"); n = len(Q)
    for i in range(n):
        ok, f = cap.read()
        if i: cv2.line(trail, tuple(map(int, P[i - 1])), tuple(map(int, P[i])), (180, 90, 30), 2, cv2.LINE_AA)
        if i % STEP and i != n // 3: continue
        f = cv2.resize(f, (FW, FH), interpolation=cv2.INTER_AREA) if ok else np.zeros((FH, FW, 3), np.uint8)
        m = trail.copy(); c = tuple(map(int, P[i])); yaw = np.radians(Q[i, 3]); fwd = np.array([-np.sin(yaw), -np.cos(yaw)])
        for side in (-1, 1):  # 90 deg field of view wedge
            a = yaw + side * np.pi / 4; e = np.array([-np.sin(a), -np.cos(a)]) * 1.6 * mp.s
            cv2.line(m, c, (int(c[0] + e[0]), int(c[1] + e[1])), (60, 140, 230), 1, cv2.LINE_AA)
        cv2.arrowedLine(m, c, (int(c[0] + fwd[0] * 11), int(c[1] + fwd[1] * 11)), (20, 20, 200), 2, cv2.LINE_AA, tipLength=.4)
        cv2.circle(m, c, 3, (20, 20, 200), -1, cv2.LINE_AA)
        hb = int((Q[i, 2] - zlo) / (zhi - zlo) * (M - 30)); cv2.rectangle(m, (M - 10, 15), (M - 4, M - 15), (220, 220, 220), -1)
        cv2.rectangle(m, (M - 10, M - 15 - hb), (M - 4, M - 15), (255, 160, 60), -1)
        txt, col = label(acts.get(i)); fr = np.full((H, W, 3), 30, np.uint8); fr[:FH, :FW] = f; fr[:FH, FW:] = m
        cv2.putText(fr, txt, (8, FH + 16), cv2.FONT_HERSHEY_SIMPLEX, .42, col, 1, cv2.LINE_AA)
        cv2.putText(fr, f"{r['id']}  {i + 1}/{n}  t={i / 10:.1f}s  h={Q[i, 2]:+.2f}m", (130, FH + 16), cv2.FONT_HERSHEY_SIMPLEX, .36, (200, 200, 200), 1, cv2.LINE_AA)
        if i % STEP == 0: yield i, n, fr
        if i == n // 3: yield -1, n, fr  # poster frame (not written to the video)
    cap.release()


def render_pack(job):
    k, R, geo_fn, out, crf, codec = job; mp4 = f"{out}/pk/pack_{k:03d}.{EXT[codec]}"; man = f"{out}/pk/pack_{k:03d}.json"
    if os.path.exists(man): return json.load(open(man))
    mp = MapPanel(np.load(geo_fn)); tmp = mp4 + f".part.{EXT[codec]}"
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(10 // STEP), "-i", "-", *CODEC[codec](crf), "-g", str(30 // STEP), "-pix_fmt", "yuv420p", *(["-movflags", "+faststart"] if EXT[codec] == "mp4" else []), tmp], stdin=subprocess.PIPE)
    t0 = 0; info = []
    for r in R:
        nw = 0
        for i, n, fr in frames(r, mp):
            if i >= 0: ff.stdin.write(fr.tobytes()); nw += 1; continue
            cv2.imwrite(f"{out}/tiles/{r['id'].replace('/', '_')}.jpg", cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
        info.append(dict(id=r["id"], pack=k, off=t0 * STEP / 10, dur=nw * STEP / 10)); t0 += nw
    ff.stdin.close(); ff.wait(); os.replace(tmp, mp4); json.dump(info, open(man, "w")); return info


def item(r, tags, pk, sheet, cell):
    meta = json.load(open(f"{r['dir']}/meta.json")); L = json.load(open(f"{r['dir']}/annotation.json")); c = lambda *ks: sum(len(L.get(k, [])) for k in ks)
    fwd, turn = c("fly_forward_small", "fly_forward_large"), c("turn_left_small", "turn_left_large", "turn_right_small", "turn_right_large")
    vert, lat = c("fly_up", "fly_down"), c("fly_left", "fly_right", "fly_backward"); tot = max(1, fwd + turn + vert + lat)
    return dict(id=r["id"], name=r["id"].replace("/", "_"), groups=tags.get(r["id"], []), pack=pk["pack"], off=pk["off"], dur=pk["dur"], sheet=sheet, cell=cell,
                coverage=r["coverage"], flight_m=r["flight_m"], collisions=r["collisions"], vertical=r["vertical_frame_frac"], frames=r["frames"], seed=r["seed"],
                fwd=round(fwd / tot, 4), turn=round(turn / tot, 4), vert=round(vert / tot, 4), lat=round(lat / tot, 4),
                escapes=meta.get("escapes", 0), pose_err=meta.get("final_pose_err_m"), fence=meta.get("fence_blocks", 0))


def build(data, out, geo, pack, procs, crf, limit=0, codec="h264"):
    for s in ("pk", "ps", "tiles"): os.makedirs(f"{out}/{s}", exist_ok=True)
    R = rows(data); tags = groups_of(R); R = R[:limit] if limit else R
    chunks = [R[i:i + pack] for i in range(0, len(R), pack)]
    with Pool(procs) as p: infos = [x for res in p.imap(render_pack, [(k, c, geo, out, crf, codec) for k, c in enumerate(chunks)]) for x in res]
    PK = {x["id"]: x for x in infos}; per = SC * SR; items = []
    for j, r in enumerate(R):
        sheet, cell = divmod(j, per); items.append(item(r, tags, PK[r["id"]], sheet, cell))
    for s in range(0, len(R), per):  # poster sprite sheets
        img = np.zeros((SR * TH, SC * TW, 3), np.uint8)
        for c, r in enumerate(R[s:s + per]):
            t = cv2.imread(f"{out}/tiles/{r['id'].replace('/', '_')}.jpg"); y, x = divmod(c, SC); img[y * TH:(y + 1) * TH, x * TW:(x + 1) * TW] = t
        cv2.imwrite(f"{out}/ps/sheet_{s // per:02d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
    paths = {}
    for r in R:  # camera path in centimetres, every 5th frame, for the 3D view
        Q = np.array(json.load(open(f"{r['dir']}/posture.txt")), float)[::5]; paths[r["id"]] = np.round(np.stack([Q[:, 0], Q[:, 2] + 1.5, Q[:, 1]], 1) * 100).astype(int).reshape(-1).tolist()
    sizes = sorted(glob.glob(f"{out}/pk/pack_*.{EXT[codec]}")) + sorted(glob.glob(f"{out}/ps/*.jpg"))
    json.dump(dict(source=f"build_gallery_v5.py from {os.path.basename(os.path.normpath(data))}", n_total=len(R), video_ext=EXT[codec], codec=codec, tile=[TW, TH, SC, SR], items=items,
                   bytes=int(sum(os.path.getsize(f) for f in sizes))), open(f"{out}/manifest.json", "w"))
    json.dump(paths, open(f"{out}/paths.json", "w"), separators=(",", ":"))
    return len(R), sum(os.path.getsize(f) for f in sizes)


def write_page(out, template=os.path.join(HERE, "gallery_v5_template.html")):
    """index.html with the manifest, every trajectory path and every archived heatmap version (oldest first) inlined."""
    man = json.load(open(f"{out}/manifest.json")); man["paths"] = json.load(open(f"{out}/paths.json"))
    if os.path.exists(f"{out}/packs.json"): man["packs"] = json.load(open(f"{out}/packs.json"))  # pack index -> artifact asset id (filled after upload)
    arch = os.path.join(HERE, "../../../docs/exploration/heatmaps")
    H = {os.path.basename(os.path.dirname(f)): json.load(open(f)) for f in sorted(glob.glob(f"{arch}/*/heatmap3d.json"), key=lambda f: nat(os.path.basename(os.path.dirname(f))))}
    for h in H.values(): h.pop("paths", None)  # the page carries every trajectory's path itself
    h3 = json.dumps(H, separators=(",", ":")) if H else "null"
    page = open(template).read().replace("/*MANIFEST*/", json.dumps(man, ensure_ascii=False, separators=(",", ":"))).replace("/*HEAT3D*/", h3)
    open(f"{out}/index.html", "w").write(page)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--geo", default="results_v4/geo/gibson_Adrian.npz"); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--pack", type=int, default=8)
    ap.add_argument("--crf", type=int, default=32); ap.add_argument("--codec", default="h264", choices=list(CODEC)); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--page_only", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if not a.page_only:
        n, b = build(a.data, a.out, a.geo, a.pack, a.procs, a.crf, a.limit, a.codec); print(n, "trajectories", round(b / 1e6, 1), "MB of media")
    write_page(a.out)
