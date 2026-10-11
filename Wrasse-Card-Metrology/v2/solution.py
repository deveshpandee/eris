"""Wrasse Card Metrology - solution v2 (robust to new islands, seasons and lighting).

Usage: python3 solution.py <public_dir> <submission_out>

Pipeline
1. Every photograph is measured with classical image processing (numpy/scipy only):
   - the reference card's solid dark rectangle (fixed physical size) is found by thresholding,
     connected components and a rectangle-shape test, then its length and width are measured
     with sub-pixel edge profiles on the full-resolution image;
   - the fish is segmented as the largest region darker or more saturated than the local board,
     its principal axis gives the pixel length and an aligned, size-normalised crop.
2. length_mm = (fish pixels / rectangle pixels) x a scale, corrected by a gradient-boosting model
   of the log-residual fitted on the training captures; the two sides are combined.
3. morph and spawning: logistic regression + gradient boosting on colour and shape features of the
   aligned crops; answers are chosen to maximise the expected (class-balanced) credit.
4. landmarks: ridge regression from the aligned crop to the fifteen points in crop coordinates,
   fitted on train_landmarks.csv, mapped back to photograph pixels.

Compliance with the challenge rules
- Data: reads only the files in <public_dir> (photos.parquet, train.csv, test.csv,
  train_targets.csv, train_landmarks.csv); no external copies of the photographs, no records of
  these fish.
- Models: every model is fitted here, from the provided training captures; no pretrained weights,
  no downloads, no network access.
- No logic depends on ids, row order or file order; test targets are never read.
- Determinism: fixed seeds (random_state=0), a fixed pool of 4 worker processes; every loop and
  every boosting run has a fixed bound.
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from multiprocessing import Pool
from scipy import ndimage as ndi
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAMES = ['snout', 'eyes_snout', 'eyes_top', 'eyes_opercular', 'eyes_pectoral', 'top_opercular',
         'medium_opercular', 'bottom_opercular', 'pectoral_top', 'pectoral_bottom', 'pelvic', 'anal_fin',
         'caudal_top', 'caudal_bottom', 'black_point']
ST = ndi.generate_binary_structure(2, 1)
CW, CH = 192, 64          # aligned crop: along the fish axis x across
RECT_AR = 2.115           # length/width of the card's dark rectangle (median measured on train)
N_WORKERS = 4
MORPHS = np.array(['f', 'm', 's'])


# ----------------------------------------------------------------------------------------------
# 1. per-photograph measurement
# ----------------------------------------------------------------------------------------------
def decode(b):
    img = np.asarray(Image.open(io.BytesIO(b)).convert('RGB'), dtype=np.float32)
    return img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]


def pca_axes(xs, ys):
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    q = pts - c
    _, evec = np.linalg.eigh(q.T @ q / len(q))
    ax = evec[:, 1]
    if ax[0] < 0:
        ax = -ax
    return c, ax, q


def rect_search(g2):
    """Card's dark rectangle on the half-resolution grey image."""
    h, w = g2.shape
    bg = np.percentile(g2, 90)
    best = None
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85):
        dark = g2 < t * bg
        for op in (0, 1):
            dk = ndi.binary_opening(dark, ST, iterations=op) if op else dark
            lab, n = ndi.label(dk)
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())[1:]
            objs = ndi.find_objects(lab)
            for i in np.argsort(-sizes)[:25]:
                if sizes[i] < 200:
                    break
                sl = objs[i]
                if sl[0].start <= 0 or sl[1].start <= 0 or sl[0].stop >= h or sl[1].stop >= w:
                    continue
                ys, xs = np.nonzero(lab[sl] == i + 1)
                c, ax, q = pca_axes(xs + sl[1].start, ys + sl[0].start)
                pr = q @ ax
                pr2 = q @ np.array([-ax[1], ax[0]])
                L = pr.max() - pr.min() + 1
                W = pr2.max() - pr2.min() + 1
                fill = sizes[i] / (L * W)
                ar = L / max(W, 1.0)
                if fill > 0.8 and 1.6 < ar < 2.8:
                    score = fill - 0.5 * abs(np.log(ar / 2.1))
                    if best is None or score > best[0]:
                        best = (score, c, ax, L, W, fill, t)
    return best


def edge_extent(prof):
    """Sub-pixel length of the dark run at the centre of a 1-D profile."""
    n = len(prof)
    inside = np.median(prof[n // 2 - n // 8: n // 2 + n // 8 + 1])
    outside = 0.5 * (np.median(prof[: n // 8 + 1]) + np.median(prof[-(n // 8) - 1:]))
    if outside - inside < 10:
        return np.nan
    thr = 0.5 * (inside + outside)
    below = prof < thr
    mid = n // 2
    if not below[mid]:
        return np.nan
    i = mid
    while i > 0 and below[i - 1]:
        i -= 1
    j = mid
    while j < n - 1 and below[j + 1]:
        j += 1
    if i == 0 or j == n - 1:
        return np.nan
    a = i - 1 + (prof[i - 1] - thr) / max(prof[i - 1] - prof[i], 1e-6)
    b = j + (thr - prof[j]) / max(prof[j + 1] - prof[j], 1e-6)
    return b - a


def rect_refine(g, c, ax, L, W):
    perp = np.array([-ax[1], ax[0]])
    out = []
    for u, v, Lu, Lv in ((ax, perp, L, W), (perp, ax, W, L)):
        ts = np.arange(-0.8 * Lu, 0.8 * Lu, 0.5)
        profs = []
        for o in np.linspace(-0.3 * Lv, 0.3 * Lv, 9):
            xs = c[0] + ts * u[0] + o * v[0]
            ys = c[1] + ts * u[1] + o * v[1]
            profs.append(ndi.map_coordinates(g, [ys, xs], order=1, mode='nearest'))
        out.append(edge_extent(np.median(np.array(profs), 0)) * 0.5)
    return out


def fish_mask(img2, rect):
    """Fish on the half-resolution image: darker or more saturated than the local board."""
    g = img2.mean(2)
    mx = img2.max(2)
    mn = img2.min(2)
    sat = (mx - mn) / (mx + 1.0)
    h, w = g.shape
    gs = g[::8, ::8]
    ss = sat[::8, ::8]
    bgl = ndi.zoom(ndi.percentile_filter(gs, 85, size=11), (h / gs.shape[0], w / gs.shape[1]), order=1)[:h, :w]
    bgs = ndi.zoom(ndi.percentile_filter(ss, 30, size=11), (h / ss.shape[0], w / ss.shape[1]), order=1)[:h, :w]
    m = ((1 - g / np.maximum(bgl, 1)) > 0.18) | (sat - bgs > 0.18)
    m = ndi.binary_opening(m, ST, iterations=1)
    m = ndi.binary_closing(m, ST, iterations=3)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    best, bscore = None, -1
    for i in np.argsort(-sizes)[:6]:
        if sizes[i] < 0.002 * h * w:
            break
        comp = lab == i + 1
        if rect is not None:
            ix = int(np.clip(rect[1][0], 0, w - 1))
            iy = int(np.clip(rect[1][1], 0, h - 1))
            if comp[iy, ix]:
                continue
        mc = img2[comp].mean(0)
        brd = comp[0].sum() + comp[-1].sum() + comp[:, 0].sum() + comp[:, -1].sum()
        score = sizes[i] * (0.3 if mc[2] - mc[0] > 8 else 1.0) * (0.3 if brd > 0.25 * np.sqrt(sizes[i]) else 1.0)
        if score > bscore:
            best, bscore = comp, score
    return best


def measure(args):
    pid, blob = args
    img = decode(blob)
    g = img.mean(2)
    img2 = img.reshape(img.shape[0] // 2, 2, img.shape[1] // 2, 2, 3).mean((1, 3))
    out = dict(photo_id=pid)
    r = rect_search(img2.mean(2))
    for k in ('rL', 'rW', 'rfill', 'rcx', 'rcy', 'rang'):
        out[k] = np.nan
    if r is not None:
        _, c2, ax, L2, W2, fill, _ = r
        c = c2 * 2 + 0.5
        Lf, Wf = rect_refine(g, c, ax, L2 * 2, W2 * 2)
        out.update(rL=Lf, rW=Wf, rfill=fill, rcx=c[0], rcy=c[1], rang=np.degrees(np.arctan2(ax[1], ax[0])))
    comp = fish_mask(img2, r)
    crop = np.zeros((CH, CW, 3), np.uint8)
    cmask = np.zeros((CH, CW), np.uint8)
    T = np.full(6, np.nan)
    for k in ['fl', 'fw', 'flc', 'farea', 'fcx', 'fcy', 'fang', 'touch', 'fr', 'fg', 'fb'] + [f'wp{k}' for k in range(20)]:
        out[k] = np.nan
    if comp is not None:
        ys, xs = np.nonzero(comp)
        xs = xs * 2 + 0.5
        ys = ys * 2 + 0.5
        c, ax, q = pca_axes(xs, ys)
        perp = np.array([-ax[1], ax[0]])
        pr = q @ ax
        pr2 = q @ perp
        lo, hi = pr.min() - 1, pr.max() + 1
        lo2, hi2 = pr2.min() - 1, pr2.max() + 1
        bins = np.clip(((pr - lo) / (hi - lo) * 20).astype(int), 0, 19)
        for k in range(20):
            sel = pr2[bins == k]
            out[f'wp{k}'] = (np.ptp(sel) if len(sel) else 0.0) / (hi - lo)
        nb = 30
        bb = np.clip(((pr - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
        cnt = np.bincount(bb, minlength=nb)
        mids = np.array([0.5 * (pr2[bb == k].min() + pr2[bb == k].max()) if cnt[k] else np.nan for k in range(nb)])
        ok = np.isfinite(mids)
        xb = (lo + (np.arange(nb) + 0.5) / nb * (hi - lo))[ok]
        mids = mids[ok]
        flc = np.sum(np.hypot(np.diff(xb), np.diff(mids))) + (xb[0] - lo) + (hi - xb[-1])
        pxc = img2[comp]
        out.update(fl=hi - lo, fw=hi2 - lo2, flc=flc, farea=len(xs) * 4.0, fcx=c[0], fcy=c[1],
                   fang=np.degrees(np.arctan2(ax[1], ax[0])),
                   touch=float(comp[0].sum() + comp[-1].sum() + comp[:, 0].sum() + comp[:, -1].sum()),
                   fr=pxc[:, 0].mean(), fg=pxc[:, 1].mean(), fb=pxc[:, 2].mean())
        sc = (hi - lo) / (CW - 8)
        mid = 0.5 * (lo + hi)
        u = (np.arange(CW) - CW / 2 + 0.5) * sc + mid
        v = (np.arange(CH) - CH / 2 + 0.5) * sc
        UU, VV = np.meshgrid(u, v)
        X = c[0] + UU * ax[0] + VV * perp[0]
        Y = c[1] + UU * ax[1] + VV * perp[1]
        for ch in range(3):
            crop[..., ch] = np.clip(ndi.map_coordinates(img[..., ch], [Y, X], order=1, mode='nearest'), 0, 255)
        cmask = ndi.map_coordinates(comp.astype(np.float32), [(Y - 0.5) / 2, (X - 0.5) / 2], order=0,
                                    mode='constant').astype(np.uint8)
        # crop pixel (row i, col j) -> photo: x = T0 + T1*j + T2*i, y = T3 + T4*j + T5*i
        T = np.array([c[0] + u[0] * ax[0] + v[0] * perp[0], sc * ax[0], sc * perp[0],
                      c[1] + u[0] * ax[1] + v[0] * perp[1], sc * ax[1], sc * perp[1]])
    return out, crop, cmask, T


# ----------------------------------------------------------------------------------------------
# 2. orientation-normalised crops (head left, back up)
# ----------------------------------------------------------------------------------------------
class Photos:
    def __init__(self, meas, crops, masks, Ts):
        self.m = meas
        self.crops, self.masks, self.T = crops, masks, Ts
        self.ix = {p: i for i, p in enumerate(meas.photo_id)}

    def row(self, pid):
        return self.m.iloc[self.ix[pid]]

    def flips(self, pid, side):
        r = self.row(pid)
        upside = bool(np.isfinite(r.rcy) and np.isfinite(r.fcy) and r.rcy > r.fcy)   # card below the fish
        return (side == 'right') ^ upside, upside

    def norm_crop(self, pid, side):
        i = self.ix[pid]
        c = self.crops[i].astype(np.float32)
        k = self.masks[i].astype(np.float32)
        # white balance: divide by the board colour around the fish (non-fish pixels of the crop)
        bgpx = c[k == 0]
        board = np.median(bgpx, 0) if len(bgpx) > 100 else np.median(c.reshape(-1, 3), 0)
        c = np.clip(c / np.maximum(board, 1.0)[None, None] * 230.0, 0, 255)
        fx, fy = self.flips(pid, side)
        if fx:
            c, k = c[:, ::-1], k[:, ::-1]
        if fy:
            c, k = c[::-1], k[::-1]
        return c, k

    def to_crop(self, pid, side, pts):
        T = self.T[self.ix[pid]]
        A = np.array([[T[1], T[2]], [T[4], T[5]]])
        ji = np.linalg.solve(A, (pts - np.array([T[0], T[3]])).T).T
        fx, fy = self.flips(pid, side)
        if fx:
            ji[:, 0] = CW - 1 - ji[:, 0]
        if fy:
            ji[:, 1] = CH - 1 - ji[:, 1]
        return ji

    def from_crop(self, pid, side, ji):
        ji = ji.copy()
        fx, fy = self.flips(pid, side)
        if fx:
            ji[:, 0] = CW - 1 - ji[:, 0]
        if fy:
            ji[:, 1] = CH - 1 - ji[:, 1]
        T = self.T[self.ix[pid]]
        return np.stack([T[0] + T[1] * ji[:, 0] + T[2] * ji[:, 1], T[3] + T[4] * ji[:, 0] + T[5] * ji[:, 1]], 1)

    def has_crop(self, pid):
        return bool(np.isfinite(self.T[self.ix[pid]][0]))


# ----------------------------------------------------------------------------------------------
# 3. features
# ----------------------------------------------------------------------------------------------
# shape and relative-position features only: nothing that identifies a capture session (card size or
# absolute position, fish colour), so the correction transfers to new islands and seasons
LEN_FEATS = ['fwfl', 'flcfl', 'ar', 'rfill', 'dang', 'side', 'rdist', 'fdist', 'dx', 'dy'] + [f'wp{k}' for k in range(20)]


def side_table(P, pids, side):
    R = pd.DataFrame([P.row(p) for p in pids]).reset_index(drop=True)
    R['ar'] = R.rL / R.rW
    R['ok'] = (R.touch == 0) & ((R.ar - RECT_AR).abs() < 0.2) & R.rL.notna() & R.fl.notna()
    R['base'] = R.fl / R.rL
    R['fwfl'] = R.fw / R.fl
    R['flcfl'] = R.flc / R.fl
    R['dang'] = ((R.fang - R.rang + 90) % 180) - 90
    R['dx'] = R.fcx - R.rcx
    R['dy'] = R.fcy - R.rcy
    R['rdist'] = np.hypot(R.rcx - 512, R.rcy - 384)
    R['fdist'] = np.hypot(R.fcx - 512, R.fcy - 384)
    R['side'] = int(side == 'right')
    return R


def combine_sides(est, L, R, fallback):
    n = len(L)
    out = np.full(n, fallback)
    for i in range(n):
        c, a = [], []
        for t, S in ((est[0], L), (est[1], R)):
            if S.ok[i] and np.isfinite(t[i]):
                c.append(t[i])
                a.append(abs(S.ar[i] - RECT_AR))
        if len(c) == 2 and abs(c[0] / c[1] - 1) > 0.05:
            c = [c[int(np.argmin(a))]]
        if not c:   # no reliable side: use any finite estimate
            c = [t[i] for t in est if np.isfinite(t[i]) and 60 < t[i] < 300]
        if c:
            out[i] = float(np.mean(c))
    return out


def color_feats(P, df):
    X = []
    for lp, rp in zip(df.left_photo, df.right_photo):
        fs = []
        for pid, s in ((lp, 'left'), (rp, 'right')):
            c, k = P.norm_crop(pid, s)
            kk = k > 0
            px = c[kk] if kk.sum() > 50 else c.reshape(-1, 3)
            mx = px.max(1)
            mn = px.min(1)
            hsv = np.stack([(mx - mn) / (mx + 1), mx / 255], 1)
            rgb = px / (px.sum(1, keepdims=True) + 1)
            fs.extend(np.percentile(px, [10, 50, 90], axis=0).ravel() / 255)
            fs.extend(np.percentile(rgb, [10, 50, 90], axis=0).ravel())
            fs.extend(np.percentile(hsv, [10, 50, 90], axis=0).ravel())
            for gy in range(2):
                for gx in range(4):
                    sl = (slice(gy * 32, gy * 32 + 32), slice(gx * 48, gx * 48 + 48))
                    k2 = kk[sl]
                    cc = c[sl][k2] if k2.sum() > 20 else c[sl].reshape(-1, 3)
                    fs.extend(cc.mean(0) / 255)
        X.append(fs)
    return np.array(X)


def small_img(P, df, w=48, h=16):
    X = []
    for lp, rp in zip(df.left_photo, df.right_photo):
        fs = []
        for pid, s in ((lp, 'left'), (rp, 'right')):
            c, _ = P.norm_crop(pid, s)
            im = Image.fromarray(np.ascontiguousarray(c).astype(np.uint8)).resize((w, h), Image.BILINEAR)
            fs.append(np.asarray(im, np.float32).ravel() / 255)
        X.append(np.concatenate(fs))
    return np.array(X)


def lm_input(P, pid, side, w=96, h=32):
    c, _ = P.norm_crop(pid, side)
    a = np.asarray(Image.fromarray(np.ascontiguousarray(c).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                   np.float32) / 255
    g = a.mean(2)
    gx = np.diff(g, axis=1, append=g[:, -1:])
    gy = np.diff(g, axis=0, append=g[-1:])
    return np.concatenate([a.ravel(), gx.ravel(), gy.ravel()])


# ----------------------------------------------------------------------------------------------
# 4. decisions that maximise the expected credit
# ----------------------------------------------------------------------------------------------
MORPH_CREDIT = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])           # [answer, truth] for f, m, s
SPAWN_CREDIT = np.array([[1 - abs(a - t) / 2 for t in range(3)] for a in range(3)])


def decide(P, class_w, credit):
    return ((P * class_w) @ credit.T).argmax(1)


def fmt_points(pts):
    return json.dumps([[round(float(x), 1), round(float(y), 1)] for x, y in pts], separators=(',', ':'))


# ----------------------------------------------------------------------------------------------
# 5. main
# ----------------------------------------------------------------------------------------------
def resolve_paths():
    if len(sys.argv) != 3:
        sys.exit('usage: python3 solution.py <public_dir> <submission_out>')
    pub, out = sys.argv[1], sys.argv[2]
    if os.path.isdir(out):
        out = os.path.join(out, 'submission.csv')
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    return pub, out


def run(pub):
    train = pd.read_csv(os.path.join(pub, 'train.csv'))
    test = pd.read_csv(os.path.join(pub, 'test.csv'))
    targets = pd.read_csv(os.path.join(pub, 'train_targets.csv'))
    lmk = pd.read_csv(os.path.join(pub, 'train_landmarks.csv'))
    train = train.merge(targets[['item_id', 'length_mm', 'morph', 'spawning']], on='item_id', how='left')
    assert train.length_mm.notna().all()

    need = set(train.left_photo) | set(train.right_photo) | set(test.left_photo) | set(test.right_photo) | set(lmk.photo_id)
    photos = pq.read_table(os.path.join(pub, 'photos.parquet'), columns=['photo_id', 'image']).to_pandas()
    photos = photos[photos.photo_id.isin(need)].sort_values('photo_id').reset_index(drop=True)
    with Pool(N_WORKERS) as pool:
        res = pool.map(measure, list(zip(photos.photo_id, photos.image)), chunksize=4)
    del photos
    P = Photos(pd.DataFrame([r[0] for r in res]), np.stack([r[1] for r in res]),
               np.stack([r[2] for r in res]), np.stack([r[3] for r in res]))

    # ---------------- length ----------------
    y_len = train.length_mm.values.astype(float)
    med = float(np.median(y_len))
    Ltr = side_table(P, train.left_photo, 'left')
    Rtr = side_table(P, train.right_photo, 'right')
    Lte = side_table(P, test.left_photo, 'left')
    Rte = side_table(P, test.right_photo, 'right')
    Xs = pd.concat([Ltr, Rtr], ignore_index=True)
    ys = np.concatenate([y_len, y_len])
    fit = Xs.ok.values
    t = np.log(ys[fit] / Xs.base.values[fit])
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20,
                                        loss='absolute_error', random_state=0)
    reg.fit(Xs.loc[fit, LEN_FEATS], t)

    def est_len(L, R):
        e = [np.exp(reg.predict(S[LEN_FEATS])) * S.base.values for S in (L, R)]
        return combine_sides(e, L, R, med)

    lo_len, hi_len = y_len.min() - 10, y_len.max() + 10   # implausible measurements fall back to the median

    def plausible(v):
        return np.where((v >= lo_len) & (v <= hi_len), v, med)

    len_tr = plausible(est_len(Ltr, Rtr))      # in-sample, used only as a classifier feature
    len_te = plausible(est_len(Lte, Rte))

    # ---------------- morph / spawning ----------------
    y_m = train.morph.map({'f': 0, 'm': 1, 's': 2}).values
    y_s = train.spawning.values.astype(int)
    shape_cols = ['fwfl'] + [f'wp{k}' for k in range(20)]

    def cls_feats(df, L, R, ln):
        Xc = color_feats(P, df)
        Xi = small_img(P, df)
        Xh = np.column_stack([L[shape_cols].values, R[shape_cols].values, ln])
        return np.column_stack([Xc, Xi, ln]), np.column_stack([Xc, Xh])

    XA_tr, XG_tr = cls_feats(train, Ltr, Rtr, len_tr)
    XA_te, XG_te = cls_feats(test, Lte, Rte, len_te)
    XA_tr, XA_te = np.nan_to_num(XA_tr), np.nan_to_num(XA_te)

    def fit_proba(y):
        lr = make_pipeline(StandardScaler(), PCA(60, random_state=0),
                           LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced'))
        lr.fit(XA_tr, y)
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=0)
        gb.fit(XG_tr, y)
        return 0.5 * lr.predict_proba(XA_te) + 0.5 * gb.predict_proba(XG_te)

    morph_pred = decide(fit_proba(y_m), 1.0 / np.bincount(y_m, minlength=3), MORPH_CREDIT)
    spawn_pred = decide(fit_proba(y_s), 1.0 / np.bincount(y_s, minlength=3), SPAWN_CREDIT)

    # ---------------- landmarks ----------------
    lmk['P'] = lmk.points.map(json.loads)
    lmk = lmk[lmk.P.map(lambda d: all(n in d for n in NAMES))]
    lmk = lmk[lmk.photo_id.map(P.has_crop)].sort_values('photo_id').reset_index(drop=True)
    true = np.array([[d[n] for n in NAMES] for d in lmk.P], float)
    J = np.array([P.to_crop(p, s, true[i]) for i, (p, s) in enumerate(zip(lmk.photo_id, lmk.side))])
    XL = np.array([lm_input(P, p, s) for p, s in zip(lmk.photo_id, lmk.side)])
    lm_model = make_pipeline(StandardScaler(), Ridge(alpha=3000))
    lm_model.fit(XL, J.reshape(len(J), -1))
    mean_abs = {s: true[(lmk.side == s).values].mean(0) for s in ('left', 'right')}

    def points(pid, side):
        if not P.has_crop(pid):
            return mean_abs[side]
        ji = lm_model.predict(lm_input(P, pid, side)[None])[0].reshape(15, 2)
        return P.from_crop(pid, side, ji)

    sub = pd.DataFrame({
        'item_id': test.item_id.values,
        'length_mm': np.round(len_te, 1),
        'morph': MORPHS[morph_pred],
        'spawning': spawn_pred,
        'left_points': [fmt_points(points(p, 'left')) for p in test.left_photo],
        'right_points': [fmt_points(points(p, 'right')) for p in test.right_photo],
    })
    return sub


def check_submission(sub, pub):
    sample = pd.read_csv(os.path.join(pub, 'sample_submission.csv'))
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample) and set(sub.item_id) == set(sample.item_id)
    assert np.isfinite(sub.length_mm).all()
    assert sub.morph.isin(['f', 'm', 's']).all() and sub.spawning.isin([0, 1, 2]).all()


if __name__ == '__main__':
    np.random.seed(0)
    pub_dir, out_path = resolve_paths()
    submission = run(pub_dir)
    check_submission(submission, pub_dir)
    submission.to_csv(out_path, index=False)
