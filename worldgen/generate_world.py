#!/usr/bin/env python3
"""
WILDHEART world generator: a massive open-world terrain for Unreal Engine 5.

Builds an 8 x 8 km (default) continent at 1 m per pixel:
  * six hand-steered biomes blended together (giant glowing rainforest, plains,
    red mesas, frozen peaks, volcanic wastes, swamp) plus an ocean coast
  * hydraulic erosion, priority-flood lakes and carved river networks
  * Avatar-style rock spires, a volcano with a crater, a Mother Tree basin
  * Unreal landscape outputs: 16-bit heightmap + 8-bit paint-layer weightmaps
  * world.json with import settings, POIs, colossal-tree and floating-island
    positions, and dino spawn zones, which ue/import_world.py turns into actors

Usage:  python generate_world.py --seed 7 --size 8129 --out output
        python generate_world.py --size 2017            (fast preview build)
Valid Unreal landscape sizes: 1009, 2017, 4033, 8129 (8129 = 8.1 km at 1 m/px).
"""
import argparse, heapq, json, math, os, time
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

MACRO = 1025            # resolution for the large-scale simulation (erosion, rivers)
SEA = 0.0               # sea level in metres
H_MIN, H_MAX = -120.0, 1800.0   # metres stored in the 16-bit heightmap range

BIOMES = [  # key, display name, preview colour
    ('jungle',   'Glowwood Rainforest', (34, 120, 96)),
    ('plains',   'Thunder Plains',      (138, 166, 74)),
    ('mesa',     'Redstone Mesas',      (184, 98, 58)),
    ('frozen',   'Frostfang Peaks',     (214, 226, 238)),
    ('volcanic', 'Ember Wastes',        (74, 58, 54)),
    ('swamp',    'Mirefen Swamp',       (72, 96, 62)),
]
BI = {b[0]: i for i, b in enumerate(BIOMES)}
SPECIES = {
    'jungle':   ['Raptor', 'Parasaur', 'Brachiosaurus', 'Dilophosaur', 'Glowtail', 'Megatherium'],
    'plains':   ['Triceratops', 'Parasaur', 'Stegosaurus', 'Carnotaurus', 'Gallimimus', 'Ankylosaurus'],
    'mesa':     ['Allosaurus', 'Pteranodon', 'Pachycephalosaur', 'Thorny Dragon', 'Raptor'],
    'frozen':   ['Mammoth', 'Direwolf', 'Woolly Rhino', 'Yutyrannus', 'Argentavis'],
    'volcanic': ['Rex', 'Magmasaur', 'Carnotaurus', 'Quetzal', 'Ankylosaurus'],
    'swamp':    ['Spinosaurus', 'Sarco', 'Baryonyx', 'Diplocaulus', 'Beelzebufo'],
}

def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)
def smoothstep(e0, e1, x): t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

def value_noise(rng, shape, cells, order=3):
    """Smooth random field: a random grid of `cells` points upsampled to `shape`."""
    gy, gx = max(2, int(cells * shape[0] / max(shape))), max(2, int(cells * shape[1] / max(shape)))
    g = rng.random((gy + 3, gx + 3)).astype(np.float32)
    z = ndi.zoom(g, ((shape[0] + 3 * shape[0] / gy) / g.shape[0], (shape[1] + 3 * shape[1] / gx) / g.shape[1]), order=order)
    oy, ox = int(shape[0] / gy), int(shape[1] / gx)
    return z[oy:oy + shape[0], ox:ox + shape[1]]

def fbm(rng, shape, cells, octaves, gain=.5, order=3):
    out = np.zeros(shape, np.float32); amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += value_noise(rng, shape, cells * 2 ** o, order) * amp; tot += amp; amp *= gain
    return out / tot

def ridged(rng, shape, cells, octaves):
    out = np.zeros(shape, np.float32); amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = 1 - np.abs(value_noise(rng, shape, cells * 2 ** o) * 2 - 1); out += n * n * amp; tot += amp; amp *= .5
    return out / tot

def warp(rng, field, strength, cells):
    n = field.shape[0]
    wx = (fbm(rng, field.shape, cells, 3) - .5) * strength * n
    wy = (fbm(rng, field.shape, cells, 3) - .5) * strength * n
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    return ndi.map_coordinates(field, [yy + wy, xx + wx], order=1, mode='nearest')

# ------------------------------------------------------------------ macro terrain
def biome_weights(rng, n):
    """Soft biome membership from warped Voronoi seeds whose biome follows world layout."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / (n - 1)
    wu = xx + (fbm(rng, (n, n), 4, 3) - .5) * .22
    wv = yy + (fbm(rng, (n, n), 4, 3) - .5) * .22
    seeds = []
    k = 8
    for j in range(k):
        for i in range(k):
            u, v = (i + .2 + rng.random() * .6) / k, (j + .2 + rng.random() * .6) / k
            if v < .24: b = 'frozen'
            elif u > .68 and v < .52: b = 'volcanic'
            elif u > .6: b = 'mesa'
            elif u < .32 and v > .64: b = 'swamp'
            elif u < .36: b = 'plains'
            else: b = 'jungle'
            seeds.append((u, v, BI[b]))
    d = np.stack([np.hypot(wu - u, wv - v) for u, v, _ in seeds])
    dmin = d.min(0)
    w = np.exp(-(d - dmin) / .028)
    W = np.zeros((len(BIOMES), n, n), np.float32)
    for s, (_, _, b) in enumerate(seeds): W[b] += w[s]
    W /= W.sum(0, keepdims=True)
    return W, xx, yy

def macro_height(rng, n):
    W, u, v = biome_weights(rng, n)
    sh = (n, n)
    rolling = fbm(rng, sh, 6, 5)
    hills = fbm(rng, sh, 14, 5)
    rid = ridged(rng, sh, 5, 6)
    rid2 = ridged(rng, sh, 9, 5)
    H = {}
    H['plains'] = 30 + rolling * 50 + hills * 25
    H['jungle'] = 45 + rolling * 90 + hills * 70 + rid2 * 60
    terr = fbm(rng, sh, 7, 4) * 6
    terr = np.floor(terr) + smoothstep(.75, 1, terr - np.floor(terr))
    canyon = 1 - np.clip(np.abs(fbm(rng, sh, 9, 3) - .5) * 14, 0, 1)
    H['mesa'] = 110 + terr * 55 - canyon * 120 + hills * 15
    H['frozen'] = 260 + rid * 1150 + hills * 80
    H['volcanic'] = 140 + rid2 * 260 + hills * 40
    H['swamp'] = 4 + hills * 10 + rolling * 8
    h = sum(W[BI[k]] * H[k] for k in H)
    # volcano: one huge cone with a crater inside the Ember Wastes
    vw = W[BI['volcanic']]
    inner = np.zeros_like(vw); m = int(n * .2); inner[m:n - m, m:n - m] = 1
    cy, cx = np.unravel_index(np.argmax(ndi.gaussian_filter(vw, n / 30) * inner), vw.shape)
    r = np.hypot(np.arange(n)[:, None] - cy, np.arange(n)[None, :] - cx) / n
    cone = np.clip(1 - r / .13, 0, 1) ** 1.6 * 1250
    crater = smoothstep(.022, .0, r) * 380
    h += cone - crater
    # coastline: ocean wraps the south and west; impassable mountain wall north and east
    coast = np.minimum(u * 1.25, (1 - v) * 1.25) + (fbm(rng, sh, 5, 4) - .5) * .25
    land = smoothstep(.04, .12, coast)
    h = h * land + (-90 + fbm(rng, sh, 8, 3) * 40) * (1 - land)
    beach = smoothstep(.12, .07, coast) * land
    h = h * (1 - beach * .85) + beach * 2.5
    wall = np.maximum(smoothstep(.06, 0, v), smoothstep(.94, 1, u)) * (fbm(rng, sh, 12, 3) * .5 + .75)
    h += wall * 900
    return h.astype(np.float32), W, (cy / (n - 1), cx / (n - 1)), land

def erode(rng, h, cell, drops=420_000, steps=70, batch=70_000):
    """Vectorised particle hydraulic erosion (many droplets simulated in parallel)."""
    hs = h / cell  # work in cell units so slopes are real slopes
    n = hs.shape[0]
    inertia, cap_k, min_cap, dep_k, ero_k, evap, grav = .1, 6., .02, .25, .35, .025, 6.
    for b in range(drops // batch):
        x = rng.uniform(1, n - 2, batch); y = rng.uniform(1, n - 2, batch)
        dx = np.zeros(batch); dy = np.zeros(batch); vel = np.ones(batch); water = np.ones(batch); sed = np.zeros(batch)
        alive = np.ones(batch, bool)
        for _ in range(steps):
            xi = np.clip(x.astype(int), 0, n - 2); yi = np.clip(y.astype(int), 0, n - 2); fx = x - xi; fy = y - yi
            h00, h10, h01, h11 = hs[yi, xi], hs[yi, xi + 1], hs[yi + 1, xi], hs[yi + 1, xi + 1]
            gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy; gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
            hc = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy
            dx = dx * inertia - gx * (1 - inertia); dy = dy * inertia - gy * (1 - inertia)
            ln = np.hypot(dx, dy); still = ln < 1e-6
            ln[still] = 1; dx /= ln; dy /= ln
            nx, ny = x + dx, y + dy
            alive &= ~still & (nx > 1) & (ny > 1) & (nx < n - 2) & (ny < n - 2)
            nxi = np.clip(nx.astype(int), 0, n - 2); nyi = np.clip(ny.astype(int), 0, n - 2); nfx = nx - nxi; nfy = ny - nyi
            hn = (hs[nyi, nxi] * (1 - nfx) * (1 - nfy) + hs[nyi, nxi + 1] * nfx * (1 - nfy) + hs[nyi + 1, nxi] * (1 - nfx) * nfy + hs[nyi + 1, nxi + 1] * nfx * nfy)
            dh = hn - hc
            cap = np.clip(-dh * vel * water * cap_k, min_cap, 4.)
            deposit = np.where(dh > 0, np.minimum(dh, sed), np.where(sed > cap, (sed - cap) * dep_k, 0))
            erosion = np.where((dh <= 0) & (sed <= cap), np.minimum(np.minimum((cap - sed) * ero_k, -dh), .35), 0)
            amt = (deposit - erosion) * alive
            sed += (erosion - deposit) * alive
            for ox, oy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                np.add.at(hs, (yi + oy, xi + ox), amt * wgt)
            vel = np.minimum(np.sqrt(np.maximum(0, vel * vel - dh * grav)), 6.); water *= 1 - evap
            x, y = np.where(alive, nx, x), np.where(alive, ny, y)
            if not alive.any(): break
        log(f'  erosion batch {b + 1}/{drops // batch}')
    return ndi.gaussian_filter(hs, .6) * cell

def priority_flood(h, ocean):
    """Fill depressions (epsilon variant) so every cell drains to the ocean. Returns filled heights."""
    n = h.shape[0]; f = h.copy(); done = np.zeros(h.shape, bool); pq = []
    border = ocean.copy(); border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    for y, x in zip(*np.nonzero(border)): heapq.heappush(pq, (float(f[y, x]), int(y), int(x))); done[y, x] = True
    fl = f.tolist(); dn = done.tolist()
    nb = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while pq:
        z, y, x = heapq.heappop(pq)
        for oy, ox in nb:
            yy, xx = y + oy, x + ox
            if 0 <= yy < n and 0 <= xx < n and not dn[yy][xx]:
                dn[yy][xx] = True
                if fl[yy][xx] <= z: fl[yy][xx] = z + 1e-3
                heapq.heappush(pq, (fl[yy][xx], yy, xx))
    return np.array(fl, np.float32)

def flow_accumulation(f):
    n = f.shape[0]; pad = np.pad(f, 1, mode='edge')
    best = np.zeros_like(f); rec = np.full(f.shape, -1, np.int64)
    idx = np.arange(n * n).reshape(n, n)
    for oy, ox in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        nbh = pad[1 + oy:1 + oy + n, 1 + ox:1 + ox + n]
        drop = (f - nbh) / math.hypot(oy, ox)
        better = drop > best
        best[better] = drop[better]
        tgt = np.clip(idx + oy * n + ox, 0, n * n - 1)
        rec[better] = tgt[better]
    order = np.argsort(-f, axis=None).tolist(); recl = rec.ravel().tolist()
    acc = [1.0] * (n * n)
    for i in order:
        r = recl[i]
        if r >= 0: acc[r] += acc[i]
    return np.array(acc, np.float32).reshape(n, n)

# ------------------------------------------------------------------ main pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--size', type=int, default=8129)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'output'))
    ap.add_argument('--drops', type=int, default=420_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    N, n = a.size, MACRO
    world_m = N - 1                       # 1 m per pixel
    cell = world_m / (n - 1)
    t0 = time.time()

    log('macro terrain + biomes'); h, W, (vcy, vcx), land = macro_height(rng, n)
    log('hydraulic erosion'); h = erode(rng, h, cell, drops=a.drops)
    ocean = h < SEA
    log('lakes + rivers (priority flood, flow accumulation)')
    f = priority_flood(h + rng.random(h.shape).astype(np.float32) * .4, ocean)
    lake = (f - h > 1.5) & ~ocean
    lake = ndi.binary_opening(lake, iterations=1)
    # keep only pleasant lakes: big flooded basins become dry valleys drained by a breached outlet
    lab, nl = ndi.label(lake)
    if nl:
        area = ndi.sum(lake, lab, range(1, nl + 1)); deep = ndi.maximum(f - h, lab, range(1, nl + 1))
        keep = np.zeros(nl + 1, bool); keep[1:] = (area >= 12) & (area <= 2500) & (deep < 45)
        lake = keep[lab]
        f = np.where((f - h > 1.5) & ~lake & ~ocean, h + .01, f)
    acc = flow_accumulation(f)
    thr = 900
    river = (acc > thr) & ~ocean & ~lake & (h < 900)
    depth = np.where(river, np.clip(np.log(acc / thr) * 3.2 + 2.5, 0, 16), 0).astype(np.float32)
    valley = ndi.gaussian_filter(depth, 1.6) * 1.6
    h = h - np.maximum(depth, valley)
    h = np.where(lake, np.minimum(h, f - 1.5), h)
    water_level = np.where(lake, f, np.where(river, h + np.maximum(depth * .55, 1.2), SEA)).astype(np.float32)

    log(f'upscale to {N}x{N} (1 m/px)')
    z = N / n
    H = ndi.zoom(h, z, order=3)[:N, :N].astype(np.float32)
    Wf = [ndi.zoom(W[i], z, order=1)[:N, :N].astype(np.float32) for i in range(len(BIOMES))]
    rivF = ndi.zoom(ndi.gaussian_filter(river.astype(np.float32), .8), z, order=1)[:N, :N]
    lakeF = ndi.zoom(lake.astype(np.float32), z, order=1)[:N, :N]
    log('fine detail')
    rough = Wf[BI['frozen']] * 1.6 + Wf[BI['volcanic']] * 1.2 + Wf[BI['mesa']] * .9 + Wf[BI['jungle']] * .7 + .35
    H += (fbm(rng, (N, N), 160, 3, order=1) - .5) * 9 * rough
    H += (fbm(rng, (N, N), 900, 2, order=1) - .5) * 1.4

    # Avatar-style sheer rock spires in the rainforest
    log('rock spires')
    yy_s = None
    spires = []
    jung = Wf[BI['jungle']]
    for _ in range(4000):
        if len(spires) >= 46: break
        py, px = rng.integers(200, N - 200, 2)
        if jung[py, px] < .75 or rivF[py, px] > .05 or H[py, px] < 10: continue
        if any(math.hypot(px - sx, py - sy) < 260 for sx, sy, *_ in spires): continue
        R = float(rng.uniform(35, 95)); top = float(rng.uniform(140, 340))
        r0 = int(R * 1.6)
        sl = np.s_[py - r0:py + r0, px - r0:px + r0]
        gy, gx = np.mgrid[-r0:r0, -r0:r0].astype(np.float32)
        ang = np.arctan2(gy, gx)
        rr = np.hypot(gx, gy) / (R * (1 + .18 * np.sin(ang * 3 + rng.random() * 6) + .08 * np.sin(ang * 7)))
        prof = smoothstep(1.0, .82, rr) * (top + 12 * np.cos(rr * 9)) + smoothstep(1.5, 1.0, rr) * 18
        H[sl] = np.maximum(H[sl], np.where(prof > .5, H[py, px] + prof, -1e9))
        spires.append((int(px), int(py), R, top))

    # Mother Tree basin: a flattened clearing in the heart of the rainforest
    jb = ndi.zoom(ndi.gaussian_filter(W[BI['jungle']] * (h > 15) * (h < 220), 18), z, order=1)[:N, :N]
    my, mx = np.unravel_index(np.argmax(jb), jb.shape)
    gy, gx = np.ogrid[:N, :N]
    rm = np.hypot(gy - my, gx - mx)
    basin = smoothstep(420, 180, rm)
    H = H * (1 - basin) + basin * (H[my, mx] - 6 + (rm < 60) * 14)
    del rm, gy, gx

    H = np.clip(H, H_MIN, H_MAX)
    log('slope + paint layers')
    gyH, gxH = np.gradient(H)
    slope = np.degrees(np.arctan(np.hypot(gxH, gyH))).astype(np.float32)
    del gxH, gyH
    wet = np.clip(rivF * 3 + lakeF, 0, 1)
    n1 = fbm(rng, (N, N), 60, 3, order=1)
    L = {}
    cliff = smoothstep(32, 46, slope)
    flat = 1 - cliff
    beachL = smoothstep(4, 1, H) * (H > -30) * flat
    snow = smoothstep(820, 1000, H + n1 * 120) * smoothstep(42, 30, slope)
    base = flat * (1 - beachL) * (1 - snow)
    L['Grass'] = base * Wf[BI['plains']]
    L['JungleFloor'] = base * Wf[BI['jungle']] * smoothstep(.62, .4, n1)
    L['Glowmoss'] = base * Wf[BI['jungle']] * smoothstep(.4, .62, n1) + base * Wf[BI['swamp']] * .3
    L['RedSand'] = base * Wf[BI['mesa']]
    L['Ash'] = base * Wf[BI['volcanic']]
    L['Mud'] = base * Wf[BI['swamp']] * .7 + wet * flat * .8
    L['Rock'] = cliff + base * Wf[BI['frozen']] * .6
    L['Snow'] = snow + base * Wf[BI['frozen']] * .4
    L['Sand'] = beachL
    tot = sum(L.values()) + 1e-6
    lay_dir = os.path.join(a.out, 'layers'); os.makedirs(lay_dir, exist_ok=True)
    for k in L:
        Image.fromarray((L[k] / tot * 255).round().astype(np.uint8)).save(os.path.join(lay_dir, f'{k}.png'), optimize=False)
    forest = (Wf[BI['jungle']] * 1.0 + Wf[BI['swamp']] * .7 + Wf[BI['plains']] * .25 + Wf[BI['frozen']] * .35) * smoothstep(30, 18, slope) * (H > 2) * (H < 950) * (1 - wet)
    forest *= smoothstep(.3, .55, fbm(rng, (N, N), 30, 3, order=1))
    Image.fromarray((np.clip(forest, 0, 1) * 255).astype(np.uint8)).save(os.path.join(lay_dir, 'ForestDensity.png'))
    del L, tot, n1, cliff, flat, base

    log('heightmap PNG (16-bit)')
    h16 = np.round((H - H_MIN) / (H_MAX - H_MIN) * 65535).astype(np.uint16)
    Image.fromarray(h16).save(os.path.join(a.out, f'heightmap_{N}.png'))
    # also a raw file (Unreal accepts .r16, little-endian)
    h16.astype('<u2').tofile(os.path.join(a.out, f'heightmap_{N}.r16'))

    log('points of interest')
    def ground(px, py): return float(H[int(py), int(px)])
    def biome_at(px, py): return BIOMES[int(np.argmax([w[int(py), int(px)] for w in Wf]))][0]
    pois = [{'type': 'MotherTree', 'name': 'The Mother Tree', 'x': int(mx), 'y': int(my), 'z': ground(mx, my), 'height_m': 600}]
    vy_, vx_ = int(vcy * (N - 1)), int(vcx * (N - 1))
    pois.append({'type': 'Volcano', 'name': 'Mount Cinderheart', 'x': vx_, 'y': vy_, 'z': ground(vx_, vy_)})
    for i, (sx, sy, R, top) in enumerate(spires):
        pois.append({'type': 'Spire', 'name': f'Skyspire {i + 1}', 'x': sx, 'y': sy, 'z': ground(sx, sy), 'radius_m': R})
    # viewpoints = tallest local maxima (Horizon-style climb-to-reveal)
    small = H[::16, ::16]
    peaks = (small == ndi.maximum_filter(small, 40)) & (small > 300)
    for py, px in sorted(zip(*np.nonzero(peaks)), key=lambda p: -small[p])[:14]:
        pois.append({'type': 'Viewpoint', 'name': f'Lookout {len([p for p in pois if p["type"] == "Viewpoint"]) + 1}', 'x': int(px * 16), 'y': int(py * 16), 'z': ground(px * 16, py * 16)})
    # player start: a gentle beach-side meadow in the plains near the coast
    best = None
    for _ in range(20000):
        px, py = rng.integers(300, N - 300, 2)
        if biome_at(px, py) != 'plains' or not (4 < H[py, px] < 40) or slope[py, px] > 8: continue
        best = (int(px), int(py)); break
    if best: pois.append({'type': 'PlayerStart', 'name': 'Landfall', 'x': best[0], 'y': best[1], 'z': ground(*best)})
    # alpha nests, one per biome, plus dino spawn zones
    zones = []
    for key, name, _ in BIOMES:
        w = Wf[BI[key]]
        cand = 0
        for _ in range(40000):
            px, py = rng.integers(150, N - 150, 2)
            if w[py, px] < .8 or H[py, px] < 3 or slope[py, px] > 22: continue
            if cand == 0: pois.append({'type': 'AlphaNest', 'name': f'{name} Alpha Nest', 'biome': key, 'x': int(px), 'y': int(py), 'z': ground(px, py)})
            elif all(math.hypot(px - z_['x'], py - z_['y']) > 420 for z_ in zones):
                sp = list(rng.choice(SPECIES[key], size=min(3, len(SPECIES[key])), replace=False))
                zones.append({'biome': key, 'x': int(px), 'y': int(py), 'z': ground(px, py), 'radius_m': int(rng.integers(120, 260)), 'species': sp, 'herd': int(rng.integers(3, 12))})
            cand += 1
            if cand > 40: break
    # ruins scattered across the world
    for i in range(30):
        for _ in range(500):
            px, py = rng.integers(200, N - 200, 2)
            if H[py, px] > 3 and slope[py, px] < 14:
                pois.append({'type': 'Ruin', 'name': f'Ancient Ruin {i + 1}', 'biome': biome_at(px, py), 'x': int(px), 'y': int(py), 'z': ground(px, py)}); break

    log('colossal trees + floating islands')
    titans = []
    step = 140
    for gy0 in range(step, N - step, step):
        for gx0 in range(step, N - step, step):
            px, py = gx0 + int(rng.integers(-50, 50)), gy0 + int(rng.integers(-50, 50))
            jw = Wf[BI['jungle']][py, px] + Wf[BI['swamp']][py, px] * .6
            if rng.random() > jw * .8 or slope[py, px] > 20 or H[py, px] < 2 or rivF[py, px] > .05 or H[py, px] > 700: continue
            if any(math.hypot(px - s[0], py - s[1]) < s[2] * 1.4 for s in spires): continue
            titans.append({'x': px, 'y': py, 'z': ground(px, py), 'height_m': round(float(rng.uniform(90, 260)), 1),
                           'trunk_m': round(float(rng.uniform(8, 22)), 1), 'yaw': round(float(rng.uniform(0, 360)), 1)})
    islands = []
    for _ in range(4000):
        if len(islands) >= 60: break
        px, py = rng.integers(300, N - 300, 2)
        if Wf[BI['jungle']][py, px] < .6: continue
        if any(math.hypot(px - i_['x'], py - i_['y']) < 220 for i_ in islands): continue
        islands.append({'x': int(px), 'y': int(py), 'z': ground(px, py) + float(rng.uniform(220, 520)), 'radius_m': round(float(rng.uniform(25, 90)), 1)})

    meta = {
        'name': 'Wildheart', 'seed': a.seed, 'size_px': N, 'world_km': round(world_m / 1000, 3), 'meters_per_px': 1.0,
        'height_range_m': [H_MIN, H_MAX], 'sea_level_m': SEA,
        'unreal_import': {
            'heightmap': f'heightmap_{N}.png', 'scale_x': 100.0, 'scale_y': 100.0,
            'scale_z': round((H_MAX - H_MIN) * 100 / 512, 4),
            'note': 'Landscape > Import from File. Set Location Z so that sea level sits at 0: see README.',
            'location_z_for_sea_at_zero_cm': round(-((H_MIN + H_MAX) / 2 - SEA) * 100, 1),
        },
        'biomes': [{'key': k, 'name': nm} for k, nm, _ in BIOMES],
        'paint_layers': ['Grass', 'JungleFloor', 'Glowmoss', 'RedSand', 'Ash', 'Mud', 'Rock', 'Snow', 'Sand'],
        'pois': pois, 'spawn_zones': zones, 'colossal_trees': titans, 'floating_islands': islands,
        'lakes_px_macro': int(lake.sum()), 'river_cells_macro': int(river.sum()),
    }
    with open(os.path.join(a.out, 'world.json'), 'w') as fp: json.dump(meta, fp, indent=1)

    log('preview map')
    P = 2048; zs = P / N
    hp = ndi.zoom(H, zs, order=1)[:P, :P]
    wl = ndi.zoom(water_level, P / n, order=1)[:P, :P]
    bw = np.stack([ndi.zoom(w, zs, order=1)[:P, :P] for w in Wf])
    col = np.tensordot(np.array([b[2] for b in BIOMES], np.float32).T, bw, axes=1).transpose(1, 2, 0)
    sl = ndi.zoom(slope, zs, order=1)[:P, :P]
    col = col * (1 - smoothstep(30, 45, sl)[..., None] * .55) + np.array([110, 104, 96]) * smoothstep(30, 45, sl)[..., None] * .55
    col = np.where((hp > 820)[..., None], col * .4 + 240 * .6, col)
    gy, gx = np.gradient(hp)
    shade = np.clip(1 + (-gx - gy) * .045, .45, 1.5)[..., None]
    col = col * shade
    lk = ndi.zoom(lakeF, zs, order=1)[:P, :P] > .5
    rv = ndi.zoom(rivF, zs, order=1)[:P, :P] > .25
    wmask = (hp < SEA) | lk | rv
    depthc = np.clip((wl - hp) / 60, 0, 1)[..., None] if False else np.clip(-hp / 90, 0, 1)[..., None]
    wcol = np.array([60, 150, 180]) * (1 - depthc) + np.array([14, 50, 92]) * depthc
    col = np.where(wmask[..., None], wcol, col)
    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    try: font = ImageFont.truetype('DejaVuSans-Bold.ttf', 22); small_f = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError: font = small_f = ImageFont.load_default()
    for t in titans: d.ellipse([t['x'] * zs - 1.5, t['y'] * zs - 1.5, t['x'] * zs + 1.5, t['y'] * zs + 1.5], fill=(20, 230, 190))
    for i_ in islands: d.ellipse([i_['x'] * zs - 4, i_['y'] * zs - 4, i_['x'] * zs + 4, i_['y'] * zs + 4], outline=(190, 120, 255), width=2)
    sty = {'MotherTree': ((40, 255, 200), 12), 'Volcano': ((255, 90, 30), 10), 'Viewpoint': ((255, 230, 90), 6), 'PlayerStart': ((255, 255, 255), 9),
           'AlphaNest': ((255, 60, 60), 7), 'Ruin': ((210, 190, 150), 4), 'Spire': ((150, 140, 130), 3)}
    for p in pois:
        c, r = sty[p['type']]; x, y = p['x'] * zs, p['y'] * zs
        d.ellipse([x - r, y - r, x + r, y + r], fill=c, outline=(0, 0, 0), width=2)
        if p['type'] in ('MotherTree', 'Volcano', 'PlayerStart', 'AlphaNest'):
            d.text((x + r + 4, y - 10), p['name'], fill=(255, 255, 255), font=small_f, stroke_width=3, stroke_fill=(0, 0, 0))
    for i in range(1, 8):  # 1 km grid
        g = i * 1000 * zs; d.line([(g, 0), (g, P)], fill=(255, 255, 255, 60), width=1); d.line([(0, g), (P, g)], fill=(255, 255, 255, 60), width=1)
    for key, name, _ in BIOMES:
        w = ndi.gaussian_filter(bw[BI[key]], 40); y, x = np.unravel_index(np.argmax(w), w.shape)
        d.text((x - 80, y + 18), name.upper(), fill=(255, 255, 255), font=font, stroke_width=4, stroke_fill=(0, 0, 0))
    d.rectangle([20, P - 70, 360, P - 20], fill=(0, 0, 0))
    d.text((32, P - 62), f'WILDHEART  ·  {world_m / 1000:.1f} × {world_m / 1000:.1f} km  ·  seed {a.seed}', fill=(255, 255, 255), font=small_f)
    d.text((32, P - 40), f'{len(titans)} colossal trees · {len(islands)} floating isles · {len(spires)} spires', fill=(180, 230, 210), font=small_f)
    img.save(os.path.join(a.out, 'preview_map.png'))
    log(f'done in {time.time() - t0:.0f}s → {a.out}')

if __name__ == '__main__':
    main()
