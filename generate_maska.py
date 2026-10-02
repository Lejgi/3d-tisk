#!/usr/bin/env python3
"""Reliéf masky podle referenční fotky, bílá + vínové pruhy.

Silueta, oko a pruhy se berou z obrázku. Model má rovná záda, takže se na
Enderu 3 V3 SE tiskne naplocho bez podpěr. V 3MF jsou dvě tělesa.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parent
REF = Path(
    "/home/ubuntu/.cursor/projects/workspace/assets/"
    "3e5becc3-77ab-4249-8110-bb47646d1cff.png"
)
BED_X = 220.0
BED_Y = 220.0
PITCH = 0.45
TARGET_H = 198.0
BASE_H = 4.8
DOME_H = 10.5
GROOVE = 1.35
RED_H = 1.05
WHITE = "#F4F4F2"
RED = "#8E1E2C"


def split_teeth(solid, lum):
    """Odřízne stíny mezi zuby, aby každý zub zůstal samostatný a spojený s čelistí."""
    height, width = solid.shape
    row = np.arange(height)[:, None]
    bright = (lum > 170) & (row > int(height * 0.76))
    bright = ndimage.binary_opening(bright, iterations=1)
    labels, count = ndimage.label(bright)
    if count == 0:
        return solid
    sizes = np.bincount(labels.ravel())[1:]
    teeth = np.zeros_like(solid)
    for index, size in enumerate(sizes):
        if size < 80:
            continue
        comp = labels == index + 1
        yy, xx = np.nonzero(comp)
        tooth_h = int(yy.max() - yy.min())
        tooth_w = int(xx.max() - xx.min())
        if tooth_h > 115 or tooth_w > 42 or tooth_h < 25:
            continue
        if xx.mean() < width * 0.55 or yy.mean() < height * 0.80:
            continue
        teeth |= comp
    if int(teeth.sum()) < 200:
        return solid
    teeth = ndimage.binary_dilation(teeth, structure=np.ones((1, 5), dtype=bool))
    grown = teeth.copy()
    wide = ndimage.binary_dilation(solid, iterations=3)
    for _ in range(48):
        up = np.zeros_like(grown)
        up[:-1] = grown[1:]
        grown |= up & wide
    cut = int(np.nonzero(teeth)[0].min()) - 16
    root = np.zeros_like(grown)
    root[cut : cut + 24] = True
    grown |= ndimage.binary_dilation(teeth, iterations=3) & root & wide
    right = np.arange(width)[None, :] > int(width * 0.50)
    out = solid.copy()
    out[cut:] = np.where(right, grown, solid)[cut:]
    # Svislé uzavření spojí kořeny s čelistí a mezery mezi zuby nechá.
    linked = ndimage.binary_closing(out, structure=np.ones((15, 1), dtype=bool))
    x0, x1 = int(width * 0.52), int(width * 0.95)
    out[cut - 8 :, x0:x1] = linked[cut - 8 :, x0:x1]
    print(f"  zuby: {int(ndimage.label(teeth)[1])} ks", flush=True)
    return out


def load_masks():
    rgb = np.asarray(Image.open(REF).convert("RGB")).astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lum = rgb.mean(axis=2)
    bg = np.abs(rgb - 89).sum(axis=2) < 18
    fg = (~bg & (lum > 100)) | ((r > g + 10) & (r > b + 8) & (r > 55) & (lum < 200) & ~bg)
    lab, _n = ndimage.label(fg)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.argsort(sizes)[::-1][:2] + 1
    mask = np.isin(lab, keep)
    ys, xs = np.where(mask)
    y0, y1 = ys.min() - 2, ys.max() + 3
    x0, x1 = xs.min() - 2, xs.max() + 3
    mask = mask[y0:y1, x0:x1]
    rgb_c = rgb[y0:y1, x0:x1]
    r, g, b = rgb_c[:, :, 0], rgb_c[:, :, 1], rgb_c[:, :, 2]
    lum = rgb_c.mean(axis=2)

    closed = ndimage.binary_closing(mask, iterations=5)
    filled = ndimage.binary_fill_holes(closed)
    holes = filled & ~closed
    hl, hn = ndimage.label(holes)
    hs = np.bincount(hl.ravel())[1:] if hn else np.array([])
    eye = np.zeros_like(closed)
    if len(hs):
        eye = hl == (int(np.argmax(hs)) + 1)
    solid = closed | (holes & ~eye)
    solid = ndimage.binary_opening(solid, iterations=1)
    solid = ndimage.binary_closing(solid, iterations=1)
    # Oko znovu proříznout, ať ho jemné uzavření nezalepí.
    solid &= ~ndimage.binary_dilation(eye, iterations=1)
    solid = split_teeth(solid, lum)

    reddish = (r > g + 10) & (r > b + 8) & (r > 60) & (lum < 195) & solid
    reddish = ndimage.binary_closing(reddish, iterations=2)
    reddish = ndimage.binary_dilation(reddish, iterations=2)
    reddish &= solid
    rl, rn = ndimage.label(reddish)
    rs = np.bincount(rl.ravel())[1:] if rn else np.array([])
    stripe = np.zeros_like(solid)
    if len(rs):
        for idx in np.argsort(rs)[::-1][:3]:
            if rs[idx] < 80:
                continue
            stripe |= rl == (idx + 1)
    stripe = ndimage.binary_dilation(stripe, iterations=1) & solid & ~eye
    print(
        f"  silueta {solid.shape[1]}×{solid.shape[0]} px, "
        f"oko {int(eye.sum())} px, pruhy {int(stripe.sum())} px",
        flush=True,
    )
    return solid, stripe


def resample(solid, stripe):
    zoom = (TARGET_H / solid.shape[0]) / PITCH
    solid_r = ndimage.zoom(solid.astype(np.float32), zoom, order=1) > 0.5
    stripe_r = ndimage.zoom(stripe.astype(np.float32), zoom, order=1) > 0.35
    stripe_r &= solid_r
    # Okraj díry nesmí být tenčí než ~2.2 mm.
    dist_in = ndimage.distance_transform_edt(solid_r) * PITCH
    thin_eye_rim = solid_r & (dist_in < 2.2)
    # Zúžit jen pruhy, ne celou siluetu.
    stripe_r &= ~thin_eye_rim
    print(
        f"  síť {solid_r.shape[1]}×{solid_r.shape[0]}  "
        f"{solid_r.shape[1] * PITCH:.1f}×{solid_r.shape[0] * PITCH:.1f} mm",
        flush=True,
    )
    return solid_r, stripe_r


def height_field(solid, stripe):
    rows, cols = solid.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    xs = (xx + 0.5) * PITCH
    ys = (yy + 0.5) * PITCH
    face = solid & (xx >= np.quantile(xx[solid], 0.55))
    cx = float(xs[face].mean())
    cy = float(ys[face].mean())
    dome = DOME_H * np.exp(-0.5 * (((xs - cx) / 24.0) ** 2 + ((ys - cy) / 32.0) ** 2))
    st = ndimage.gaussian_filter(stripe.astype(np.float32), 0.85)
    st = np.clip(st, 0.0, 1.0)
    st *= solid
    height = BASE_H + dome - GROOVE * st
    return height.astype(np.float32), st.astype(np.float32)


def sdf_volume(sdf2d, z_lo, z_hi):
    """sdf2d < 0 uvnitř půdorysu. Těleso je mezi z_lo a z_hi (pole, mm)."""
    zmax = float(np.max(z_hi)) + 3 * PITCH
    zmin = -2 * PITCH
    zs = np.arange(zmin, zmax, PITCH)
    vol = np.empty((len(zs),) + sdf2d.shape, dtype=np.float32)
    for i, z in enumerate(zs):
        vol[i] = np.maximum(np.maximum(sdf2d, z - z_hi), z_lo - z)
    return vol, zs


def sdf2d_of(region):
    inside = ndimage.distance_transform_edt(region)
    outside = ndimage.distance_transform_edt(~region)
    sdf = (outside - inside) * PITCH
    return ndimage.gaussian_filter(sdf, 0.4).astype(np.float32)


def mesh_from_sdf(vol, origin_xy, zs):
    verts, faces, _n, _v = marching_cubes(vol, level=0.0, spacing=(PITCH, PITCH, PITCH))
    # marching cubes: osa 0 = z, osa 1 = řádek (obrázek dolů), osa 2 = x
    world = np.column_stack(
        [
            verts[:, 2] + origin_xy[0],
            -(verts[:, 1] + origin_xy[1]),
            verts[:, 0] + zs[0],
        ]
    )
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    parts = mesh.split(only_watertight=False)
    parts = sorted(parts, key=lambda p: len(p.faces), reverse=True)
    print(
        "  komponenty:",
        [
            (
                len(p.faces),
                np.round(p.bounds[0], 1).tolist(),
                np.round(p.bounds[1] - p.bounds[0], 1).tolist(),
            )
            for p in parts[:6]
        ],
        flush=True,
    )
    kept = [p for p in parts if len(p.faces) >= 2500]
    if len(kept) == 1:
        return kept[0]
    return trimesh.util.concatenate(kept)


def center_on_bed(meshes):
    allv = np.vstack([m.vertices for m in meshes])
    shift = np.array(
        [
            -(allv[:, 0].min() + allv[:, 0].max()) * 0.5,
            -(allv[:, 1].min() + allv[:, 1].max()) * 0.5,
            -min(0.0, allv[:, 2].min()),
        ]
    )
    for m in meshes:
        m.vertices += shift
    allv = np.vstack([m.vertices for m in meshes])
    ext = allv.max(axis=0) - allv.min(axis=0)
    print(f"  na podložce XYZ {ext.round(1).tolist()} mm, z {allv[:, 2].min():.2f}…{allv[:, 2].max():.1f}", flush=True)
    if ext[0] > BED_X - 8 or ext[1] > BED_Y - 8 or ext[2] > 40:
        raise RuntimeError(f"rozměr nesedí na podložku: {ext}")
    return meshes


def _triangles_xml(mesh, pid, pindex):
    sv = "".join(
        f'<vertex x="{p[0]:.3f}" y="{p[1]:.3f}" z="{p[2]:.3f}"/>' for p in mesh.vertices
    )
    st = "".join(
        f'<triangle v1="{int(a)}" v2="{int(b)}" v3="{int(c)}" pid="{pid}" p1="{pindex}"/>'
        for a, b, c in mesh.faces
    )
    return f"<mesh><vertices>{sv}</vertices><triangles>{st}</triangles></mesh>"


def write_3mf_single(mesh, path: Path):
    """Jen bílá maska, drážky zůstanou prázdné pro malování."""
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="cs-CZ" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">'
        '<metadata name="Title">Maska relief bila</metadata>'
        '<metadata name="Description">Bily relief s drazkami na cervene pruhy. Rovne zady na podlozku.</metadata>'
        "<resources>"
        '<m:basematerials id="1">'
        f'<m:base name="Bila" displaycolor="{WHITE}FF"/>'
        "</m:basematerials>"
        '<object id="2" name="Maska bila" type="model">'
        f"{_triangles_xml(mesh, 1, 0)}</object>"
        "</resources><build><item objectid=\"2\"/></build></model>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        "</Relationships>"
    )
    ctype = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        "</Types>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ctype)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", model.encode("utf-8"))


def write_3mf(white, red, path: Path):
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="cs-CZ" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">'
        '<metadata name="Title">Maska relief bila a cervena</metadata>'
        '<metadata name="Description">Relief masky podle fotky. Rovne zady na podlozku. Bile teleso a vinove pruhy. Ender 3 V3 SE, PLA, tryska 0,4 mm.</metadata>'
        "<resources>"
        '<m:basematerials id="1">'
        f'<m:base name="Bila" displaycolor="{WHITE}FF"/>'
        f'<m:base name="Cervena" displaycolor="{RED}FF"/>'
        "</m:basematerials>"
        '<object id="2" name="Maska bila" type="model">'
        f"{_triangles_xml(white, 1, 0)}</object>"
        '<object id="3" name="Pruhy cervene" type="model">'
        f"{_triangles_xml(red, 1, 1)}</object>"
        "</resources><build><item objectid=\"2\"/><item objectid=\"3\"/></build></model>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        "</Relationships>"
    )
    ctype = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        "</Types>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ctype)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", model.encode("utf-8"))


def _raster(img, zb, pts, faces, color, light):
    h, w = img.shape[:2]
    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    shade = np.clip(nrm @ light, 0.0, 1.0)
    col = np.clip(color * (0.34 + 0.7 * shade)[:, None], 0, 1)
    area2 = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    idx = np.flatnonzero((area2 > 0.6) & (nrm[:, 2] > 0.0))
    for i in idx:
        tri = np.vstack([v0[i], v1[i], v2[i]])
        minx = max(int(np.floor(tri[:, 0].min())), 0)
        maxx = min(int(np.ceil(tri[:, 0].max())), w - 1)
        miny = max(int(np.floor(tri[:, 1].min())), 0)
        maxy = min(int(np.ceil(tri[:, 1].max())), h - 1)
        if minx > maxx or miny > maxy:
            continue
        xs = np.arange(minx, maxx + 1, dtype=np.float32)
        ys = np.arange(miny, maxy + 1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        den = (v0[i, 0] - v2[i, 0]) * (v1[i, 1] - v2[i, 1]) - (v1[i, 0] - v2[i, 0]) * (v0[i, 1] - v2[i, 1])
        if abs(den) < 1e-6:
            continue
        w0 = ((gx - v2[i, 0]) * (v1[i, 1] - v2[i, 1]) - (gy - v2[i, 1]) * (v1[i, 0] - v2[i, 0])) / den
        w1 = ((gy - v2[i, 1]) * (v0[i, 0] - v2[i, 0]) - (gx - v2[i, 0]) * (v0[i, 1] - v2[i, 1])) / den
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        if not np.any(inside):
            continue
        depth = w0 * v0[i, 2] + w1 * v1[i, 2] + (1 - w0 - w1) * v2[i, 2]
        view = zb[miny : maxy + 1, minx : maxx + 1]
        closer = inside & (depth > view)
        if not np.any(closer):
            continue
        view[closer] = depth[closer]
        img[miny : maxy + 1, minx : maxx + 1][closer] = col[i]


def render(white, red, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    light = np.array([0.2, 0.35, 0.85], dtype=np.float64)
    light /= np.linalg.norm(light)
    views = (0.0, -58.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 8.4), facecolor="#5c5c5c")
    size = 760
    rgb_w = np.array([0.95, 0.95, 0.93])
    rgb_r = np.array([0.62, 0.12, 0.17])
    for ax, az in zip(axes, views):
        img = np.full((size, size, 3), 0.33, np.float32)
        zb = np.full((size, size), -1e9, np.float32)
        ca, sa = np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az))

        def cam(v):
            x, y, z = v[:, 0], v[:, 1], v[:, 2]
            xr = ca * x + sa * z
            zr = -sa * x + ca * z
            return np.column_stack([xr, y, zr])

        clouds = [cam(m.vertices) for m in (white, red)]
        allp = np.vstack(clouds)
        span = max(float(np.ptp(allp[:, 0])), float(np.ptp(allp[:, 1])))
        scale = (size * 0.88) / span
        mid = np.array([allp[:, 0].mean(), (allp[:, 1].min() + allp[:, 1].max()) * 0.5, 0.0])
        for mesh, rgb in ((white, rgb_w), (red, rgb_r)):
            pts = (cam(mesh.vertices) - mid) * scale
            pts[:, 0] += size * 0.5
            pts[:, 1] += size * 0.42
            _raster(img, zb, pts, mesh.faces, rgb, light)
        ax.imshow(img, origin="lower")
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    print("Čtu fotku…", flush=True)
    solid, stripe = load_masks()
    solid, stripe = resample(solid, stripe)
    height, st = height_field(solid, stripe)
    print("Bílá skořepina…", flush=True)
    vol, zs = sdf_volume(sdf2d_of(solid), np.float32(0.0), height)
    white = mesh_from_sdf(vol, (0.0, 0.0), zs)
    print("Červené pruhy…", flush=True)
    red_region = st > 0.5
    z_lo = height - 0.12
    z_hi = height + RED_H
    vol_r, zs_r = sdf_volume(sdf2d_of(red_region), z_lo, z_hi)
    red = mesh_from_sdf(vol_r, (0.0, 0.0), zs_r)
    white, red = center_on_bed([white, red])
    for name, mesh in ("bílá", white), ("červená", red):
        print(
            f"  {name}: watertight={mesh.is_watertight} "
            f"objem={mesh.volume:.0f} mm3 stěn={len(mesh.faces)}",
            flush=True,
        )
    out = ROOT / "modely" / "maska-hollow.3mf"
    write_3mf(white, red, out)
    print(f"  3MF {out.name} {out.stat().st_size / 1e6:.1f} MB", flush=True)
    bila = ROOT / "modely" / "maska-hollow-bila.3mf"
    write_3mf_single(white, bila)
    print(f"  3MF {bila.name} {bila.stat().st_size / 1e6:.1f} MB", flush=True)
    preview = ROOT / "nahledy" / "maska-hollow.png"
    render(white, red, preview)
    print(f"  náhled {preview.name}", flush=True)


if __name__ == "__main__":
    main()
