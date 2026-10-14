#!/usr/bin/env python3
"""Generate a Škoda winged-arrow desk lightbox matching the Audi logo 3MF style.

Parts (same construction as Logo Audi.3mf):
  1. Black housing — solid back, front pocket, through openings, two tabs
  2. White diffuser face — flat base + rounded top profile
  3. Black base (socle) — chamfered block with two through-slots for the tabs

Sized larger than the Audi reference (~222×97×8 mm rings): Škoda circle Ø300 mm.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import cadquery as cq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.validation import make_valid
from svgpathtools import parse_path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "modely"
PREVIEW_DIR = ROOT / "nahledy"
SVG_BUNDLE = ROOT / "assets" / "skoda-winged-arrow.svg"

# --- Parameters taken from Logo Audi.3mf ---
BLACK_Z = 8.0
WHITE_Z = 10.0
WHITE_DOME_Z = 5.0
POCKET_Z = 4.0
BASE_H = 15.0
BASE_D = 72.0  # Audi 60 mm, slightly deeper for the larger logo
BASE_L = 240.0  # Audi 170 mm
TAB_W = 15.0
TAB_H = 15.0
SLOT_CLEAR = 0.3
BASE_CHAMFER = 12.0  # Audi ~12 mm horizontal inset on top face

# --- Larger than Audi ---
BLACK_OUTER_D = 300.0
LIP = 5.0
WHITE_OUTER_D = BLACK_OUTER_D - 2.0 * LIP
TAB_SPACING = 130.0

SVG_VIEW = 24.0
# SVG ring inner margin (simple-icons path).
SVG_RING_INSET = 1.0364


def load_svg_path() -> str:
    text = SVG_BUNDLE.read_text(encoding="utf-8")
    m = re.search(r'<path[^>]*\sd="([^"]+)"', text)
    if not m:
        raise RuntimeError(f"no path data in {SVG_BUNDLE}")
    return m.group(1)


def sample_subpath(sp, density: float = 10.0) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for seg in sp:
        n = max(20, int(density * max(abs(seg.length()), 0.2)))
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            p = seg.point(t)
            pts.append((float(p.real), float(-p.imag)))
    p = sp[-1].point(1.0)
    pts.append((float(p.real), float(-p.imag)))
    out = [pts[0]]
    for p in pts[1:]:
        if (p[0] - out[-1][0]) ** 2 + (p[1] - out[-1][1]) ** 2 > 1e-12:
            out.append(p)
    if out[0] != out[-1]:
        out.append(out[0])
    return np.array(out, dtype=float)


def _xform_poly(poly: Polygon, cx: float, cy: float, scale: float) -> Polygon:
    return Polygon(
        [((x - cx) * scale, (y - cy) * scale) for x, y in poly.exterior.coords],
        [
            [((x - cx) * scale, (y - cy) * scale) for x, y in hole.coords]
            for hole in poly.interiors
        ],
    )


def build_face_parts_mm() -> tuple[float, float, Polygon]:
    """Return (R_outer, R_inner, arrow_polygon_mm)."""
    path = parse_path(load_svg_path())
    subs = path.continuous_subpaths()
    if len(subs) < 4:
        raise RuntimeError(f"expected 4 SVG subpaths, got {len(subs)}")

    polys = []
    for sp in subs:
        poly = make_valid(Polygon(sample_subpath(sp)))
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        polys.append(poly)
    _outer, _inner, arrow, eye = polys

    cx, cy = 12.0, -12.0
    scale = WHITE_OUTER_D / SVG_VIEW
    arrow_mm = make_valid(
        _xform_poly(arrow, cx, cy, scale).difference(_xform_poly(eye, cx, cy, scale))
    )
    if arrow_mm.geom_type == "MultiPolygon":
        arrow_mm = max(arrow_mm.geoms, key=lambda g: g.area)
    arrow_mm = make_valid(arrow_mm.simplify(0.2, preserve_topology=True).buffer(0))

    r_outer = 12.0 * scale
    r_inner = (12.0 - SVG_RING_INSET) * scale
    return r_outer, r_inner, arrow_mm


def face_polygon_mm(r_outer: float, r_inner: float, arrow: Polygon):
    from shapely.geometry import Point

    ring = Point(0, 0).buffer(r_outer, resolution=96).difference(
        Point(0, 0).buffer(r_inner, resolution=96)
    )
    return make_valid(ring.union(arrow))


def clean_ring_coords(coords) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in list(coords)[:-1]]
    out = [pts[0]]
    for p in pts[1:]:
        if (p[0] - out[-1][0]) ** 2 + (p[1] - out[-1][1]) ** 2 > 1e-8:
            out.append(p)
    return out


def cq_from_polygon(poly: Polygon, height: float) -> cq.Workplane:
    poly = make_valid(poly).buffer(0)
    if poly.geom_type == "MultiPolygon":
        solids = [cq_from_polygon(g, height) for g in poly.geoms if g.area > 1e-3]
        out = solids[0]
        for s in solids[1:]:
            out = out.union(s)
        return out

    ext = clean_ring_coords(poly.exterior.coords)
    if not Polygon(ext).exterior.is_ccw:
        ext = ext[::-1]
    wp = cq.Workplane("XY").polyline(ext).close()
    for hole in poly.interiors:
        hc = clean_ring_coords(hole.coords)
        if Polygon(hc).exterior.is_ccw:
            hc = hc[::-1]
        wp = wp.moveTo(hc[0][0], hc[0][1]).polyline(hc[1:] + [hc[0]]).close()
    return wp.extrude(height)


def cq_ring(r_outer: float, r_inner: float, height: float) -> cq.Workplane:
    return cq.Workplane("XY").circle(r_outer).circle(r_inner).extrude(height)


def cq_face_solid(r_outer: float, r_inner: float, arrow: Polygon, height: float) -> cq.Workplane:
    return cq_ring(r_outer, r_inner, height).union(cq_from_polygon(arrow, height))


def cq_to_trimesh(workplane: cq.Workplane, tol: float = 0.12) -> trimesh.Trimesh:
    shape = workplane.val()
    verts, faces = shape.tessellate(tol, 0.4)
    mesh = trimesh.Trimesh(
        vertices=np.array([(v.x, v.y, v.z) for v in verts], dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
        process=True,
    )
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def dome_white(r_outer: float, r_inner: float, arrow: Polygon) -> cq.Workplane:
    """10 mm face with top-edge chamfer (Audi white has a rounded front profile)."""
    solid = cq_face_solid(r_outer, r_inner, arrow, WHITE_Z)
    # Top chamfer echoes Audi's rounded white profile (4 mm breaks manifold).
    chamfer = min(3.5, WHITE_DOME_Z - 0.5, (r_outer - r_inner) / 2.0 - 0.8)
    return solid.faces(">Z").edges().chamfer(chamfer)


def make_black_cq(r_outer: float, r_inner: float, arrow: Polygon) -> cq.Workplane:
    face = face_polygon_mm(r_outer, r_inner, arrow)
    outer = make_valid(face.buffer(LIP))
    body = cq_from_polygon(outer, BLACK_Z)

    # Front pocket / LED cavity.
    pocket_shape = make_valid(face.buffer(-0.25))
    pocket = cq_from_polygon(pocket_shape, POCKET_Z + 0.1).translate(
        (0, 0, BLACK_Z - POCKET_Z)
    )
    body = body.cut(pocket)

    # Two Audi-style tabs under the logo.
    ymin = outer.bounds[1]
    tab_top = ymin + 2.0
    tab_bot = tab_top - TAB_H
    for sign in (-1.0, 1.0):
        cx = sign * (TAB_SPACING / 2.0)
        tab = (
            cq.Workplane("XY")
            .center(cx, (tab_top + tab_bot) / 2.0)
            .rect(TAB_W, TAB_H)
            .extrude(BLACK_Z)
        )
        body = body.union(tab)
    return body


def make_base_cq() -> cq.Workplane:
    """Trapezoid socle with two through-slots."""
    # Bottom rectangle
    bot = cq.Workplane("XY").rect(BASE_L, BASE_D).extrude(0.01)
    top = (
        cq.Workplane("XY")
        .workplane(offset=BASE_H)
        .rect(BASE_L - 2 * BASE_CHAMFER, BASE_D - 2 * BASE_CHAMFER)
        .extrude(0.01)
    )
    # Loft via hull of the two thin plates.
    solid = cq.Workplane("XY").add(bot.val()).union(top)
    # Convex hull through CQ: use loft between wires instead.
    bottom_wire = cq.Workplane("XY").rect(BASE_L, BASE_D).wires().val()
    top_wire = (
        cq.Workplane("XY")
        .workplane(offset=BASE_H)
        .rect(BASE_L - 2 * BASE_CHAMFER, BASE_D - 2 * BASE_CHAMFER)
        .wires()
        .val()
    )
    solid = cq.Workplane("XY").add(cq.Solid.makeLoft([bottom_wire, top_wire]))

    slot_w = TAB_W + SLOT_CLEAR
    slot_t = BLACK_Z + SLOT_CLEAR
    for sign in (-1.0, 1.0):
        cx = sign * (TAB_SPACING / 2.0)
        slot = (
            cq.Workplane("XY")
            .center(cx, 0)
            .rect(slot_w, slot_t)
            .extrude(BASE_H + 2)
            .translate((0, 0, -1))
        )
        solid = solid.cut(slot)
    return solid


def write_3mf(
    parts: list[tuple[str, trimesh.Trimesh, str]],
    path: Path,
    title: str,
    description: str,
) -> None:
    colors = {"black": "1A1A1AFF", "white": "F5F5F5FF"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        model = io.StringIO()
        model.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        model.write(
            '<model unit="millimeter" xml:lang="cs-CZ" '
            'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
            'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">\n'
        )
        model.write(f'  <metadata name="Title">{title}</metadata>\n')
        model.write(f'  <metadata name="Description">{description}</metadata>\n')
        model.write('  <metadata name="Application">generate_skoda_logo.py</metadata>\n')
        model.write("  <resources>\n")
        model.write('    <m:basematerials id="1">\n')
        model.write(f'      <m:base name="Cerna" displaycolor="{colors["black"]}"/>\n')
        model.write(f'      <m:base name="Bila" displaycolor="{colors["white"]}"/>\n')
        model.write("    </m:basematerials>\n")

        build_items = []
        obj_id = 2
        extents = []
        for name, mesh, material in parts:
            mat_index = 0 if material == "black" else 1
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            model.write(f'    <object id="{obj_id}" type="model" name="{name}">\n')
            model.write("      <mesh>\n        <vertices>\n")
            for x, y, z in verts:
                model.write(f'          <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>\n')
            model.write("        </vertices>\n        <triangles>\n")
            for a, b, c in faces:
                model.write(
                    f'          <triangle v1="{a}" v2="{b}" v3="{c}" '
                    f'pid="1" p1="{mat_index}"/>\n'
                )
            model.write("        </triangles>\n      </mesh>\n    </object>\n")
            build_items.append(obj_id)
            extents.append(mesh.extents)
            obj_id += 1

        model.write("  </resources>\n  <build>\n")
        cursor_x = 0.0
        gap = 10.0
        for oid, (name, mesh, _), ext in zip(build_items, parts, extents):
            cx = mesh.bounds[0][0] + ext[0] / 2.0
            cy = mesh.bounds[0][1] + ext[1] / 2.0
            cz = mesh.bounds[0][2]
            tx = cursor_x + ext[0] / 2.0 - cx
            ty = -cy
            tz = -cz
            model.write(
                f'    <item objectid="{oid}" '
                f'transform="1 0 0 0 1 0 0 0 1 {tx:.4f} {ty:.4f} {tz:.4f}"/>\n'
            )
            cursor_x += ext[0] + gap
        model.write("  </build>\n</model>\n")
        zf.writestr("3D/3dmodel.model", model.getvalue().encode("utf-8"))
        zf.writestr(
            "[Content_Types].xml",
            b"""<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
""",
        )
        zf.writestr(
            "_rels/.rels",
            b"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
""",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


def write_single_3mf(mesh: trimesh.Trimesh, path: Path, title: str, material: str) -> None:
    write_3mf([(title, mesh, material)], path, title, title)


def render_previews(face, black: trimesh.Trimesh, white: trimesh.Trimesh, base: trimesh.Trimesh) -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    geoms = [face] if face.geom_type == "Polygon" else list(face.geoms)
    for poly in geoms:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, facecolor="#7CB342", edgecolor="#111", lw=0.6)
        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, facecolor="white", edgecolor="#111", lw=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"Škoda — plocha (bílá Ø {WHITE_OUTER_D:.0f} / černá Ø {BLACK_OUTER_D:.0f} mm)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "skoda-logo-2d.png", dpi=150)
    plt.close(fig)

    def save_mesh_view(mesh, path, color, elev=25, azim=-60):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        v = mesh.vertices
        f = mesh.faces
        tris = v[f]
        if len(f) > 35000:
            tris = tris[np.linspace(0, len(f) - 1, 35000, dtype=int)]
        coll = Poly3DCollection(tris, linewidths=0.0, alpha=1.0)
        coll.set_facecolor(color)
        ax.add_collection3d(coll)
        c = (v.min(axis=0) + v.max(axis=0)) / 2.0
        ext = (v.max(axis=0) - v.min(axis=0)).max() / 2.0 * 1.08
        ax.set_xlim(c[0] - ext, c[0] + ext)
        ax.set_ylim(c[1] - ext, c[1] + ext)
        ax.set_zlim(c[2] - ext, c[2] + ext)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="#111")
        plt.close(fig)

    save_mesh_view(black, PREVIEW_DIR / "skoda-cerna.png", "#2a2a2a")
    save_mesh_view(white, PREVIEW_DIR / "skoda-bila.png", "#e8e8e8")
    save_mesh_view(base, PREVIEW_DIR / "skoda-podstavec.png", "#2a2a2a", elev=30, azim=-50)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, mesh, col, title in [
        (axes[0], black, "#333", "Černý rám"),
        (axes[1], white, "#bbb", "Bílá plocha"),
        (axes[2], base, "#333", "Podstavec"),
    ]:
        z = mesh.bounds[0][2] + max(mesh.extents[2] * 0.35, 0.5)
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is not None:
            path2d, _ = sec.to_2D()
            for e in path2d.entities:
                discrete = e.discrete(path2d.vertices)
                ax.plot(discrete[:, 0], discrete[:, 1], color=col, lw=0.7)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Škoda lightbox — větší než Audi (~Ø300 mm)")
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "skoda-rezy.png", dpi=140)
    plt.close(fig)


def validate(mesh: trimesh.Trimesh, name: str) -> None:
    print(
        f"{name}: extents={np.round(mesh.extents, 2)} "
        f"watertight={mesh.is_watertight} faces={len(mesh.faces)} "
        f"volume={mesh.volume:.0f}"
    )
    if not mesh.is_watertight:
        raise RuntimeError(f"{name} is not watertight")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    print("Building Škoda face…")
    r_outer, r_inner, arrow = build_face_parts_mm()
    face = face_polygon_mm(r_outer, r_inner, arrow)
    print(
        f"  R_outer={r_outer:.2f} R_inner={r_inner:.2f} "
        f"ring_w={r_outer - r_inner:.2f} face_area={face.area:.0f}"
    )

    print("Black housing…")
    black = cq_to_trimesh(make_black_cq(r_outer, r_inner, arrow))
    validate(black, "black")

    print("White diffuser…")
    white = cq_to_trimesh(dome_white(r_outer, r_inner, arrow), tol=0.15)
    validate(white, "white")

    print("Base…")
    base = cq_to_trimesh(make_base_cq(), tol=0.2)
    validate(base, "base")

    render_previews(face, black, white, base)

    desc = (
        "Skoda okridleny sip na podstavci, styl Audi lightbox. "
        f"Cerny ram {BLACK_Z:.0f} mm, bila plocha {WHITE_Z:.0f} mm, "
        f"prumer {BLACK_OUTER_D:.0f} mm (vetsi nez Audi 222 mm). "
        "Dva cepy do podstavce."
    )
    out = OUT_DIR / "skoda-logo-lightbox.3mf"
    write_3mf(
        [
            ("Skoda cerny ram", black, "black"),
            ("Skoda bila plocha", white, "white"),
            ("Skoda podstavec", base, "black"),
        ],
        out,
        "Skoda logo lightbox",
        desc,
    )
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")

    write_single_3mf(black, OUT_DIR / "skoda-logo-cerna.3mf", "Skoda cerny ram", "black")
    write_single_3mf(white, OUT_DIR / "skoda-logo-bila.3mf", "Skoda bila plocha", "white")
    write_single_3mf(base, OUT_DIR / "skoda-logo-podstavec.3mf", "Skoda podstavec", "black")
    print("Wrote separate part files.")


if __name__ == "__main__":
    main()
