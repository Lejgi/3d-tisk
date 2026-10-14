#!/usr/bin/env python3
"""Generate wall-mounted football medal holders as 3MF files.

One solid per name. Printed flat on an Ender 3 V3 SE, 0.4 mm nozzle.
Everything up to 6.0 mm is black. From 6.0 mm upward only the name and
the soccer-ball panels remain, so a single filament change turns them white.
Medal hooks are tabs on the bottom edge, inside the black layers.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cadquery as cq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "modely"
PREVIEW_DIR = ROOT / "nahledy"
FONT = "/usr/share/fonts/truetype/noto/NotoSansDisplay-Bold.ttf"

# Ender 3 V3 SE printable volume is 220 x 220 x 250 mm.
# Keep a margin so clips and a brim still fit.
BED_X = 220.0
BED_Y = 220.0

# Plaque lies flat: Z is up off the bed (pegs on top). When screwed to the
# wall, +Y is up and +Z points out of the wall.
W = 200.0
H = 180.0
T = 6.0
CORNER_R = 8.0

FRAME_H = 1.6

# Countersunk clearance for a 4 mm flat-head wood screw.
# Opening is wider than the head so the head sits below the face
# (45 degree walls, printable without supports).
HOLE_INSET_X = 24.0
HOLE_INSET_Y = 24.0
HOLE_D = 5.8
CSINK_D = 11.0
CSINK_DEPTH = (CSINK_D - HOLE_D) / 2.0  # 2.6 mm, 45 deg

BALL_R = 20.0
BALL_CX = 0.0
BALL_CY = 28.0
BALL_GAP = 1.2
# Filament change plane. Geometry above this is only the name and ball panels.
COLOR_Z = 6.0
WHITE_H = 1.6

PEG_Y = -50.0
PEG_XS = [-72.0, -48.0, -24.0, 0.0, 24.0, 48.0, 72.0]
SHAFT_R = 3.5
CAP_R = 5.7
BASE_R = 5.6

FONT_SIZE = 36.0
TEXT_H = 1.6
# Shared baseline so descenders (j, p) hang the same way on every plaque.
BASELINE_Y = -27.5

NAMES = ["Matěj", "Kuba", "Filip"]


def even_permutations(x: float, y: float, z: float) -> list[tuple[float, float, float]]:
    return [(x, y, z), (z, x, y), (y, z, x)]


def truncated_icosahedron_vertices() -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    seeds = [
        (0.0, 1.0, 3.0 * phi),
        (1.0, 2.0 + phi, 2.0 * phi),
        (2.0, 1.0 + 2.0 * phi, phi),
    ]
    pts: list[tuple[float, float, float]] = []
    for seed in seeds:
        for x, y, z in even_permutations(*seed):
            xs = [0.0] if abs(x) < 1e-9 else [x, -x]
            ys = [0.0] if abs(y) < 1e-9 else [y, -y]
            zs = [0.0] if abs(z) < 1e-9 else [z, -z]
            for sx in xs:
                for sy in ys:
                    for sz in zs:
                        pts.append((sx, sy, sz))
    arr = np.unique(np.round(np.array(pts, dtype=float), 8), axis=0)
    if len(arr) != 60:
        raise RuntimeError(f"expected 60 ball vertices, got {len(arr)}")
    return arr


def rotation_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = src / np.linalg.norm(src)
    b = dst / np.linalg.norm(dst)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(axis, a)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def hull_faces(points: np.ndarray) -> list[np.ndarray]:
    hull = ConvexHull(points)
    groups: dict[tuple[float, ...], list[int]] = {}
    for i, eq in enumerate(hull.equations):
        key = tuple(np.round(eq, 5))
        groups.setdefault(key, []).append(i)
    faces: list[np.ndarray] = []
    for idxs in groups.values():
        eq = hull.equations[idxs[0]]
        normal = eq[:3]
        normal = normal / np.linalg.norm(normal)
        vids = sorted({int(v) for i in idxs for v in hull.simplices[i]})
        pts = points[vids]
        center = pts.mean(axis=0)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(normal, tmp)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(normal, t1)
        rel = pts - center
        ang = np.arctan2(rel @ t2, rel @ t1)
        faces.append(pts[np.argsort(ang)])
    if len(faces) != 32:
        raise RuntimeError(f"expected 32 ball faces, got {len(faces)}")
    return faces


def soccer_panel_polygons(radius: float, gap: float) -> tuple[list[Polygon], int]:
    """Front view of a truncated icosahedron, pentagon centered and pointing up."""
    verts = truncated_icosahedron_vertices()
    faces = hull_faces(verts)
    pentagons = [f for f in faces if len(f) == 5]
    center_face = max(pentagons, key=lambda f: f.mean(axis=0)[2])
    normal = np.cross(center_face[1] - center_face[0], center_face[2] - center_face[0])
    if normal[2] < 0:
        normal = -normal
    rot = rotation_align(normal, np.array([0.0, 0.0, 1.0]))
    verts_r = verts @ rot.T
    faces_r = hull_faces(verts_r)
    pentagons = [f for f in faces_r if len(f) == 5]
    center_face = max(pentagons, key=lambda f: f.mean(axis=0)[2])
    vertex = center_face[np.argmax(np.linalg.norm(center_face[:, :2], axis=1))]
    ang = np.arctan2(vertex[0], vertex[1])
    c, s = np.cos(-ang), np.sin(-ang)
    zrot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    faces_r = hull_faces(verts_r @ zrot.T)

    projected: list[tuple[Polygon, float]] = []
    for face in faces_r:
        n = np.cross(face[1] - face[0], face[2] - face[0])
        n = n / np.linalg.norm(n)
        if n[2] < 0.20:
            continue
        poly = Polygon([(p[0], p[1]) for p in face])
        if not poly.is_valid or poly.area < 1e-6:
            continue
        projected.append((poly, n[2]))

    all_pts = np.vstack([np.array(p.exterior.coords) for p, _ in projected])
    scale = (radius * 0.90) / np.max(np.linalg.norm(all_pts, axis=1))
    panels: list[tuple[Polygon, float]] = []
    for poly, nz in projected:
        scaled = Polygon(np.array(poly.exterior.coords) * scale)
        inset = scaled.buffer(-gap / 2.0, join_style="mitre", mitre_limit=2.0)
        if inset.is_empty:
            continue
        geoms = [inset] if inset.geom_type == "Polygon" else list(inset.geoms)
        for g in geoms:
            if g.area < 3.0 or g.buffer(-0.55).is_empty:
                continue
            panels.append((g, nz))
    if not panels:
        raise RuntimeError("soccer ball produced no printable panels")
    # Center pentagon is the most front-facing panel.
    center_i = int(np.argmax([nz for _, nz in panels]))
    return [p for p, _ in panels], center_i


def rounded_plate(w: float, h: float, t: float, r: float) -> cq.Workplane:
    return cq.Workplane("XY").rect(w, h).extrude(t).edges("|Z").fillet(r)


def hook_at(x: float) -> cq.Workplane:
    """Downward ribbon hook in the plane of the plaque, entirely below the color change."""
    root = cq.Workplane("XY").center(x, -94.6).rect(11.0, 10.0).extrude(T)
    neck = cq.Workplane("XY").center(x, -103.0).rect(7.2, 8.0).extrude(T)
    head = cq.Workplane("XY").center(x, -109.2).rect(13.0, 6.2).extrude(T).edges("|Z").fillet(2.0)
    return root.union(neck).union(head)


def border_groove() -> cq.Workplane:
    outer = rounded_plate(W - 8.0, H - 8.0, 1.2, 4.0)
    inner = rounded_plate(W - 11.2, H - 11.2, 1.6, 2.4).translate((0, 0, -0.2))
    return outer.cut(inner).translate((0, 0, COLOR_Z - 0.6))


def ball_ring() -> cq.Workplane:
    outer = cq.Workplane("XY").circle(22.2).extrude(1.2).translate((BALL_CX, BALL_CY, COLOR_Z - 0.6))
    inner = cq.Workplane("XY").circle(20.6).extrude(1.6).translate((BALL_CX, BALL_CY, COLOR_Z - 0.8))
    return outer.cut(inner)


def polygon_solid(poly: Polygon, z0: float, height: float, dx: float = 0.0, dy: float = 0.0) -> cq.Workplane:
    if dx or dy:
        poly = affinity.translate(poly, xoff=dx, yoff=dy)

    def coords(ring) -> list[tuple[float, float]]:
        pts = [(float(x), float(y)) for x, y in list(ring.coords)[:-1]]
        if len(pts) < 3:
            raise RuntimeError("polygon ring has fewer than 3 points")
        return pts

    solid = cq.Workplane("XY").polyline(coords(poly.exterior)).close().extrude(height)
    for ring in poly.interiors:
        cutter = (
            cq.Workplane("XY")
            .polyline(coords(ring))
            .close()
            .extrude(height + 0.4)
            .translate((0, 0, -0.2))
        )
        solid = solid.cut(cutter)
    if z0:
        solid = solid.translate((0, 0, z0))
    return solid


def ball_disc() -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .circle(BALL_R)
        .extrude(BALL_BODY_H + 0.05)
        .translate((BALL_CX, BALL_CY, T - 0.05))
    )


def name_workplane(name: str) -> cq.Workplane:
    text = cq.Workplane("XY").text(
        name,
        FONT_SIZE,
        TEXT_H,
        fontPath=FONT,
        kind="regular",
        halign="center",
        valign="bottom",
    )
    return text.translate((0, BASELINE_Y, COLOR_Z))


def footprint(solid: cq.Shape) -> Polygon:
    """2D outline of an extruded glyph, including holes in letters."""
    verts, tris = solid.tessellate(0.04, 0.12)
    pts = np.array([(v.x, v.y, v.z) for v in verts])
    tri = pts[np.array(tris)]
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    normal = np.cross(edge1, edge2)
    length = np.linalg.norm(normal, axis=1)
    zmax = solid.BoundingBox().zmax
    top = (normal[:, 2] > 0.85 * np.maximum(length, 1e-9)) & (tri[:, :, 2].mean(axis=1) > zmax - 0.25)
    pieces = [Polygon([(p[0], p[1]) for p in t]) for t in tri[top] if Polygon([(p[0], p[1]) for p in t]).area > 1e-6]
    if not pieces:
        raise RuntimeError("glyph has no top face")
    merged = unary_union(pieces).buffer(0.03).buffer(-0.03)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    if not isinstance(merged, Polygon) or merged.is_empty:
        raise RuntimeError("glyph footprint collapsed")
    return merged


def merge_marks(polys: list[Polygon]) -> list[Polygon]:
    """Join carons and dots to their letters so each white piece is one part."""
    big: list[Polygon] = []
    small: list[Polygon] = []
    for poly in polys:
        height = poly.bounds[3] - poly.bounds[1]
        # Carons and dots are short. Letter bodies are much taller.
        if height < 8:
            small.append(poly)
        else:
            big.append(poly)
    if not big:
        return polys
    attached: dict[int, list[Polygon]] = {id(glyph): [] for glyph in big}
    leftover: list[Polygon] = []
    for mark in small:
        cx = mark.centroid.x
        hosts = [glyph for glyph in big if glyph.bounds[0] - 1.5 <= cx <= glyph.bounds[2] + 1.5]
        if not hosts:
            leftover.append(mark)
            continue
        host = min(hosts, key=lambda glyph: abs(glyph.centroid.x - cx))
        direction = -1.0 if mark.centroid.y > host.centroid.y else 1.0
        moved = mark
        for dist in np.arange(0.0, 8.01, 0.25):
            candidate = affinity.translate(mark, yoff=direction * dist)
            if candidate.intersects(host.buffer(0.05)):
                moved = affinity.translate(mark, yoff=direction * (dist + 0.45))
                break
        attached[id(host)].append(moved)
    merged: list[Polygon] = []
    for glyph in big:
        whole = unary_union([glyph, *attached[id(glyph)]]).buffer(0)
        if whole.geom_type == "MultiPolygon":
            merged.extend(whole.geoms)
        else:
            merged.append(whole)
    merged.extend(leftover)
    return merged


def glyphs_for(name: str) -> list[Polygon]:
    solids = name_workplane(name).solids().vals()
    polys = [footprint(solid) for solid in solids]
    print(f"  {name}: {len(polys)} tahů před spojením háčků")
    for poly in sorted(polys, key=lambda item: item.centroid.x):
        box = poly.bounds
        print(
            f"    plocha {poly.area:6.1f}  "
            f"x {box[0]:6.1f}..{box[2]:6.1f}  y {box[1]:6.1f}..{box[3]:6.1f}"
        )
    merged = merge_marks(polys)
    merged.sort(key=lambda poly: poly.centroid.x)
    print(f"  {name}: {len(merged)} bílých dílů jména")
    return merged


def fit_inlay(poly: Polygon) -> Polygon:
    for clearance in (INLAY_CLEARANCE, 0.20, 0.12):
        inset = poly.buffer(-clearance, join_style="round", quad_segs=6)
        if inset.is_empty:
            continue
        geoms = [inset] if inset.geom_type == "Polygon" else list(inset.geoms)
        geoms = [geom for geom in geoms if geom.area > 1.5 and not geom.buffer(-0.55).is_empty]
        if not geoms:
            continue
        whole = unary_union(geoms)
        if whole.geom_type != "Polygon":
            whole = max(whole.geoms, key=lambda geom: geom.area)
        if not poly.buffer(0.02).covers(whole):
            continue
        return whole
    raise RuntimeError("white inlay became too thin to print")


def ball_pockets(panels: list[Polygon]) -> list[cq.Workplane]:
    z0 = T + BALL_BODY_H - POCKET_DEPTH
    return [polygon_solid(poly, z0, POCKET_DEPTH + 0.5, BALL_CX, BALL_CY) for poly in panels]


def name_pockets(glyphs: list[Polygon]) -> list[cq.Workplane]:
    z0 = T - POCKET_DEPTH
    return [polygon_solid(poly, z0, POCKET_DEPTH + 0.5) for poly in glyphs]


def hole_cutter(x: float, y: float) -> cq.Workplane:
    through = (
        cq.Workplane("XY")
        .circle(HOLE_D / 2.0)
        .extrude(T + 4.0)
        .translate((x, y, -2.0))
    )
    # Large end of the cone is at the front face. 45 deg from horizontal.
    cone = (
        cq.Workplane("XY")
        .circle(HOLE_D / 2.0)
        .workplane(offset=CSINK_DEPTH)
        .circle(CSINK_D / 2.0)
        .loft(combine=True)
        .translate((x, y, T - CSINK_DEPTH))
    )
    return through.union(cone)


def fuse_many(parts: list[cq.Workplane]) -> cq.Workplane:
    result = parts[0]
    total = len(parts)
    for i, part in enumerate(parts[1:], start=2):
        print(f"  slučuji {i}/{total}", flush=True)
        result = result.union(part)
    return result


def check_layout(name_bb: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = name_bb
    if xmax - xmin > W - 36:
        raise RuntimeError(f"name is too wide: {xmax - xmin:.1f} mm")
    if ymax > BALL_CY - BALL_R - 4:
        raise RuntimeError("name collides with the ball")
    if ymin < -H / 2 + 8:
        raise RuntimeError("name collides with the bottom edge")
    if max(abs(x) for x in PEG_XS) + 6.5 > W / 2 - 8:
        raise RuntimeError("hooks extend past the plaque edge")
    top_limit = H / 2 - HOLE_INSET_Y - CSINK_D / 2 - 4
    if BALL_CY + BALL_R > top_limit:
        raise RuntimeError("ball collides with a mounting hole")


def build_black() -> cq.Workplane:
    parts: list[cq.Workplane] = [rounded_plate(W, H, T, CORNER_R)]
    for x in PEG_XS:
        parts.append(hook_at(x))
    print("  skládám desku a háčky", flush=True)
    body = fuse_many(parts)
    body = apply_holes(body)
    return body.cut(border_groove()).cut(ball_ring())


def build_white(name: str, panels: list[Polygon]) -> cq.Workplane:
    parts = [name_workplane(name)]
    parts.extend(polygon_solid(poly, COLOR_Z, WHITE_H, BALL_CX, BALL_CY) for poly in panels)
    print(f"  přidávám bílé jméno a panely ({len(parts)} dílů)", flush=True)
    return fuse_many(parts)


def apply_holes(body: cq.Workplane) -> cq.Workplane:
    hx = W / 2 - HOLE_INSET_X
    hy = H / 2 - HOLE_INSET_Y
    cutters = [hole_cutter(sx * hx, sy * hy) for sx in (-1, 1) for sy in (-1, 1)]
    tool = fuse_many(cutters)
    return body.cut(tool)


def shape_to_mesh(shape: cq.Workplane) -> trimesh.Trimesh:
    # 0.08 mm linear deflection keeps the name and ball smooth
    # without an oversized file.
    verts, faces = shape.val().tessellate(0.08, 0.15)
    mesh = trimesh.Trimesh(
        vertices=np.array([(v.x, v.y, v.z) for v in verts], dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def write_3mf(mesh: trimesh.Trimesh, path: Path, title: str, description: str) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        model = io.StringIO()
        model.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        model.write(
            '<model unit="millimeter" xml:lang="cs-CZ" '
            'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        )
        model.write(f"  <metadata name=\"Title\">{title}</metadata>\n")
        model.write(f"  <metadata name=\"Description\">{description}</metadata>\n")
        model.write('  <metadata name="Application">generate_drzak.py</metadata>\n')
        model.write(
            '  <metadata name="Printer">Creality Ender 3 V3 SE, tryska 0,4 mm</metadata>\n'
        )
        model.write('  <resources>\n')
        model.write(f'    <object id="1" type="model" name="{title}">\n')
        model.write("      <mesh>\n")
        model.write("        <vertices>\n")
        for x, y, z in vertices:
            model.write(f'          <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>\n')
        model.write("        </vertices>\n")
        model.write("        <triangles>\n")
        for a, b, c in faces:
            model.write(f'          <triangle v1="{a}" v2="{b}" v3="{c}"/>\n')
        model.write("        </triangles>\n")
        model.write("      </mesh>\n")
        model.write("    </object>\n")
        model.write("  </resources>\n")
        model.write("  <build>\n")
        model.write('    <item objectid="1"/>\n')
        model.write("  </build>\n")
        model.write("</model>\n")
        zf.writestr("3D/3dmodel.model", model.getvalue().encode("utf-8"))
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
""".encode("utf-8"),
        )
        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
""".encode("utf-8"),
        )
    path.write_bytes(buf.getvalue())


def save_ball_preview(panels: list[Polygon], center_i: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    disc = plt.Circle((BALL_CX, BALL_CY), BALL_R, facecolor="#d9dde3", edgecolor="#222")
    ax.add_patch(disc)
    for i, poly in enumerate(panels):
        moved = Polygon([(x + BALL_CX, y + BALL_CY) for x, y in poly.exterior.coords])
        xs, ys = moved.exterior.xy
        color = "#f4f1ea" if i == center_i else "#f7f7f5"
        ax.fill(xs, ys, facecolor=color, edgecolor="#222222", linewidth=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-W / 2, W / 2)
    ax.set_ylim(-H / 2, H / 2)
    ax.set_title("Fotbalový míč — panely")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def mesh_to_vtk(mesh: trimesh.Trimesh):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    pts = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(pts, deep=True))
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    packed = np.empty((len(faces), 4), dtype=np.int64)
    packed[:, 0] = 3
    packed[:, 1:] = faces
    cells = vtk.vtkCellArray()
    cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(packed.ravel(), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def render_views(mesh: trimesh.Trimesh, stem: str) -> None:
    import vtk

    poly = mesh_to_vtk(mesh)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    views = {
        "zepredu": {
            "pos": (center[0], center[1], center[2] + 420),
            "up": (0, 1, 0),
            "parallel": True,
        },
        "perspektiva": {
            "pos": (center[0] + 210, center[1] - 250, center[2] + 170),
            "up": (0, 0.45, 0.9),
            "parallel": False,
        },
        "zboku": {
            "pos": (center[0] + 420, center[1], center[2]),
            "up": (0, 0, 1),
            "parallel": True,
        },
    }
    for name, cam in views.items():
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(0.90, 0.91, 0.89)
        prop.SetAmbient(0.18)
        prop.SetDiffuse(0.72)
        prop.SetSpecular(0.18)
        prop.SetSpecularPower(18)
        prop.SetInterpolationToPhong()
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0.16, 0.17, 0.19)
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(cam["pos"])
        light.SetFocalPoint(*center)
        light.SetIntensity(1.0)
        renderer.AddLight(light)
        fill = vtk.vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(center[0] - 100, center[1] + 200, center[2] + 80)
        fill.SetFocalPoint(*center)
        fill.SetIntensity(0.45)
        renderer.AddLight(fill)
        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1)
        window.AddRenderer(renderer)
        window.SetSize(1600, 1200)
        camera = renderer.GetActiveCamera()
        camera.SetPosition(*cam["pos"])
        camera.SetFocalPoint(*center)
        camera.SetViewUp(*cam["up"])
        if cam["parallel"]:
            camera.ParallelProjectionOn()
        renderer.ResetCamera()
        if cam["parallel"]:
            camera.Zoom(1.15)
        else:
            camera.Zoom(1.25)
        window.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(window)
        w2i.SetScale(1)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        out = PREVIEW_DIR / f"{stem}_{name}.png"
        writer.SetFileName(str(out))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        window.Finalize()


def save_sections(mesh: trimesh.Trimesh) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    peg_section = mesh.section(plane_origin=[PEG_XS[3], 0, 0], plane_normal=[1, 0, 0])
    hole_x = W / 2 - HOLE_INSET_X
    hole_section = mesh.section(plane_origin=[hole_x, 0, 0], plane_normal=[1, 0, 0])
    for ax, section, title in (
        (axes[0], peg_section, "Řez středovým háčkem (Y vodorovně, Z nahoru)"),
        (axes[1], hole_section, "Řez montážním otvorem"),
    ):
        if section is None:
            ax.set_title(title + " — prázdný")
            continue
        for entity in section.entities:
            pts = section.vertices[entity.points]
            ax.plot(pts[:, 1], pts[:, 2], color="#1f4b99", linewidth=0.8)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3)
        ax.set_title(title)
        ax.set_xlabel("Y [mm]")
        ax.set_ylabel("Z [mm]")
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "rezy.png", dpi=140)
    plt.close(fig)


def validate_mesh(mesh: trimesh.Trimesh, name: str) -> None:
    extents = mesh.extents
    if extents[0] > BED_X - 4 or extents[1] > BED_Y - 4 or extents[2] > 40:
        raise RuntimeError(f"{name} exceeds the printer volume: {extents}")
    if mesh.bounds[0][2] < -0.05:
        raise RuntimeError(f"{name} extends below the bed")
    if not mesh.is_watertight:
        raise RuntimeError(f"{name} mesh is not watertight")
    if mesh.volume <= 0:
        raise RuntimeError(f"{name} has non-positive volume")
    print(
        f"{name}: {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm, "
        f"objem {mesh.volume / 1000:.1f} cm3, "
        f"trojuhelniku {len(mesh.faces)}"
    )


def filename_for(name: str) -> str:
    ascii_name = (
        name.replace("ě", "e")
        .replace("š", "s")
        .replace("č", "c")
        .replace("ř", "r")
        .replace("ž", "z")
        .replace("ý", "y")
        .replace("á", "a")
        .replace("í", "i")
        .replace("é", "e")
        .replace("ů", "u")
        .replace("ú", "u")
        .replace("ď", "d")
        .replace("ť", "t")
        .replace("ň", "n")
    )
    return f"drzak-medaile-{ascii_name}.3mf"


def layout_sheet(glyphs: list[Polygon], panels: list[Polygon]) -> tuple[list[Polygon], tuple[float, float]]:
    """Place white pieces flat, ball kept in its real arrangement, name underneath."""
    ball = [affinity.translate(poly, xoff=BALL_CX, yoff=BALL_CY) for poly in panels]
    ball_inlay = [fit_inlay(poly) for poly in ball]
    name_inlay = [fit_inlay(poly) for poly in glyphs]
    all_parts = ball_inlay + name_inlay
    minx = min(poly.bounds[0] for poly in all_parts)
    miny = min(poly.bounds[1] for poly in all_parts)
    maxx = max(poly.bounds[2] for poly in all_parts)
    maxy = max(poly.bounds[3] for poly in all_parts)
    # 8 mm margin from the sheet origin. Pieces keep their relative places,
    # so the name stays readable and each panel matches its pocket.
    return [affinity.translate(poly, xoff=8 - minx, yoff=8 - miny) for poly in all_parts], (maxx - minx + 16, maxy - miny + 16)


def meshes_from_parts(parts: list[cq.Workplane]) -> trimesh.Trimesh:
    meshes = [shape_to_mesh(part) for part in parts]
    combined = trimesh.util.concatenate(meshes)
    combined.merge_vertices()
    combined.update_faces(combined.unique_faces())
    combined.remove_unreferenced_vertices()
    combined.fix_normals()
    return combined


def render_color_preview(mesh: trimesh.Trimesh, path: Path) -> None:
    from PIL import Image, ImageDraw

    scale = 7
    xmin, xmax = -115.0, 115.0
    ymin, ymax = -125.0, 100.0
    image = Image.new("RGB", (int((xmax - xmin) * scale), int((ymax - ymin) * scale)), (214, 216, 214))
    draw = ImageDraw.Draw(image)
    triangles = mesh.vertices[mesh.faces]
    order = np.argsort(triangles[:, :, 2].mean(axis=1))
    for tri in triangles[order]:
        color = (245, 245, 242) if float(tri[:, 2].mean()) > COLOR_Z + 0.2 else (24, 24, 26)
        draw.polygon([((x - xmin) * scale, (ymax - y) * scale) for x, y, _z in tri], fill=color)
    image.save(path)


def assert_color_split(mesh: trimesh.Trimesh, name: str) -> None:
    triangles = mesh.vertices[mesh.faces]
    zmin = triangles[:, :, 2].min(axis=1)
    zmax = triangles[:, :, 2].max(axis=1)
    crossing = (zmin < COLOR_Z - 0.08) & (zmax > COLOR_Z + 0.08)
    if crossing.any():
        raise RuntimeError(f"{name}: {int(crossing.sum())} triangles cross the filament change")
    if mesh.bounds[1][2] < COLOR_Z + WHITE_H - 0.15:
        raise RuntimeError(f"{name}: white name and panels are missing")
    if mesh.extents[0] > 210 or mesh.extents[1] > 210:
        raise RuntimeError(f"{name} does not leave room for a brim: {mesh.extents}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    print("Skládám panely míče…")
    panels, center_i = soccer_panel_polygons(BALL_R, BALL_GAP)
    print(f"panelů: {len(panels)}, středový index: {center_i}")
    save_ball_preview(panels, center_i, PREVIEW_DIR / "mic-panely.png")

    probe = name_workplane("Matěj")
    bb = probe.val().BoundingBox()
    check_layout((bb.xmin, bb.xmax, bb.ymin, bb.ymax))
    print(
        f"Matěj bbox {bb.xmax - bb.xmin:.1f} x {bb.ymax - bb.ymin:.1f} mm, "
        f"Y {bb.ymin:.1f}..{bb.ymax:.1f}"
    )

    print("Sestavuji černou desku s háčky…")
    black = build_black()

    for name in NAMES:
        print(f"Sestavuji {name}…")
        body = black.union(build_white(name, panels))
        mesh = shape_to_mesh(body)
        validate_mesh(mesh, name)
        assert_color_split(mesh, name)
        out = OUT_DIR / filename_for(name)
        write_3mf(
            mesh,
            out,
            f"Drzak medaile {name}",
            (
                f"Drzak na medaile {name}. Tisknout cerne PLA do vysky {COLOR_Z:.1f} mm, "
                f"pak vymenit za bile PLA. Bile jsou jen jmeno a dily mice. "
                "Hacky jsou na spodni hrane. Bez podpor."
            ),
        )
        print(
            f"  ulozeno {out.name}, vymena filamentu ve vysce {COLOR_Z:.1f} mm, "
            f"rozmer {mesh.extents[0]:.0f} x {mesh.extents[1]:.0f} x {mesh.extents[2]:.1f} mm"
        )
        render_color_preview(mesh, PREVIEW_DIR / f"{filename_for(name).removesuffix('.3mf')}-shora.png")
        if name == "Matěj":
            save_sections(mesh)


if __name__ == "__main__":
    main()
