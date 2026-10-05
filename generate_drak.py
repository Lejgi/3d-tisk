#!/usr/bin/env python3
"""Artikulovaný drak dlouhý 750 mm s pevnými kulovými klouby.

Koule sedí v objímce až za rovníkem. Při ohýbání proto nevypadne,
uvnitř je ale vůle, takže se článek pořád točí. Tiskne se na břicho
na Ender 3 V3 SE (220×220×250), tryska 0,4 mm, PLA.
Trny, rohy i vějíř jsou součást těla, nic se na to nelepí.
Na podložku 220 mm se 75 cm nevejde, proto se tiskne po spojených kusech.
"""

from __future__ import annotations

import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "modely"
PREVIEW_DIR = ROOT / "nahledy"

BED = 220.0
MARGIN = 5.0
USABLE = BED - 2.0 * MARGIN
GAP = 3.2
TARGET_LEN = 750.0

BLACK = "#171717"
RED = "#D10E18"
EYE = "#FF2A1A"

# Kloub páteře. Rozměry jsou v mm a drží pohromadě i při ohnutí.
BALL_R = 4.80
CLEAR = 0.40
CAVITY_R = BALL_R + CLEAR
INTER = 0.62
LIP_R = BALL_R - INTER
LIP_DEG = 17.0
STEM_R = 2.42
BOWL_WALL = 2.55
RIM_T = 2.25
AXIS_Z = 8.55
SLIT_W = 1.22
SLIT_N = 4
# Ořez koule zespodu, ať spodní vrchlík neleží ve vzduchu.
CUT_DEG = 52.0
# Úzká patka, která při tisku spojeného kusu projde dnem objímky a po tisku se ulomí.
SPRUE_R = 2.85
HOLE_R = 3.60


@dataclass
class Joint:
    ball_r: float = BALL_R
    clear: float = CLEAR
    inter: float = INTER
    lip_deg: float = LIP_DEG
    stem_r: float = STEM_R
    bowl_wall: float = BOWL_WALL
    rim_t: float = RIM_T
    axis_z: float = AXIS_Z
    slit_w: float = SLIT_W
    slit_n: int = SLIT_N
    cut_deg: float = CUT_DEG

    @property
    def cavity_r(self) -> float:
        return self.ball_r + self.clear

    @property
    def lip_r(self) -> float:
        """Nejužší otvor. Menší než koule, větší než průřez koule v místě hrdla."""
        cross = math.sqrt(max(0.0, self.ball_r**2 - self.throat_x**2))
        return cross + max(0.28, self.clear * 0.75)

    @property
    def throat_x(self) -> float:
        # Hrdlo je až za rovníkem, proto koule po zacvaknutí nevypadne.
        return self.ball_r * 0.60

    @property
    def lip_x(self) -> float:
        return self.throat_x

    @property
    def face_x(self) -> float:
        return self.throat_x + 1.65

    @property
    def collar_r(self) -> float:
        return self.lip_r + 2.15

    def scaled(self, s: float) -> "Joint":
        j = Joint(
            ball_r=self.ball_r * s,
            clear=max(0.34, self.clear * s),
            inter=self.inter * s,
            lip_deg=self.lip_deg,
            stem_r=max(2.05, self.stem_r * s),
            bowl_wall=max(2.15, self.bowl_wall * s),
            rim_t=max(1.9, self.rim_t * s),
            axis_z=self.axis_z,
            slit_w=max(1.05, self.slit_w * s),
            slit_n=self.slit_n,
            cut_deg=self.cut_deg,
        )
        return j


SPINE = Joint()
LIMB = SPINE.scaled(0.90)


def _clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def unite(meshes: list[trimesh.Trimesh], label: str) -> trimesh.Trimesh:
    parts = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not parts:
        raise RuntimeError(f"{label}: nic ke spojení")
    if len(parts) == 1:
        return parts[0].copy()
    out = trimesh.boolean.union(parts, engine="manifold")
    if out is None or len(out.faces) == 0:
        raise RuntimeError(f"{label}: prázdné sjednocení")
    return out


def cut(solid: trimesh.Trimesh, voids: list[trimesh.Trimesh], label: str) -> trimesh.Trimesh:
    voids = [v for v in voids if v is not None and len(v.faces) > 0]
    if not voids:
        return solid.copy()
    out = trimesh.boolean.difference([solid, *voids], engine="manifold")
    if out is None or len(out.faces) == 0:
        raise RuntimeError(f"{label}: rozdíl smazal těleso")
    return out


def overlap_volume(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    inter = trimesh.boolean.intersection([a, b], engine="manifold")
    if inter is None or len(inter.faces) == 0:
        return 0.0
    return abs(float(inter.volume))


def box(cx, cy, cz, dx, dy, dz) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=(dx, dy, dz))
    m.apply_translation((cx, cy, cz))
    return m


def cyl_z(radius, z0, z1, sections=48, center=(0.0, 0.0)) -> trimesh.Trimesh:
    h = abs(z1 - z0)
    m = trimesh.creation.cylinder(radius=radius, height=h, sections=sections)
    m.apply_translation((center[0], center[1], (z0 + z1) / 2.0))
    return m


def cyl_x(radius, x0, x1, y=0.0, z=0.0, sections=40) -> trimesh.Trimesh:
    h = abs(x1 - x0)
    m = trimesh.creation.cylinder(radius=radius, height=h, sections=sections)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, [0, 1, 0]))
    m.apply_translation(((x0 + x1) / 2.0, y, z))
    return m


def sphere(radius, center, subdiv=3) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=radius)
    m.apply_translation(center)
    return m


def rot_matrix(axis, angle, origin=(0, 0, 0)):
    return trimesh.transformations.rotation_matrix(angle, axis, origin)


def apply_copy(mesh, matrix) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_transform(matrix)
    return m


def loft_solid(rings: list[np.ndarray]) -> trimesh.Trimesh:
    """Uzavřený loft. Všechny prstence mají stejný počet bodů a stejný smysl."""
    n = len(rings[0])
    for ring in rings:
        if len(ring) != n:
            raise RuntimeError("loft: nestejné prstence")
    verts = np.vstack(rings)
    faces: list[tuple[int, int, int]] = []

    def vid(r, i):
        return r * n + (i % n)

    for r in range(len(rings) - 1):
        for i in range(n):
            a = vid(r, i)
            b = vid(r, i + 1)
            c = vid(r + 1, i + 1)
            d = vid(r + 1, i)
            faces.append((a, b, c))
            faces.append((a, c, d))
    c0 = len(verts)
    c1 = c0 + 1
    verts = np.vstack([verts, rings[0].mean(axis=0), rings[-1].mean(axis=0)])
    last = len(rings) - 1
    for i in range(n):
        faces.append((c0, vid(0, i + 1), vid(0, i)))
        faces.append((c1, vid(last, i), vid(last, i + 1)))
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    _clean(mesh)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def rounded_rect_ring(x, width, height, radius, keel, n_corner=4) -> np.ndarray:
    """Obrys pancíře. Břicho je dole na z=0, nahoře je hřeben."""
    rad = min(radius, width * 0.5 - 0.4, height * 0.45)
    hw = width * 0.5
    z0 = 0.0
    z1 = height
    pts: list[list[float]] = []

    def corner(cx, cz, a0, a1):
        for i in range(n_corner + 1):
            a = a0 + (a1 - a0) * i / n_corner
            pts.append([x, cx + rad * math.cos(a), cz + rad * math.sin(a)])

    # Začíná na spodní hraně, jde proti směru hodinových ručiček při pohledu +X.
    y_l = -hw + rad
    y_r = hw - rad
    # spodní hrana zleva doprava
    pts.append([x, y_l, z0])
    pts.append([x, 0.0, z0])
    pts.append([x, y_r, z0])
    # pravý dolní roh, pravá stěna, pravý horní roh
    corner(y_r, rad, -math.pi / 2, 0.0)
    pts.append([x, hw, height * 0.55])
    corner(y_r, z1 - rad, 0.0, math.pi / 2)
    # horní hrana zprava doleva s hřebenem
    for t in (0.75, 0.5, 0.25):
        y = y_r + (y_l - y_r) * (1.0 - t) if False else y_r * t + y_l * (1 - t)
        # t=0.75 je blíž k pravé straně, chceme zprava doleva: y od y_r k y_l
    top_ys = [y_r * 0.72, y_r * 0.38, 0.0, y_l * 0.38, y_l * 0.72]
    for y in top_ys:
        u = 1.0 - abs(y) / max(hw, 1e-6)
        pts.append([x, y, z1 + keel * max(0.0, u) ** 1.35])
    corner(y_l, z1 - rad, math.pi / 2, math.pi)
    pts.append([x, -hw, height * 0.55])
    corner(y_l, rad, math.pi, math.pi * 1.5)
    # poslední bod rohu se potkává se začátkem; rohová funkce ho přidá, začátek je zvlášť.
    # Odstraň případný duplicitní bod na spoji.
    arr = np.array(pts, dtype=float)
    # vyhodit body příliš blízko následujícího
    keep = [0]
    for i in range(1, len(arr)):
        if np.linalg.norm(arr[i, 1:] - arr[keep[-1], 1:]) > 0.15:
            keep.append(i)
    arr = arr[keep]
    if np.linalg.norm(arr[0, 1:] - arr[-1, 1:]) < 0.15:
        arr = arr[:-1]
    return arr


def resample_ring(ring: np.ndarray, n: int) -> np.ndarray:
    """Sjednotí počet bodů po obvodu, ať jdou prstence loftovat."""
    p = ring[:, 1:]
    seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    per = float(seg.sum())
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    cum[-1] = per
    targets = np.linspace(0.0, per, n, endpoint=False)
    out = []
    j = 0
    m = len(p)
    for t in targets:
        while j < m - 1 and cum[j + 1] < t:
            j += 1
        span = cum[j + 1] - cum[j]
        u = 0.0 if span < 1e-9 else (t - cum[j]) / span
        yz = p[j] * (1 - u) + p[(j + 1) % m] * u
        out.append([ring[0, 0], yz[0], yz[1]])
    return np.array(out)


def teardrop_profile(radius: float, n: int = 28) -> np.ndarray:
    """Profil vodorovného otvoru. Špička míří nahoru, stěny mají 45°."""
    pts = []
    a0 = math.radians(45.0)
    a1 = math.radians(315.0)
    for i in range(n):
        a = a0 + (a1 - a0) * i / (n - 1)
        pts.append((radius * math.sin(a), radius * math.cos(a)))
    pts.append((0.0, radius * math.sqrt(2.0)))
    return np.array(pts)


def extrude_profile_x(profile: np.ndarray, x0: float, x1: float) -> trimesh.Trimesh:
    ring0 = np.column_stack([np.full(len(profile), x0), profile[:, 0], profile[:, 1]])
    ring1 = np.column_stack([np.full(len(profile), x1), profile[:, 0], profile[:, 1]])
    return loft_solid([ring0, ring1])


def frustum_teardrop(r0, x0, r1, x1) -> trimesh.Trimesh:
    p0 = teardrop_profile(r0)
    p1 = teardrop_profile(r1)
    ring0 = np.column_stack([np.full(len(p0), x0), p0[:, 0], p0[:, 1]])
    ring1 = np.column_stack([np.full(len(p1), x1), p1[:, 0], p1[:, 1]])
    return loft_solid([ring0, ring1])


def roof_void(joint: Joint, center) -> trimesh.Trimesh:
    """Kužel nad dutinou. Strop otvoru má 45°, takže se netiskne do vzduchu."""
    r = joint.cavity_r
    h0 = r * math.sin(math.radians(45.0))
    r0 = r * math.cos(math.radians(45.0))
    cone = trimesh.creation.cone(radius=r0, height=r0, sections=40)
    cone.apply_translation((center[0], center[1], center[2] + h0))
    return cone


def socket_voids(joint: Joint) -> list[trimesh.Trimesh]:
    """Mísa, úzké hrdlo za rovníkem a zářezy. Osa X, otvor ve směru +X."""
    sph = sphere(joint.cavity_r, (0, 0, 0), subdiv=3)
    # Mísa končí v hrdle. Dál už je jen úzký otvor, kterým rovník koule neprojde.
    cap = box(joint.throat_x + 25.0, 0.0, 0.0, 50.0, 50.0, 50.0)
    bowl = cut(sph, [cap], "mísa")
    voids = [
        bowl,
        roof_void(joint, (0, 0, 0)),
        frustum_teardrop(
            joint.lip_r,
            joint.throat_x - 0.30,
            joint.lip_r,
            joint.throat_x + joint.rim_t,
        ),
        frustum_teardrop(
            joint.lip_r,
            joint.throat_x + joint.rim_t - 0.08,
            min(joint.lip_r + 0.55, joint.ball_r - 0.18),
            joint.face_x + 0.7,
        ),
    ]
    radial = joint.cavity_r + joint.bowl_wall + 2.5
    x0 = joint.throat_x - 0.15
    x1 = joint.face_x + 0.45
    x_mid = (x0 + x1) * 0.5
    x_len = (x1 - x0)
    for i in range(joint.slit_n):
        ang = math.radians(45.0 + i * (360.0 / joint.slit_n))
        plate = box(x_mid, 0.0, radial * 0.38, x_len, joint.slit_w, radial * 0.85)
        plate.apply_transform(rot_matrix([1, 0, 0], ang))
        voids.append(plate)
        hole = cyl_x(0.70, joint.throat_x - 1.3, joint.throat_x - 0.15, sections=16)
        hole.apply_translation(
            (0.0, math.sin(ang) * (joint.lip_r + 0.2), math.cos(ang) * (joint.lip_r + 0.2))
        )
        voids.append(hole)
    return voids


def transform_voids(voids, matrix) -> list[trimesh.Trimesh]:
    return [apply_copy(v, matrix) for v in voids]


def socket_matrix(center, direction) -> np.ndarray:
    """Lokální +X (směr otvoru) natočí do `direction` a posune do `center`."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    x = np.array([1.0, 0.0, 0.0])
    c = float(np.dot(x, d))
    if c > 0.999:
        rot = np.eye(4)
    elif c < -0.999:
        rot = rot_matrix([0, 0, 1], math.pi)
    else:
        axis = np.cross(x, d)
        axis = axis / np.linalg.norm(axis)
        rot = rot_matrix(axis, math.acos(c))
    rot[:3, 3] = center
    return rot


def ball_mesh(joint: Joint) -> trimesh.Trimesh:
    """Koule s plochým spodkem. Ploska je schovaná v patce a do objímky nedosáhne."""
    c = np.array([0.0, 0.0, joint.axis_z])
    ball = sphere(joint.ball_r, c, subdiv=3)
    beta = math.radians(joint.cut_deg)
    plane = joint.axis_z - joint.ball_r * math.cos(beta)
    cutter = box(0, 0, plane - 20, 40, 40, 40)
    return cut(ball, [cutter], "ořez koule")


def ball_plane(joint: Joint) -> float:
    return joint.axis_z - joint.ball_r * math.cos(math.radians(joint.cut_deg))


def print_support(joint: Joint, neck_x1: float, pitch: float) -> trimesh.Trimesh:
    """Patka vyplní všechno, co by jinak viselo ve vzduchu. Po tisku se uřízne."""
    plane = ball_plane(joint)
    flat_r = joint.ball_r * math.sin(math.radians(joint.cut_deg)) + 0.25
    puck = cyl_z(flat_r, 0.0, plane + 0.15, sections=40)
    # pod dříkem a předním krčkem, až do výšky osy
    z_top = joint.axis_z + 0.4
    front = box(neck_x1 * 0.5, 0.0, z_top * 0.5, neck_x1 + flat_r, joint.collar_r * 1.15, z_top)
    # pod zadním krčkem u objímky
    x0 = pitch - 7.2
    back = box((x0 + pitch + joint.face_x) * 0.5, 0.0, z_top * 0.5, (pitch + joint.face_x - x0), joint.collar_r * 1.2, z_top)
    return unite([puck, front, back], "patka")


def stem_mesh(joint: Joint, x0: float, x1: float) -> trimesh.Trimesh:
    return cyl_x(joint.stem_r, x0, x1, z=joint.axis_z, sections=36)


def housing_solid(joint: Joint, x_center: float) -> trimesh.Trimesh:
    """Pouzdro. Spodní vrchlík je nahrazený svislou sukní až na podložku."""
    r = joint.cavity_r + joint.bowl_wall
    ball = sphere(r, (x_center, 0.0, joint.axis_z), subdiv=2)
    z_cut = joint.axis_z - r * math.cos(math.radians(50.0))
    r_sec = r * math.sin(math.radians(50.0))
    below = box(x_center, 0.0, z_cut * 0.5 - 2.0, r * 3.2, r * 3.2, z_cut + 4.0)
    ball = cut(ball, [below], "pouzdro spodek")
    skirt = cyl_z(r_sec + 0.08, 0.0, z_cut + 0.25, sections=40, center=(x_center, 0.0))
    return unite([ball, skirt], "pouzdro")


def face_cutter(joint: Joint, x_center: float) -> trimesh.Trimesh:
    x = x_center + joint.face_x
    return box(x + 30, 0, 20, 60, 80, 80)


def collar_void(joint: Joint, x_center: float) -> trimesh.Trimesh:
    """U otvoru nechá jen prstenec. Vysoká stěna by se při ohnutí hned srazila."""
    slab = box(
        x_center + joint.face_x - 1.1,
        0.0,
        joint.axis_z,
        5.5,
        48,
        48,
    )
    keep = cyl_x(
        joint.collar_r,
        x_center + joint.throat_x - 0.5,
        x_center + joint.face_x + 2.0,
        z=joint.axis_z,
        sections=40,
    )
    return cut(slab, [keep], "límec")


def dragon_ring(x, width, height, keel) -> np.ndarray:
    """Pancíř: ploché břicho, kulaté boky, hřbet s kýlem."""
    n = 36
    hw = width * 0.5
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        s = math.sin(t)
        c = math.cos(t)  # +1 nahoře, -1 na břiše
        belly = max(0.0, -c)
        y = s * hw * (1.0 - 0.22 * belly)
        if c > -0.05:
            u = (c + 0.05) / 1.05
            z = height * (0.16 + 0.84 * max(u, 0.0) ** 0.9)
            z += keel * max(c, 0.0) ** 1.7
        else:
            z = 0.0
        pts.append([x, y, z])
    return np.array(pts)


def armor_loft(x0, x1, width, height, keel, axis_z) -> trimesh.Trimesh:
    """Konce jsou úzký krček kolem osy kloubu, prostředek leží na podložce."""
    ts = [0.0, 0.10, 0.24, 0.42, 0.62, 0.82, 1.0]
    scales = [0.40, 0.62, 0.88, 1.0, 1.02, 0.78, 0.42]
    rings = []
    for t, s in zip(ts, scales):
        x = x0 + (x1 - x0) * t
        h = height * (0.58 + 0.42 * s)
        lift = 0.0 if s > 0.85 else max(0.0, axis_z - h * 0.62)
        ring = dragon_ring(x, max(11.0, width * s), h, keel * s)
        ring[:, 2] += lift
        rings.append(ring)
    return loft_solid(rings)


def scythe(length, width, thick, lean_deg, z0, x0=0.0, y0=0.0, curl=0.16, yaw_deg=0.0) -> trimesh.Trimesh:
    """Zahnutý trn. Patka je zanořená do pancíře, ostří míří dozadu."""
    n = 9
    rings = []
    yaw = math.radians(yaw_deg)
    for i in range(n):
        t = i / (n - 1)
        ang = math.radians(lean_deg) * (0.82 + 0.18 * t)
        dist = length * t
        back = dist * math.sin(ang) + curl * length * t * t
        up = dist * math.cos(ang)
        center = np.array(
            [x0 + back * math.cos(yaw), y0 + back * math.sin(yaw), z0 + up],
            dtype=float,
        )
        heading = np.array(
            [math.sin(ang) * math.cos(yaw), math.sin(ang) * math.sin(yaw), math.cos(ang)],
            dtype=float,
        )
        side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        upv = np.cross(heading, side)
        norm = np.linalg.norm(upv)
        if norm < 1e-8:
            upv = np.array([0.0, 0.0, 1.0])
        else:
            upv = upv / norm
        w = width * (1 - t) ** 0.72 + 0.9
        th = thick * (1 - t) ** 0.75 + 0.7
        # šířka je ve směru ostří (z boku viditelná), tloušťka je do stran
        ring = [
            center - side * (th * 0.5) - upv * (w * 0.42),
            center + side * (th * 0.45) - upv * (w * 0.05) + heading * (th * 0.35),
            center + side * (th * 0.5) + upv * (w * 0.48),
            center - side * (th * 0.4) + upv * (w * 0.12),
        ]
        rings.append(np.array(ring))
    return loft_solid(rings)


def sprue_mesh(joint: Joint) -> trimesh.Trimesh:
    """Tenká patka pod koulí. Projde dírou v objímce a po tisku se ulomí."""
    plane = ball_plane(joint)
    return cyl_z(SPRUE_R, 0.0, plane + 0.12, sections=24)


def socket_sprue_hole(joint: Joint, x_center: float) -> trimesh.Trimesh:
    return cyl_z(HOLE_R, -0.4, joint.axis_z - 1.85, sections=28, center=(x_center, 0.0))


def neck_rib(joint: Joint, x0: float, x1: float) -> trimesh.Trimesh:
    """Žebro pod krčkem. Drží ho při tisku a pak se ulomí spolu s patkou."""
    ztop = joint.axis_z * 0.92
    return box((x0 + x1) * 0.5, 0.0, ztop * 0.5, max(1.2, x1 - x0), 1.8, ztop)


def square_peg(x, y, z0, height, size_bot, size_top) -> trimesh.Trimesh:
    """Mírně kuželový hranol. Špička navede do otvoru, pata drží."""
    n = 4
    rings = []
    for i, (z, s) in enumerate(((z0, size_bot), (z0 + height, size_top))):
        h = s / 2.0
        ring = np.array(
            [
                [x - h, y - h, z],
                [x + h, y - h, z],
                [x + h, y + h, z],
                [x - h, y + h, z],
            ]
        )
        rings.append(ring)
    # loft čtverce; čtverec má 4 body
    return loft_solid(rings)


def digit_boxes(text: str, x, y_face, z, depth=0.7, h=3.4, w=2.15, t=0.42) -> list[trimesh.Trimesh]:
    """Číslice vyříznuté do boku. 1 je u hlavy. Zářezy jsou vodorovné, bez stropu dolů."""
    glyphs = {
        "0": "abcedf",
        "1": "bc",
        "2": "abged",
        "3": "abgcd",
        "4": "fgbc",
        "5": "afgcd",
        "6": "afgecd",
        "7": "abc",
        "8": "abcdefg",
        "9": "abfgcd",
    }
    boxes = []
    cursor = x
    for ch in text:
        segs = glyphs.get(ch, "")
        cx = cursor
        yc = y_face + depth / 2.0
        def add(dx, dz, ox, oz):
            boxes.append(box(cx + ox, yc, z + oz, dx, depth, dz))
        if "a" in segs:
            add(w, t, 0, h / 2)
        if "d" in segs:
            add(w, t, 0, -h / 2)
        if "g" in segs:
            add(w, t, 0, 0)
        if "f" in segs:
            add(t, h / 2, -w / 2, h / 4)
        if "b" in segs:
            add(t, h / 2, w / 2, h / 4)
        if "e" in segs:
            add(t, h / 2, -w / 2, -h / 4)
        if "c" in segs:
            add(t, h / 2, w / 2, -h / 4)
        cursor += w + 0.85
    return boxes


def peg_pair(x_mid, z_top, joint_scale=1.0) -> list[trimesh.Trimesh]:
    """Dva různě tlusté kolíky. Hřeben nejde nasadit obráceně."""
    big = 2.72 * max(0.85, joint_scale)
    small = 2.18 * max(0.85, joint_scale)
    h = 4.7
    return [
        square_peg(x_mid - 3.6, 0.0, z_top, h, big, big * 0.72),
        square_peg(x_mid + 3.6, 0.0, z_top, h, small, small * 0.72),
    ]


def make_segment(
    pitch, width, height, keel, index, sides: tuple[bool, bool], joint: Joint, spine_len=0.0
):
    """Článek páteře. Koule v x=0, objímka v x=pitch. Trny jsou součást těla."""
    x_body0 = 6.9
    x_body1 = pitch - 6.0
    if x_body1 - x_body0 < 2.4:
        raise RuntimeError(f"článek {index}: rozteč {pitch:.1f} je krátká")
    body = armor_loft(x_body0, x_body1, width, height, keel, joint.axis_z)
    ball = ball_mesh(joint)
    stem = stem_mesh(joint, 1.6, x_body0 + 0.8)
    house = housing_solid(joint, pitch)
    positives = [body, ball, stem, house]
    reds: list[trimesh.Trimesh] = []
    if spine_len > 8.0:
        mid = x_body0 + (x_body1 - x_body0) * 0.58
        z_root = height * 0.62
        reds.append(scythe(spine_len, max(8.0, width * 0.38), 2.8, 36.0, z_root, mid, 0.0, 0.12, 0.0))
        reds.append(
            scythe(spine_len * 0.55, max(6.0, width * 0.22), 2.2, 40.0, height * 0.42, mid - 0.6, 0.0, 0.06, 50.0)
        )
        reds.append(
            scythe(spine_len * 0.55, max(6.0, width * 0.22), 2.2, 40.0, height * 0.42, mid - 0.6, 0.0, 0.06, -50.0)
        )
    voids = transform_voids(
        socket_voids(joint),
        socket_matrix((pitch, 0.0, joint.axis_z), (1, 0, 0)),
    )
    voids.append(face_cutter(joint, pitch))
    voids.append(collar_void(joint, pitch))
    voids.append(socket_sprue_hole(joint, pitch))
    # boční objímky nohou
    limb = LIMB
    # boční kloub má vlastní výšku osy shodnou s páteří, ať noha sedí uprostřed
    limb = Joint(
        ball_r=limb.ball_r,
        clear=limb.clear,
        inter=limb.inter,
        lip_deg=limb.lip_deg,
        stem_r=limb.stem_r,
        bowl_wall=limb.bowl_wall,
        rim_t=limb.rim_t,
        axis_z=joint.axis_z,
        slit_w=limb.slit_w,
        slit_n=limb.slit_n,
        cut_deg=limb.cut_deg,
    )
    x_hip = (x_body0 + x_body1) * 0.48
    for sign, enabled in ((1.0, sides[0]), (-1.0, sides[1])):
        if not enabled:
            continue
        y_face = sign * (width * 0.48)
        # střed dutiny je zanořený, čelo je na boku těla
        y_c = y_face - sign * limb.face_x
        center = np.array([x_hip, y_c, joint.axis_z])
        boss_r = limb.cavity_r + limb.bowl_wall
        boss = cyl_z(boss_r, 0.0, joint.axis_z + boss_r * 0.15, sections=36, center=(x_hip, y_c))
        positives.append(boss)
        voids.extend(
            transform_voids(socket_voids(limb), socket_matrix(center, (0, sign, 0)))
        )
        # oříznout čelo objímky, ať zůstane rovný věnec
        if sign > 0:
            voids.append(box(x_hip, y_face + 25, 20, 50, 50, 70))
        else:
            voids.append(box(x_hip, y_face - 25, 20, 50, 50, 70))
    label = f"{index:02d}"
    voids.extend(
        digit_boxes(label, (x_body0 + x_body1) * 0.5 - 2.2, -width * 0.5 + 0.15, height * 0.48)
    )
    functional = cut(unite(positives, f"článek {index}"), voids, f"otvory {index}")
    support = segment_support(joint, x_body0, pitch, narrow=False)
    parts = [functional, support, *reds]
    printable = unite(parts, f"tisk {index}")
    return printable, functional, reds


def segment_support(joint: Joint, x_body0: float, pitch: float, narrow: bool, rear: bool = True) -> trimesh.Trimesh:
    """Patka pod koulí. Zadní žebro jen když v objímce při tomto tisku nic nesedí."""
    parts = []
    z_top = joint.axis_z + 0.4
    if narrow:
        parts.append(sprue_mesh(joint))
    else:
        plane = ball_plane(joint)
        flat_r = joint.ball_r * math.sin(math.radians(joint.cut_deg)) + 0.25
        parts.append(cyl_z(flat_r, 0.0, plane + 0.15, sections=32))
        parts.append(box(x_body0 * 0.42, 0.0, z_top * 0.5, max(4.0, x_body0 * 0.75), 8.0, z_top))
    rib_end = min(x_body0 + 3.2, pitch - 7.5)
    if rib_end > 7.4:
        parts.append(neck_rib(joint, 7.2, rib_end))
    if rear:
        x0 = max(x_body0 + 1.5, pitch - 6.2)
        if pitch - x0 > 1.5:
            parts.append(box((x0 + pitch) * 0.5, 0.0, z_top * 0.5, pitch - x0, 9.0, z_top))
    return unite(parts, "patka")


def blade(length, base_w, base_t, lean_deg, sweep_deg=0.0, z0=0.0) -> trimesh.Trimesh:
    """Červený trn. Náklon je od svislice, tiskne se bez podpěr."""
    n = 10
    rings = []
    for i in range(n):
        t = i / (n - 1)
        ang = math.radians(lean_deg) * (0.25 + 0.75 * t)
        sw = math.radians(sweep_deg) * t
        z = z0 + length * t * math.cos(ang)
        x = length * t * math.sin(ang)
        y = length * t * math.sin(sw) * 0.35
        w = base_w * (1 - t) ** 0.85 + 0.8
        th = base_t * (1 - t) ** 0.75 + 0.7
        ring = np.array(
            [
                [x - w * 0.5, y - th * 0.5, z],
                [x + w * 0.35, y - th * 0.35, z + th * 0.15],
                [x + w * 0.15, y + th * 0.5, z + w * 0.08],
                [x - w * 0.45, y + th * 0.35, z],
            ]
        )
        rings.append(ring)
    return loft_solid(rings)


def crest(length_scale, label: str) -> trimesh.Trimesh:
    """Hřeben se třemi trny sklopenými k ocasu, jako hřbet na předloze."""
    base_h = 5.8
    base = box(0, 0, base_h / 2, 16.2 * length_scale, 12.4 * length_scale, base_h)
    big = 2.72
    small = 2.18
    # otvor je o 0,16 mm menší než pata kolíku
    holes = [
        square_peg(-3.6, 0.0, -0.2, 5.5, big - 0.16, big - 0.16),
        square_peg(3.6, 0.0, -0.2, 5.5, small - 0.16, small - 0.16),
    ]
    mouths = [
        cyl_z((big - 0.16) * 0.85, -0.15, 1.3, sections=16, center=(-3.6, 0)),
        cyl_z((small - 0.16) * 0.85, -0.15, 1.15, sections=16, center=(3.6, 0)),
    ]
    spines = [
        blade(31.0 * length_scale, 9.2 * length_scale, 3.1, 40.0, 0.0, base_h - 0.3),
        blade(21.0 * length_scale, 6.4 * length_scale, 2.4, 38.0, 34.0, base_h - 0.3),
        blade(21.0 * length_scale, 6.4 * length_scale, 2.4, 38.0, -34.0, base_h - 0.3),
    ]
    spines[1].apply_translation((1.2, 2.8, 0))
    spines[2].apply_translation((1.2, -2.8, 0))
    body = unite([base, *spines], f"hřeben {label}")
    return cut(body, holes + mouths, f"dírky hřebene {label}")


def horn(length, base_r, tip_r, peg, label) -> trimesh.Trimesh:
    """Roh podél +Z. Dírka je zespodu, při tisku leží pata na podložce."""
    base_h = 8.0
    base = cyl_z(base_r, 0.0, base_h, sections=36)
    n = 8
    rings = []
    for i in range(n):
        t = i / (n - 1)
        z = base_h + (length - 2.0) * t
        r = base_r * (1 - t) + tip_r * t
        # mírný oblouk dozadu, pořád v bezpečném náklonu
        x = 0.22 * length * t * t
        ring = []
        for k in range(20):
            a = 2 * math.pi * k / 20
            ring.append([x + r * math.cos(a), r * math.sin(a) * 0.78, z])
        rings.append(np.array(ring))
    cone = loft_solid(rings)
    hole = square_peg(0, 0, -0.3, 6.4, peg - 0.10, peg - 0.10)
    mouth = cyl_z(peg * 0.55, -0.2, 1.4, sections=16)
    return cut(unite([base, cone], label), [hole, mouth], label)


def map_z_to_dir(mesh: trimesh.Trimesh, direction, origin) -> trimesh.Trimesh:
    """Posadí díl postavený podél +Z na dané místo a směr."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(np.dot(z, d), -1.0, 1.0))
    m = mesh.copy()
    if c < 0.999:
        if c < -0.999:
            rot = rot_matrix([1.0, 0.0, 0.0], math.pi)
        else:
            axis = np.cross(z, d)
            axis = axis / np.linalg.norm(axis)
            rot = rot_matrix(axis, math.acos(c))
        m.apply_transform(rot)
    m.apply_translation(np.asarray(origin, float))
    return m


def orient_axis_up(mesh: trimesh.Trimesh, direction) -> trimesh.Trimesh:
    """Natočí `direction` do +Z a položí díl na podložku."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(np.dot(d, z), -1, 1))
    if c > 0.999:
        rot = np.eye(4)
    elif c < -0.999:
        rot = rot_matrix([1, 0, 0], math.pi)
    else:
        axis = np.cross(d, z)
        axis = axis / np.linalg.norm(axis)
        rot = rot_matrix(axis, math.acos(c))
    m = apply_copy(mesh, rot)
    m.apply_translation((0, 0, -m.bounds[0, 2]))
    return m


def head_mesh(joint: Joint):
    """Hlava v jednom kuse. Rohy a oči jsou s ní srostlé, na náhledu jsou červené."""
    stations = [
        (-124, 7.5, 8.5, 0.4),
        (-110, 12.0, 11.0, 0.9),
        (-94, 17.0, 15.0, 1.4),
        (-78, 28.0, 20.0, 2.2),
        (-62, 42.0, 30.0, 3.6),
        (-46, 52.0, 44.0, 6.4),
        (-32, 46.0, 38.0, 4.6),
        (-18, 30.0, 26.0, 2.6),
        (-6, 18.0, 20.0, 1.3),
    ]
    rings = [resample_ring(dragon_ring(x, w, h, k), 40) for x, w, h, k in stations]
    skull = loft_solid(rings)
    brow = scale_about(sphere(11.0, (-42, 0, 32), subdiv=2), (1.45, 1.2, 0.5), (-42, 0, 32))
    jaw = scale_about(sphere(8.5, (-70, 0, 6.2), subdiv=2), (2.4, 1.2, 0.72), (-70, 0, 6.2))
    house = housing_solid(joint, 0.0)
    horns = [
        scythe(62, 10.0, 4.6, 28, 34.0, -40, 12.0, 0.22, 16),
        scythe(62, 10.0, 4.6, 28, 34.0, -40, -12.0, 0.22, -16),
        scythe(40, 7.0, 3.6, 24, 38.0, -30, 6.0, 0.16, 7),
        scythe(40, 7.0, 3.6, 24, 38.0, -30, -6.0, 0.16, -7),
        scythe(20, 4.2, 2.4, 32, 10.0, -102, 0.0, 0.06, 180),
        scythe(24, 6.5, 2.6, 32, 28.0, -20, 0.0, 0.12, 0),
        scythe(18, 5.0, 2.2, 34, 22.0, -10, 0.0, 0.10, 0),
        scythe(16, 5.2, 2.2, 42, 16.0, -58, 12.0, 0.06, 64),
        scythe(16, 5.2, 2.2, 42, 16.0, -58, -12.0, 0.06, -64),
    ]
    eye_centers = [(-62, 15.5, 18.0), (-62, -15.5, 18.0)]
    eyes = [sphere(3.45, c, subdiv=3) for c in eye_centers]
    voids = transform_voids(socket_voids(joint), socket_matrix((0, 0, joint.axis_z), (1, 0, 0)))
    voids.append(face_cutter(joint, 0.0))
    voids.append(collar_void(joint, 0.0))
    voids.append(socket_sprue_hole(joint, 0.0))
    for s in (1.0, -1.0):
        voids.append(box(-86, s * 13.5, 9.0, 36, 8.0, 2.4))
    voids.append(sphere(1.6, (-108, 2.6, 8.0), subdiv=2))
    voids.append(sphere(1.6, (-108, -2.6, 8.0), subdiv=2))
    for c in eye_centers:
        voids.append(sphere(2.5, c, subdiv=2))
    teeth = [tooth(x, 0.0, 10.4) for x in np.linspace(-108, -72, 7)]
    solid = unite([skull, brow, jaw, house, *teeth], "hlava")
    solid = cut(solid, [box(0, 0, -15, 280, 140, 30)], "podložka hlavy")
    functional = cut(solid, voids, "hlava otvory")
    dress = [(h, RED) for h in horns] + [(e, EYE) for e in eyes]
    return functional, dress


def cyl_between(p0, p1, radius, sections=20) -> trimesh.Trimesh:
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    h = np.linalg.norm(d)
    m = trimesh.creation.cylinder(radius=radius, height=h, sections=sections)
    # válec je podél Z, střed v 0
    direction = d / h
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(np.dot(z, direction), -1, 1))
    if c > 0.999:
        rot = np.eye(4)
    elif c < -0.999:
        rot = rot_matrix([1, 0, 0], math.pi)
    else:
        axis = np.cross(z, direction)
        axis = axis / np.linalg.norm(axis)
        rot = rot_matrix(axis, math.acos(c))
    m.apply_transform(rot)
    m.apply_translation((p0 + p1) / 2.0)
    return m


def square_along(p0, p1, size) -> trimesh.Trimesh:
    """Hranol mezi dvěma body, čtvercový průřez."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    h = np.linalg.norm(d)
    # loft dvou čtverců podél Z a otočit
    s0 = size / 2
    s1 = size * 0.34
    def square_at(z, s):
        return np.array([[-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]])
    mesh = loft_solid([square_at(0, s0), square_at(h, s1)])
    direction = d / h
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(np.dot(z, direction), -1, 1))
    if c > 0.999:
        rot = np.eye(4)
    elif c < -0.999:
        rot = rot_matrix([1, 0, 0], math.pi)
    else:
        axis = np.cross(z, direction)
        axis = axis / np.linalg.norm(axis)
        rot = rot_matrix(axis, math.acos(np.clip(c, -1, 1)))
    mesh.apply_transform(rot)
    mesh.apply_translation(p0)
    return mesh


def scale_about(mesh: trimesh.Trimesh, factors, origin) -> trimesh.Trimesh:
    origin = np.asarray(origin, float)
    mesh.apply_translation(-origin)
    mesh.apply_scale(factors)
    mesh.apply_translation(origin)
    return mesh


def tooth(x, y, z) -> trimesh.Trimesh:
    cone = trimesh.creation.cone(radius=0.9, height=2.4, sections=10)
    cone.apply_translation((x, y, z))
    return cone


def limb_segment(pitch, width, height, index, joint: Joint, peg=True):
    x0, x1 = 7.2, pitch - 6.4
    # noha používá vlastní výšku osy, břicho na podložce
    j = Joint(
        ball_r=joint.ball_r,
        clear=joint.clear,
        inter=joint.inter,
        lip_deg=joint.lip_deg,
        stem_r=joint.stem_r,
        bowl_wall=joint.bowl_wall,
        rim_t=joint.rim_t,
        axis_z=max(joint.axis_z, joint.cavity_r + joint.bowl_wall * 0.15 + 0.4),
        slit_w=joint.slit_w,
        slit_n=joint.slit_n,
        cut_deg=joint.cut_deg,
    )
    # u nohy snížíme osu tak, aby pouzdro došlo na podložku a strop zůstal v těle
    lo = j.cavity_r + 1.6
    hi = height - j.cavity_r - 1.3
    if hi < lo:
        hi = lo
    j.axis_z = max(lo, min(j.axis_z, hi))
    body = armor_loft(x0, x1, width, height, 1.1, j.axis_z)
    ball = ball_mesh(j)
    stem = stem_mesh(j, 1.5, x0 + 0.6)
    house = housing_solid(j, pitch)
    positives = [body, ball, stem, house]
    if peg:
        positives.append(
            scythe(max(10.0, height * 0.85), 4.4, 1.9, 34.0, height * 0.55, (x0 + x1) * 0.55, curl=0.12)
        )
    voids = transform_voids(socket_voids(j), socket_matrix((pitch, 0, j.axis_z), (1, 0, 0)))
    voids.append(face_cutter(j, pitch))
    voids.append(collar_void(j, pitch))
    functional = cut(unite(positives, f"končetina {index}"), voids, f"končetina {index}")
    support = print_support(j, x0 + 1.0, pitch)
    printable = unite([functional, support], f"tisk nohy {index}")
    return printable, functional, j


def make_foot(joint: Joint):
    """Chodidlo se třemi drápy. V použití míří drápy dopředu, tisk je natočený drápy nahoru."""
    j = Joint(
        ball_r=joint.ball_r,
        clear=joint.clear,
        inter=joint.inter,
        lip_deg=joint.lip_deg,
        stem_r=joint.stem_r,
        bowl_wall=joint.bowl_wall,
        rim_t=joint.rim_t,
        axis_z=joint.cavity_r + 2.4,
        slit_w=joint.slit_w,
        slit_n=joint.slit_n,
        cut_deg=joint.cut_deg,
    )
    ball = ball_mesh(j)
    stem = stem_mesh(j, 1.4, 6.2)
    body = armor_loft(5.6, 16.5, 13.5, 12.5, 0.8, j.axis_z)
    claws = []
    for ang, length in ((-18, 16.5), (0, 19.0), (18, 16.5)):
        claws.append(claw(length, math.radians(ang), math.radians(-8)))
    dew = claw(11.0, math.radians(180), math.radians(18), base_z=j.axis_z)
    dew.apply_translation((4.5, 0, 0))
    solid = unite([ball, stem, body, *claws, dew], "chodidlo")
    solid = cut(solid, [box(8, 0, -12, 40, 40, 24)], "chodidlo podložka")
    return solid, j


def claw(length, yaw, pitch, base_z=6.0) -> trimesh.Trimesh:
    n = 7
    rings = []
    direction = np.array(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ]
    )
    start = np.array([12.0, 0.0, base_z])
    side = np.array([-direction[1], direction[0], 0.0])
    if np.linalg.norm(side) < 1e-6:
        side = np.array([0.0, 1.0, 0.0])
    side = side / np.linalg.norm(side)
    up = np.cross(direction, side)
    for i in range(n):
        t = i / (n - 1)
        r = 2.15 * (1 - t) + 0.55
        c = start + direction * (length * t)
        ring = []
        for k in range(10):
            a = 2 * math.pi * k / 10
            ring.append(c + side * math.cos(a) * r + up * math.sin(a) * r)
        rings.append(np.array(ring))
    return loft_solid(rings)


def make_fin(joint: Joint, keel_len=52.0):
    """Vějíř v jednom kuse. Listy vyrůstají z brka, na náhledu jsou červené."""
    ball = ball_mesh(joint)
    stem = stem_mesh(joint, 1.5, 7.2)
    keel = fin_keel(keel_len)
    black = unite([ball, stem, keel], "brko")
    blades = []
    for x, yaw, lean, length in (
        (14.0, -30.0, 34.0, 48.0),
        (22.0, -15.0, 32.0, 56.0),
        (32.0, 0.0, 30.0, 64.0),
        (24.0, 15.0, 32.0, 54.0),
        (16.0, 30.0, 34.0, 46.0),
    ):
        t = min(1.0, max(0.0, (x - 5.0) / keel_len))
        h = 18.0 * (1.0 - t) + 8.0 * t
        blades.append(scythe(length, 7.6, 2.6, lean, max(6.0, h * 0.72), x, 0.0, 0.08, yaw))
    solid = unite([black, *blades], "vějíř")
    printable = unite([solid, print_support(joint, 8.0, 12.0)], "tisk vějíře")
    return printable, black, blades


def fin_keel(length) -> trimesh.Trimesh:
    """Svislé brko. Spodek leží na podložce, ať vějíř nemá převis."""
    rings = []
    n = 7
    for i in range(n):
        t = i / (n - 1)
        x = 5.0 + length * t
        h = 18.0 * (1 - t) + 8.0 * t
        th = 3.6 * (1 - t) + 1.8
        rings.append(
            np.array(
                [
                    [x, -th / 2, 0.0],
                    [x, th / 2, 0.0],
                    [x, th / 2, h],
                    [x, -th / 2, h],
                ]
            )
        )
    return loft_solid(rings)


def red_fin_leaf(w, h) -> trimesh.Trimesh:
    """List tisknutý patou dolů. Díra je o 0,16 mm menší než kolík."""
    base = box(0, 0, 3.4, w, 3.3, 6.8)
    hole = square_peg(0, 0, -0.2, 5.6, 2.00, 2.00)
    spine = blade(h, w * 0.92, 2.3, 34.0, 0.0, 5.2)
    return cut(unite([base, spine], "list"), [hole], "list díra")


def mounted_leaf(origin, direction, length) -> trimesh.Trimesh:
    """Nasadí list na kolík tak, aby čepel mířila dál od těla, ne zpátky do něj."""
    origin = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    leaf = red_fin_leaf(8.4, length)
    mounted = map_z_to_dir(leaf, d, origin)
    far = mounted.vertices[np.argmax(np.linalg.norm(mounted.vertices - origin, axis=1))]
    horiz = np.array([d[0], d[1], 0.0])
    tip = far - origin
    tip[2] = 0.0
    if float(np.dot(horiz, tip)) < 0.0:
        rolled = red_fin_leaf(8.4, length)
        rolled.apply_transform(rot_matrix([0, 0, 1], math.pi))
        mounted = map_z_to_dir(rolled, d, origin)
    return mounted


def eye_mesh() -> trimesh.Trimesh:
    ball = sphere(3.05, (0, 0, 3.4), subdiv=3)
    cutter = box(0, 0, -4, 16, 16, 8)
    ball = cut(ball, [cutter], "oko ořez")
    hole = square_peg(0, 0, -0.2, 4.6, 2.00, 2.00)
    return cut(ball, [hole], "oko")


def lay_on_bed(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_translation((-m.bounds[0, 0], -m.bounds[0, 1], -m.bounds[0, 2]))
    return m


def overhang_fraction(mesh: trimesh.Trimesh, limit=-0.78, zmin=0.8) -> tuple[float, int]:
    n = mesh.face_normals
    area = mesh.area_faces
    cents = mesh.triangles_center
    bad = (n[:, 2] < limit) & (cents[:, 2] > zmin)
    total = float(area.sum()) + 1e-9
    return float(area[bad].sum()) / total, int(bad.sum())


def pack_plates(items: list[tuple[str, trimesh.Trimesh, str]]):
    """Položí díly na desky 220 mm. Vrací seznam desek."""
    placed_items = []
    for name, mesh, color in items:
        m = lay_on_bed(mesh)
        ext = m.extents
        if ext[0] > USABLE or ext[1] > USABLE:
            raise RuntimeError(f"{name} se nevejde na podložku: {ext[:2]}")
        if ext[2] > 245:
            raise RuntimeError(f"{name} je vyšší než tiskárna: {ext[2]:.1f}")
        placed_items.append((name, m, color, ext[0], ext[1]))
    placed_items.sort(key=lambda p: p[4], reverse=True)
    plates = []
    cur = []
    x = y = row_h = 0.0
    for name, mesh, color, w, h in placed_items:
        if x + w > USABLE:
            x = 0.0
            y += row_h + GAP
            row_h = 0.0
        if y + h > USABLE:
            plates.append(cur)
            cur = []
            x = y = row_h = 0.0
        mesh.apply_translation((MARGIN + x, MARGIN + y, 0))
        cur.append((name, mesh, color))
        x += w + GAP
        row_h = max(row_h, h)
    if cur:
        plates.append(cur)
    return plates


def write_3mf(objects: list[tuple[str, trimesh.Trimesh, str]], path: Path, title: str, description: str):
    """3MF s více tělesy. Barva je jen označení, každá deska se tiskne jedním filamentem."""
    colors = {BLACK: "Cerna", RED: "Cervena", EYE: "Oko"}
    # map color hex to index
    palette = []
    for _, _, color in objects:
        if color not in palette:
            palette.append(color)
    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        '<model unit="millimeter" xml:lang="cs-CZ" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">'
    )
    parts.append(f"  <metadata name=\"Title\">{title}</metadata>")
    parts.append(f"  <metadata name=\"Description\">{description}</metadata>")
    parts.append('  <metadata name="Application">generate_drak.py</metadata>')
    parts.append('  <metadata name="Printer">Creality Ender 3 V3 SE, tryska 0,4 mm</metadata>')
    parts.append("  <resources>")
    parts.append('    <m:basematerials id="1">')
    for color in palette:
        parts.append(
            f'      <m:base name="{colors.get(color, "Barva")}" displaycolor="{color}FF"/>'
        )
    parts.append("    </m:basematerials>")
    for i, (name, mesh, color) in enumerate(objects, start=2):
        pid = palette.index(color)
        v = np.asarray(mesh.vertices, dtype=float)
        f = np.asarray(mesh.faces, dtype=int)
        parts.append(f'    <object id="{i}" type="model" name="{name}">')
        parts.append("      <mesh><vertices>")
        for x, y, z in v:
            parts.append(f'        <vertex x="{x:.3f}" y="{y:.3f}" z="{z:.3f}"/>')
        parts.append("      </vertices><triangles>")
        for a, b, c in f:
            parts.append(
                f'        <triangle v1="{a}" v2="{b}" v3="{c}" pid="1" p1="{pid}"/>'
            )
        parts.append("      </triangles></mesh>")
        parts.append("    </object>")
    parts.append("  </resources><build>")
    for i in range(2, len(objects) + 2):
        parts.append(f'    <item objectid="{i}"/>')
    parts.append("  </build></model>")
    model = "\n".join(parts).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("3D/3dmodel.model", model)
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


def chain_length(head_len, pitches, fin_len) -> float:
    return head_len + float(np.sum(pitches)) + fin_len


def measure_tip(mesh: trimesh.Trimesh, axis=0, end="min") -> float:
    return float(mesh.bounds[0, axis] if end == "min" else mesh.bounds[1, axis])


def place_limb(mesh, heading, socket, drop_deg, yaw_deg):
    """Kouli nohy posadí do objímky. +X nohy míří ven, pak se sklopí k zemi."""
    m = mesh.copy()
    # lokální koule je ve výšce osy nohy, ne v nule
    # srovnáme ji do počátku, natočíme a posuneme do objímky
    axis_z = float(np.median(m.vertices[:, 2]))
    # spolehlivěji: koule je u minimálního X
    # použijeme známou osu z geometrie končetiny — je blízko středu Z u kořene
    root = m.vertices[np.argmin(m.vertices[:, 0])]
    # radši pevně: koule má střed v x=0
    ball_z = LIMB.axis_z if abs(root[2] - LIMB.axis_z) > 8 else root[2]
    # najdi vrchol nejblíž (0, 0, nějaké z) mezi těmi s malým x
    near = m.vertices[m.vertices[:, 0] < 1.2]
    if len(near):
        ball_z = float(np.median(near[:, 2]))
    m.apply_translation((0, 0, -ball_z))
    # +X -> heading, potom sklopit kolem osy kolmé
    h = np.asarray(heading, float)
    h = h / np.linalg.norm(h)
    x = np.array([1.0, 0.0, 0.0])
    c = float(np.clip(np.dot(x, h), -1, 1))
    if c < 0.999:
        axis = np.cross(x, h)
        if np.linalg.norm(axis) < 1e-8:
            axis = np.array([0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        m.apply_transform(rot_matrix(axis, math.acos(c)))
    # sklopení k zemi: rotace kolem osy světového X×heading, tedy kolem vodorovné kolmice
    side = np.cross(h, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(side) < 1e-8:
        side = np.array([1.0, 0.0, 0.0])
    side = side / np.linalg.norm(side)
    m.apply_transform(rot_matrix(side, math.radians(drop_deg)))
    m.apply_transform(rot_matrix(np.array([0, 0, 1.0]), math.radians(yaw_deg * 0.15)))
    m.apply_translation(socket)
    return m


def limb_end(heading, socket, pitch, drop_deg, yaw_deg):
    if pitch <= 0:
        return heading, socket
    h = np.asarray(heading, float)
    h = h / np.linalg.norm(h)
    side = np.cross(h, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(side) < 1e-8:
        side = np.array([1.0, 0.0, 0.0])
    side = side / np.linalg.norm(side)
    # směr po sklopení
    d = h * math.cos(math.radians(drop_deg)) + np.array([0.0, 0.0, -1.0]) * math.sin(math.radians(abs(drop_deg)))
    # drop je záporný (dolů). rotace kolem `side` o záporný úhel sklopí +heading k -Z
    ang = math.radians(drop_deg)
    # Rodrigues
    k = side
    d = h * math.cos(ang) + np.cross(k, h) * math.sin(ang) + k * np.dot(k, h) * (1 - math.cos(ang))
    d = d / np.linalg.norm(d)
    return d, socket + d * pitch


def pose_yaw_pitch(yaw_deg, pitch_deg, origin):
    y = rot_matrix([0, 0, 1], math.radians(yaw_deg), origin)
    p = rot_matrix([0, 1, 0], math.radians(pitch_deg), origin)
    return y @ p


FOV_Y = math.radians(28.0)


def camera_fit(pts, eye_dir, size, fill=0.84):
    """Odstup kamery tak, aby se celý drak vešel do záběru i s okrajem."""
    w, h = size
    pts = np.asarray(pts, float)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    eye_dir = np.asarray(eye_dir, float)
    eye_dir = eye_dir / np.linalg.norm(eye_dir)
    forward = -eye_dir
    up_guess = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_guess)
    if np.linalg.norm(right) < 1e-6:
        up_guess = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up_guess)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rel = pts - center
    xs = rel @ right
    ys = rel @ true_up
    focal = (h * 0.5) / math.tan(FOV_Y * 0.5)
    dist = max(
        float(np.max(np.abs(xs))) * focal / (w * 0.5 * fill),
        float(np.max(np.abs(ys))) * focal / (h * 0.5 * fill),
        80.0,
    )
    return center + eye_dir * dist * 1.08, center


def render_scene(entries, path: Path, eye, target, up=(0, 0, 1), size=(1400, 900)):
    """Jednoduchý z-buffer. entries = [(mesh, rgb), ...]."""
    w, h = size
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    fov = FOV_Y
    focal = (h * 0.5) / math.tan(fov * 0.5)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    img = np.ones((h, w, 3), dtype=np.float32)
    light = (forward * 0.25 + true_up * 0.82 + right * 0.45)
    light = light / np.linalg.norm(light)
    fill = (-right * 0.5 + true_up * 0.2 + forward * 0.2)
    fill = fill / np.linalg.norm(fill)

    def project(pts):
        rel = pts - eye
        x = rel @ right
        y = rel @ true_up
        z = rel @ forward
        sx = focal * x / np.maximum(z, 1e-3) + w * 0.5
        sy = h * 0.5 - focal * y / np.maximum(z, 1e-3)
        return np.column_stack([sx, sy, z])

    for mesh, rgb in entries:
        rgb = np.asarray(rgb, dtype=np.float32)
        verts = project(np.asarray(mesh.vertices))
        normals = mesh.face_normals
        faces = mesh.faces
        shade = np.clip(normals @ light, 0, 1) * 0.75 + np.clip(normals @ fill, 0, 1) * 0.25
        shade = 0.28 + 0.72 * shade
        cols = np.clip(rgb * shade[:, None], 0, 1)
        for fi, (i0, i1, i2) in enumerate(faces):
            tri = verts[[i0, i1, i2]]
            if np.any(tri[:, 2] < 1.0):
                continue
            # backface
            area2 = (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (tri[1, 1] - tri[0, 1]) * (
                tri[2, 0] - tri[0, 0]
            )
            if area2 >= 0:
                continue
            minx = max(int(tri[:, 0].min()), 0)
            maxx = min(int(tri[:, 0].max()) + 1, w - 1)
            miny = max(int(tri[:, 1].min()), 0)
            maxy = min(int(tri[:, 1].max()) + 1, h - 1)
            if minx >= maxx or miny >= maxy:
                continue
            if (maxx - minx) * (maxy - miny) > 8000:
                continue
            xs = np.arange(minx, maxx)
            ys = np.arange(miny, maxy)
            gx, gy = np.meshgrid(xs, ys)
            den = area2
            w0 = ((tri[1, 0] - gx) * (tri[2, 1] - gy) - (tri[2, 0] - gx) * (tri[1, 1] - gy)) / den
            w1 = ((tri[2, 0] - gx) * (tri[0, 1] - gy) - (tri[0, 0] - gx) * (tri[2, 1] - gy)) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not np.any(inside):
                continue
            z = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
            view = zbuf[miny:maxy, minx:maxx]
            update = inside & (z < view)
            if not np.any(update):
                continue
            view[update] = z[update]
            img[miny:maxy, minx:maxx][update] = cols[fi]
    # jemný stín pod modelem: ztmavit pixely těsně u nejnižšího pásu necháme bílé pozadí
    out = Image.fromarray((img * 255).astype(np.uint8), "RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def section_plot(mesh: trimesh.Trimesh, origin, normal, path: Path, title: str):
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    fig, ax = plt.subplots(figsize=(8, 5))
    if sec is not None:
        sec2d, _ = sec.to_2D()
        for entity in sec2d.entities:
            pts = sec2d.vertices[entity.points]
            ax.plot(pts[:, 0], pts[:, 1], color="#9b1b24", lw=1.0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, lw=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def joint_self_test(joint: Joint) -> dict:
    """Dva články proti sobě. Ověří vůli, ohnutí a to, že koule neprojde ven."""
    pitch = 18.0
    print_a, func_a, _ = make_segment(pitch, 28, 17.5, 3.2, 1, (False, False), joint, spine_len=0.0)
    _, func_b, _ = make_segment(pitch, 28, 17.5, 3.2, 2, (False, False), joint, spine_len=0.0)
    seat = np.eye(4)
    seat[:3, 3] = (pitch, 0, 0)
    seated = apply_copy(func_b, seat)
    gap_vol = overlap_volume(func_a, seated)
    # tah ven: posun o 2 mm ve směru ven z objímky (+X)
    pulled = apply_copy(func_b, seat.copy())
    pulled.apply_translation((joint.throat_x + 0.7, 0, 0))
    pull_vol = overlap_volume(func_a, pulled)
    lifted = apply_copy(seated, np.eye(4))
    lifted.apply_translation((0, 0, 1.4))
    lift_vol = overlap_volume(func_a, lifted)
    # ohnutí
    angles = {}
    origin = np.array([pitch, 0.0, joint.axis_z])
    for name, axis in ("yaw", [0, 0, 1]), ("pitch", [0, 1, 0]):
        lo, hi = 0.0, 48.0
        for _ in range(10):
            mid = (lo + hi) / 2
            R = rot_matrix(axis, math.radians(mid), origin)
            turned = apply_copy(seated, R)
            if overlap_volume(func_a, turned) < 1.5:
                lo = mid
            else:
                hi = mid
        angles[name] = lo
    frac, nbad = overhang_fraction(print_a)
    info = {
        "gap_mm3": gap_vol,
        "pull_mm3": pull_vol,
        "lift_mm3": lift_vol,
        "yaw": angles["yaw"],
        "pitch": angles["pitch"],
        "overhang": frac,
        "overhang_faces": nbad,
        "watertight": bool(print_a.is_watertight and func_a.is_watertight),
        "lip_d": joint.lip_r * 2,
        "ball_d": joint.ball_r * 2,
    }
    print(
        f"Kloub: vůle-kolize {gap_vol:.2f} mm3, tah {pull_vol:.1f} mm3, zdvih {lift_vol:.1f} mm3, "
        f"ohnutí yaw {angles['yaw']:.1f}° pitch {angles['pitch']:.1f}°, "
        f"převis {frac*100:.2f} %, watertight {info['watertight']}"
    )
    if gap_vol > 2.0:
        raise RuntimeError("Kloub se v klidu protíná")
    if pull_vol < 8.0:
        raise RuntimeError("Objímka kuli nedrží při tahu")
    if lift_vol < 8.0:
        raise RuntimeError("Koule může z objímky vyjet nahoru")
    if angles["yaw"] < 20 or angles["pitch"] < 16:
        raise RuntimeError(f"Kloub se málo ohýbá: {angles}")
    if frac > 0.012:
        raise RuntimeError(f"Příliš mnoho převisů: {frac:.3f}")
    if not info["watertight"]:
        raise RuntimeError("Článek není uzavřený")
    return info


def build_all():
    print("Zkouším kloub…")
    stats = joint_self_test(SPINE)
    print("Modeluji hlavu…")
    head, head_dress = head_mesh(SPINE)
    head_len = -measure_tip(head, 0, "min")
    print(f"  hlava {head_len:.1f} mm, rozsah {head.extents.round(1)}")

    print("Modeluji vějíř…")
    fin_print, fin_func, fin_leaves = make_fin(SPINE)
    fin_len = float(fin_func.bounds[1, 0])
    for leaf in fin_leaves:
        fin_len = max(fin_len, float(leaf.bounds[1, 0]))
    frac, _ = overhang_fraction(fin_print)
    if frac > 0.02:
        raise RuntimeError(f"vějíř má převis {frac:.3f}")
    if not fin_print.is_watertight:
        raise RuntimeError("vějíř není uzavřený")
    print(f"  vějíř {fin_len:.1f} mm, převis {frac:.2%}")

    remaining = TARGET_LEN - head_len - fin_len
    # 5 krk, 9 tělo, 14 ocas. Rozteč musí zůstat dost dlouhá na krček.
    weights = np.array([1.00] * 5 + [1.12] * 9 + list(np.linspace(1.06, 0.92, 14)))
    pitches = weights / weights.sum() * remaining
    if pitches.min() < 15.0:
        raise RuntimeError(f"rozteč vyšla krátká: {pitches.min():.2f}")
    print(
        f"Články {len(pitches)} ks, rozteč {pitches.min():.1f}–{pitches.max():.1f} mm, "
        f"délka řetězu {chain_length(head_len, pitches, fin_len):.1f} mm"
    )

    # ramena na 2. článku těla (index 7), kyčle na 9. (index 14) — počítáno od 0
    shoulder_i = 6
    hip_i = 12
    segments = []
    printables = []
    print("Modeluji články…")
    for i, pitch in enumerate(pitches):
        t = i / (len(pitches) - 1)
        width = 36.0 * (1 - t) + 18.0 * t
        height = 21.0 * (1 - t) + 15.0 * t
        keel = 4.2 * (1 - t) + 1.6 * t
        if i < 6:
            width *= 0.84
            height *= 0.90
        sides = (i == shoulder_i or i == hip_i, i == shoulder_i or i == hip_i)
        spine_len = (height + keel) * (1.35 if i >= 4 else 1.05)
        printable, functional, reds = make_segment(
            float(pitch), width, height, keel, i + 1, sides, SPINE, spine_len=spine_len
        )
        if not printable.is_watertight:
            raise RuntimeError(f"článek {i+1} není uzavřený")
        frac, _ = overhang_fraction(printable)
        if frac > 0.02:
            raise RuntimeError(f"článek {i+1} má převis {frac:.3f}")
        segments.append(
            {
                "mesh": functional,
                "reds": reds,
                "pitch": float(pitch),
                "width": width,
                "x_body0": 6.9,
                "crown": height + keel,
                "hip_x": (6.9 + (pitch - 6.0)) * 0.48,
                "sides": sides,
                "index": i,
            }
        )
        printables.append((f"clanek-{i+1:02d}", printable))
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(pitches)}")

    print("Modeluji nohy…")
    legs = []
    leg_crests = []  # trn je už součást nohy
    # přední a zadní, levá a pravá — stejný tvar, jen zrcadlený
    for place in ("predni", "zadni"):
        for side, mirror in (("L", False), ("P", True)):
            chain = []
            for part, pitch, w, h in (("stehno", 30.0, 16.5, 15.2), ("holen", 26.0, 14.2, 14.0)):
                pr, fn, _ = limb_segment(pitch, w, h, f"{place}-{side}-{part}", LIMB, peg=True)
                if mirror:
                    fn = mirror_y(fn)
                    pr = mirror_y(pr)
                frac, _ = overhang_fraction(pr)
                if frac > 0.02:
                    raise RuntimeError(f"noha {place}-{side}-{part} má převis {frac:.3f}")
                legs.append((f"noha-{place}-{side}-{part}", pr, fn, pitch))
                chain.append((fn, pitch))
            foot, _ = make_foot(LIMB)
            if mirror:
                foot = mirror_y(foot)
            foot_print = orient_toes_up(foot)
            legs.append((f"noha-{place}-{side}-tlapka", foot_print, foot, 0.0))
            chain.append((foot, 0.0))

    # sestava pro náhled
    print("Skládám náhled…")
    straight, posed = assemble_preview(
        head, segments, fin_func, fin_leaves, legs, pitches, head_dress
    )
    length = chain_length(head_len, pitches, fin_len)
    return {
        "stats": stats,
        "head": head,
        "segments": segments,
        "printables": printables,
        "legs": legs,
        "fin_leaves": fin_leaves,
        "fin_func": fin_func,
        "head_dress": head_dress,
        "fin_print": fin_print,
        "straight": straight,
        "posed": posed,
        "length": length,
        "n_seg": len(pitches),
        "pitches": pitches,
        "head_len": head_len,
        "fin_len": fin_len,
    }


def small_crest(peg) -> trimesh.Trimesh:
    base = box(0, 0, 3.1, 8.5, 6.5, 6.2)
    hole = square_peg(0, 0, -0.2, 5.0, peg - 0.12, peg - 0.12)
    sp = blade(13.0, 4.2, 2.0, 32.0, 0.0, 5.6)
    return cut(unite([base, sp], "trn"), [hole], "trn díra")


def mirror_y(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.vertices[:, 1] *= -1
    m.faces = m.faces[:, ::-1]
    return m


def orient_toes_up(foot: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drápy míří v modelu zhruba +X. Pro tisk je postavíme nahoru a kouli dolů."""
    m = foot.copy()
    m.apply_transform(rot_matrix([0, 1, 0], -math.pi / 2))
    m.apply_translation((0, 0, -m.bounds[0, 2]))
    # malý límec kvůli stabilitě, ulomí se s patkou
    return m


def assemble_preview(head, segments, fin_func, fin_leaves, legs, pitches, head_dress):
    """Vrátí seznam (mesh, barva) pro rovnou i ohnutou pózu."""
    ball = np.array([0.0, 0.0, SPINE.axis_z])

    def chain(angles):
        items = []
        frames = []
        T = np.eye(4)
        items.append((apply_copy(head, T), BLACK))
        for piece, color in head_dress:
            items.append((apply_copy(piece, T), color))
        for seg, (yaw, pitch_deg) in zip(segments, angles):
            R = pose_yaw_pitch(yaw, pitch_deg, ball)
            T = T @ R
            frames.append(T.copy())
            items.append((apply_copy(seg["mesh"], T), BLACK))
            for red in seg["reds"]:
                items.append((apply_copy(red, T), RED))
            step = np.eye(4)
            step[0, 3] = seg["pitch"]
            T = T @ step
        items.append((apply_copy(fin_func, T), BLACK))
        for leaf in fin_leaves:
            items.append((apply_copy(leaf, T), RED))
        return items, frames

    n = len(segments)
    straight_angles = [(0.0, 0.0)] * n
    posed_angles = []
    for i in range(n):
        t = i / max(1, n - 1)
        # esíčko v půdorysu, ocas se zvedne až ke konci
        yaw = 12.0 * math.sin(2.0 * math.pi * t)
        pitch = -1.2 * math.sin(math.pi * t)
        if t > 0.78:
            u = (t - 0.78) / 0.22
            pitch -= 7.5 * u
        yaw = max(-16.0, min(16.0, yaw))
        pitch = max(-12.0, min(6.0, pitch))
        posed_angles.append((yaw, pitch))
    straight, frames_s = chain(straight_angles)
    posed, frames_p = chain(posed_angles)
    # nohy: stehno, holeň, tlapka v pořadí, jak se ukládaly
    # 4 nohy × (stehno, holeň, tlapka) = 12 položek v `legs`
    leg_groups = [legs[i : i + 3] for i in range(0, len(legs), 3)]
    anchors = []
    for seg in segments:
        if seg["sides"][0]:
            anchors.append(seg)
    # očekáváme rameno a kyčel, ke každé dvě nohy (L/P) — pořadí skupin je predni L, predni P, zadni L, zadni P
    if len(anchors) >= 2 and len(leg_groups) >= 4:
        pairs = [
            (anchors[0], leg_groups[0], 1.0),
            (anchors[0], leg_groups[1], -1.0),
            (anchors[1], leg_groups[2], 1.0),
            (anchors[1], leg_groups[3], -1.0),
        ]
        for target, seg in ((straight, frames_s), (posed, frames_p)):
            for anchor, group, sign in pairs:
                idx = anchor["index"]
                frame = seg if False else (frames_s if target is straight else frames_p)[idx]
                add_leg(target, frame, anchor, group, sign)
    return straight, posed


def add_leg(items, frame, anchor, group, sign):
    """Nasadí stehno, holeň a tlapku do boční objímky a mírně je sklopí."""
    limb_face = LIMB.face_x
    y_face = sign * (anchor["width"] * 0.48)
    y_c = y_face - sign * limb_face
    local_socket = np.array([anchor["hip_x"], y_c, SPINE.axis_z, 1.0])
    socket = (frame @ local_socket)[:3]
    outward = np.array([0.0, sign, 0.0])
    # stehno míří ven a dolů
    drops = (-35.0, -28.0, -12.0)
    yaw_spread = 18.0 * sign
    cursor = socket
    heading = outward
    for ( _name, _pr, fn, pitch), drop in zip(group, drops):
        placed = place_limb(fn, heading, cursor, drop, yaw_spread)
        items.append((placed, BLACK))
        # další kloub je na konci článku v jeho lokální ose +X
        # po place_limb míří +X článku ve směru heading, sklopeném o drop
        heading, cursor = limb_end(heading, cursor, pitch, drop, yaw_spread)


def write_settings(path: Path, info: dict, plate_names: list[str]):
    s = info["stats"]
    text = f"""Artikulovaný drak — 750 mm
Tisk na Creality Ender 3 V3 SE, tryska 0,4 mm, PLA

JAK JE DLOUHÝ
  V natažené poloze {info['length']:.0f} mm od špičky čenichu po konec ocasního vějíře.
  Článků páteře: {info['n_seg']}. Hlava, čtyři nohy a vějíř jsou navíc.
  Ohnutý na stole je kratší. Délka 75 cm je délka řetězu, ne krabice.

PROČ SE TO NEROZPADNE A PŘITOM TO CHODÍ
  Kloub je koule Ø {s['ball_d']:.1f} mm v objímce.
  Otvor objímky má Ø {s['lip_d']:.1f} mm, tedy o {s['ball_d']-s['lip_d']:.1f} mm méně než koule.
  Koule zapadne až za rovník. Do boku ani ohnutím nevypadne.
  Ven jde jen přímým tahem po ose, když se věnec objímky rozevře. Na to je potřeba síla.
  Uvnitř je vůle {CLEAR:.2f} mm, proto se článek točí.
  Změřené ohnutí jednoho kloubu: do stran {s['yaw']:.0f}°, nahoru a dolů {s['pitch']:.0f}°.
  Na {info['n_seg']} kloubech z toho vznikne plynulé esíčko i zvednutý ocas.
  Dřík koule má Ø {STEM_R*2:.1f} mm, věnec objímky je tlustý {BOWL_WALL:.1f} mm.
  Čtyři zářezy ve věnci jsou schválně: při nasazení se objímka rozevře a pak zase sevře.
  Na konci každého zářezu je kulatá dírka, ať věnec nepraskne.

JAK TO VYPADÁ
  Jeden drak: černé tělo, červený hřbet, rohy, oči a ocasní vějíř.
  Trny, rohy, oči i listy vějíře jsou s tělem srostlé. Nic se na to nelepí
  a nejsou na to kolíky.
  Na náhledu je hřbet červený. Tiskárna má jednu trysku, takže z ní vyleze
  jedna barva. Červené plochy pak natři, nebo ve sliceru vyměň filament
  ve výšce, kde končí pancíř a začínají trny.
  Na boku článku je číslo. 01 je u hlavy.

PROČ TO NENÍ JEDEN TISK
  Drak měří 75 cm, podložka Enderu 22×22 cm. Najednou se to nevejde.
  Každý soubor je jeden už spojený kus páteře: články jsou v sobě natištěné
  a po ulomení patek se hýbou. Kusy do sebe zacvakneš stejnou koulí.
  Nohy jsou zvlášť, zacvaknou se do boku.

SOUBORY
{chr(10).join('  ' + n for n in plate_names)}

JAK TO POLOŽIT
  Díly už leží tak, jak se mají tisknout. V sliceru je neotáčej a nerozděluj.
  V jednom souboru je víc těles schválně: jsou to články už nasazené v sobě.
  Podpěry: vypnout.
  Pod koulí, která je uvnitř objímky, je tenká patka. Ta patka je jen na tisk.
  Po vytištění každý kloub párkrát ohněte, patka se ulomí a článek se rozhýbe.
  První koule každého kusu (ta, co ještě v ničem nesedí) má širší patku.
  Tu ulom dřív, než kus zacvakneš do předchozího.
  Otřep na plochém spodku koule nevadí, ta plocha do stěny objímky nedosáhne.

MATERIÁL
  PLA, jedna barva na celý kus. Náhled ukazuje, co natřít na červeno.

  Tryska          205 °C (první vrstva 210 °C)
  Podložka        60 °C
  Výška vrstvy    0,20 mm
  Šířka čáry      0,42 mm
  Stěny           6
                  věnec objímky má {BOWL_WALL:.1f} mm, při šesti stěnách je plný
  Plné vrstvy     5 dole a 5 nahoře
  Výplň           20 % gyroid
                  hlava snese 15 %, klouby rozhodují stěny, ne výplň
  Rychlost        vnější stěny 40 mm/s
                  vnitřní stěny 50 mm/s
                  na číslech a zářezech nezrychluj
  Chlazení        100 % od 4. vrstvy
  Podpěry         vypnout
  Brim            4 mm, ať se dlouhý kus na podložce nezkroutí

SKLÁDÁNÍ
  1. V každém kusu ohněte klouby, ať odpadne tenká patka uvnitř objímky.
     Širší patku na volné kouli ulomte kleštěmi.
  2. Volnou kouli dalšího kusu zatlačte do objímky předchozího.
     Tlačte rovně v ose. Má to jít ztuha. Když věnec bělá, přestaňte
     a zkuste to znovu rovně.
  3. Stehna nasuňte do ramenního a kyčelního článku (mají otvor na boku).
     Holeň do stehna, tlapku do holeně. Drápy míří dopředu.
  4. Čísla na bocích jdou od 01 u hlavy k ocasu.

KDYŽ TO NEJDE NASADIT
  Díra po tisku bývá o chlup menší. Stačí pár tahů pilníkem na kouli,
  neubírejte věnec objímky. Když uberete věnec, článek začne vypadávat.

POKLÁDÁNÍ
  Drž trup a ohni ho do strany nebo nahoru. Neškubej jeden článek od druhého.
  Nohy ohnutím nastav, jak mají stát.
"""
    path.write_text(text, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    info = build_all()
    print(f"Délka {info['length']:.1f} mm")
    if abs(info["length"] - TARGET_LEN) > 2.0:
        raise RuntimeError(f"délka {info['length']:.1f} není 750")

    for stale in (
        "drak-telo-1.3mf",
        "drak-hlava-nohy-1.3mf",
        "drak-cervena-1.3mf",
    ):
        old = OUT_DIR / stale
        if old.exists():
            old.unlink()

    print("Skládám spojené kusy na podložku…")
    chain = [
        {
            "name": "hlava",
            "black": info["head"],
            "reds": [m for m, _ in info["head_dress"]],
            "has_ball": False,
            "pitch": 0.0,
            "x_body0": 6.9,
        }
    ]
    for seg in info["segments"]:
        chain.append(
            {
                "name": f"clanek-{seg['index']+1:02d}",
                "black": seg["mesh"],
                "reds": seg["reds"],
                "has_ball": True,
                "pitch": seg["pitch"],
                "x_body0": seg["x_body0"],
            }
        )
    chain.append(
        {
            "name": "vejir",
            "black": info["fin_func"],
            "reds": info["fin_leaves"],
            "has_ball": True,
            "pitch": 0.0,
            "x_body0": 8.0,
        }
    )
    limit = USABLE - 4.0
    groups: list[list[dict]] = []
    cur: list[dict] = []
    origin = 0.0
    base = 0.0
    for item in chain:
        meshes = [item["black"], *item["reds"]]
        local_min = min(float(m.bounds[0, 0]) for m in meshes)
        local_max = max(float(m.bounds[1, 0]) for m in meshes)
        world_min = origin + local_min
        world_max = origin + local_max
        if cur and world_max - base > limit:
            groups.append(cur)
            cur = []
            base = world_min
        if not cur:
            base = world_min
        placed = dict(item)
        placed["origin"] = origin
        placed["narrow"] = bool(cur) and item["has_ball"]
        cur.append(placed)
        origin += item["pitch"]
    if cur:
        groups.append(cur)

    plate_files = []
    desc = (
        "Artikulovany drak 750 mm, jeden spojený kus páteře. "
        "Tisknout tak jak leží, bez podpor, části nerozdělovat. "
        "Patku pod koulí po tisku ulomit. PLA, tryska 0,4 mm, vrstva 0,20 mm, 6 stěn."
    )
    for gi, group in enumerate(groups, start=1):
        objects = []
        built = []
        for item in group:
            if item["has_ball"]:
                is_last = item is group[-1] and item["name"] != "vejir"
                support = segment_support(
                    SPINE,
                    item["x_body0"],
                    max(item["pitch"], 16.0),
                    item["narrow"],
                    rear=is_last,
                )
                parts = [item["black"].copy(), support, *[r.copy() for r in item["reds"]]]
            else:
                parts = [item["black"].copy(), *[r.copy() for r in item["reds"]]]
            mesh = unite(parts, item["name"])
            mesh.apply_translation((item["origin"], 0.0, 0.0))
            built.append(mesh)
            objects.append((item["name"], mesh, BLACK))
        for a, b, na, nb in zip(built, built[1:], group, group[1:]):
            vol = overlap_volume(a, b)
            if vol > 2.0:
                raise RuntimeError(f"{na['name']} a {nb['name']} se prolínají o {vol:.1f} mm3")
        mins = np.min([m.bounds[0] for m in built], axis=0)
        for mesh in built:
            mesh.apply_translation((-mins[0] + MARGIN, -mins[1] + MARGIN, -mins[2]))
        ext = np.max([m.bounds[1] for m in built], axis=0) - np.min([m.bounds[0] for m in built], axis=0)
        if ext[0] > USABLE + 0.5 or ext[1] > USABLE + 0.5:
            raise RuntimeError(f"kus {gi} se nevejde na podložku: {ext[:2].round(1)}")
        if ext[2] > 245:
            raise RuntimeError(f"kus {gi} je vyšší než tiskárna: {ext[2]:.0f} mm")
        name = f"drak-kus-{gi}.3mf"
        write_3mf(objects, OUT_DIR / name, f"Drak kus {gi}", desc)
        plate_files.append(f"modely/{name}  (spojený kus, {len(objects)} článků)")
        print(f"  {name}: {len(objects)} článků, {ext[0]:.0f}×{ext[1]:.0f} mm")

    leg_items = [(n, pr, BLACK) for n, pr, _fn, _pitch in info["legs"]]
    leg_plates = pack_plates(leg_items)
    for i, plate in enumerate(leg_plates, start=1):
        name = f"drak-nohy-{i}.3mf"
        write_3mf(plate, OUT_DIR / name, f"Drak nohy {i}", desc)
        plate_files.append(f"modely/{name}  (nohy, {len(plate)} dílů)")
        print(f"  {name}: {len(plate)} dílů")

    write_settings(ROOT / "NASTAVENI_DRAK.txt", info, plate_files)
    print("Náhledy…")
    # jen póza, ať render stihne rozumný počet trojúhelníků
    def tint(entries):
        out = []
        for mesh, color in entries:
            if color == RED or color == EYE:
                rgb = (0.82, 0.09, 0.11)
            else:
                rgb = (0.20, 0.20, 0.21)
            out.append((mesh, rgb))
        return out

    size = (1200, 680)
    for name, entries, eye_dir in (
        ("drak-zboku.png", info["straight"], np.array([0.0, -1.0, 0.16])),
        ("drak-perspektiva.png", info["posed"], np.array([-0.28, -0.90, 0.32])),
    ):
        pts = np.vstack([m.vertices for m, _ in entries])
        eye, center = camera_fit(pts, eye_dir, size)
        render_scene(
            tint(entries),
            PREVIEW_DIR / name,
            eye,
            center,
            size=size,
        )
        print(f"  {name}")
    print("Hotovo.")


if __name__ == "__main__":
    main()
