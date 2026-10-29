"""2-D axisymmetric (r, z) mesh of the BABY-1L vessel: ClLiF melt (group 3) in Inconel 625 (group 1).

    python baby_mesh.py [--gui]      -> baby_2d.msh

The boundary groups (11-33, 99) are the ID_* constants of baby_festim.py.
"""

import sys

import gmsh

import experimental_data

CM = 1e-2


def generate_mesh(fname="baby_2d.msh", show_gui=False, mesh_size=0.001):
    gmsh.initialize()
    gmsh.model.add("BABY_2D")
    occ = gmsh.model.occ

    # radii [m]
    r_axis = 0.0
    r_heater = experimental_data.R_HEATER
    r_cllif = experimental_data.R_CLLIF
    r_inconel = experimental_data.R_INCONEL

    # heights [m], from the bottom of the base plate up; the base, alumina and He layers below the
    # vessel are not meshed, they only set the z origin
    y_inconel_bottom = (0.786 + 0.635 + 0.600) * CM
    y_cllif_bottom = y_inconel_bottom + experimental_data.T_BOTTOM
    y_cllif_top = y_cllif_bottom + experimental_data.T_CLLIF
    y_gap_top = y_cllif_top + experimental_data.T_GAP
    y_cap_top = y_gap_top + experimental_data.T_CAP
    y_heater_bottom = y_cllif_bottom + experimental_data.T_HEATER_GAP

    # the vessel bottom, wall and cap, and the melt below and around the heater well
    bottom = occ.addRectangle(
        r_axis, y_inconel_bottom, 0, r_inconel, experimental_data.T_BOTTOM
    )
    melt_lower = occ.addRectangle(
        r_axis, y_cllif_bottom, 0, r_cllif, y_heater_bottom - y_cllif_bottom
    )
    melt_upper = occ.addRectangle(
        r_heater, y_heater_bottom, 0, r_cllif - r_heater, y_cllif_top - y_heater_bottom
    )
    wall = occ.addRectangle(
        r_cllif, y_cllif_bottom, 0, r_inconel - r_cllif, y_gap_top - y_cllif_bottom
    )
    cap = occ.addRectangle(
        r_heater, y_gap_top, 0, r_inconel - r_heater, experimental_data.T_CAP
    )
    occ.synchronize()

    # The melt is one domain; it is drawn as two rectangles only because of the heater-well notch.
    # Fusing them changes the curve tags, and the boundary tags below are the ones after the fuse.
    melt, _ = occ.fuse([(2, melt_lower)], [(2, melt_upper)])
    occ.synchronize()
    occ.fragment([(2, bottom), (2, wall), (2, cap)] + melt, [])
    occ.synchronize()

    # sort the surfaces into melt and Inconel by their centre of mass
    tol = 1e-8

    def inside(v, lo, hi):
        return lo - tol <= v <= hi + tol

    def is_melt(r, z):
        return (
            inside(z, y_cllif_bottom, y_heater_bottom) and inside(r, r_axis, r_cllif)
        ) or (inside(z, y_heater_bottom, y_cllif_top) and inside(r, r_heater, r_cllif))

    def is_inconel(r, z):
        return (
            (
                inside(z, y_inconel_bottom, y_cllif_bottom)
                and inside(r, r_axis, r_inconel)
            )
            or (inside(r, r_cllif, r_inconel) and inside(z, y_cllif_bottom, y_gap_top))
            or (inside(z, y_gap_top, y_cap_top) and inside(r, r_heater, r_inconel))
        )

    inconel, cllif = [], []
    for dim, tag in gmsh.model.getEntities(2):
        r, z, _ = occ.getCenterOfMass(dim, tag)
        if is_melt(r, z):
            cllif.append(tag)
        elif is_inconel(r, z):
            inconel.append(tag)

    gmsh.model.addPhysicalGroup(2, inconel, 1)
    gmsh.model.setPhysicalName(2, 1, "inconel625")
    gmsh.model.addPhysicalGroup(2, cllif, 3)
    gmsh.model.setPhysicalName(2, 3, "cllif_natural")
    gmsh.model.setColor([(2, t) for t in inconel], 100, 149, 237)
    gmsh.model.setColor([(2, t) for t in cllif], 255, 180, 50)

    # boundary groups: name, curve tags, group id
    for name, curves, gid in (
        ("inconel_outer_bottom", [1], 31),
        ("inconel_outer_side", [2, 6], 32),
        ("inconel_outer_top", [11, 12], 33),
        ("left_symmetry_liquid", [17], 21),
        ("left_symmetry_inconel", [5], 22),
        ("top_cap_bc", [10], 11),
        ("gap_sidewall_bc", [8], 12),
        ("liquid_surface_bc", [14], 13),
        ("heater_cap_bc", [13], 14),
        ("liquid_heater_interface_bc", [16, 15], 16),
    ):
        gmsh.model.addPhysicalGroup(1, curves, tag=gid)
        gmsh.model.setPhysicalName(1, gid, name)

    def boundary(tags):
        return set(
            gmsh.model.getBoundary(
                [(2, t) for t in tags], oriented=False, recursive=False
            )
        )

    interface = [c[1] for c in boundary(cllif) & boundary(inconel)]
    gmsh.model.addPhysicalGroup(1, interface, tag=99)
    gmsh.model.setPhysicalName(1, 99, "liquid_inconel_interface")
    occ.synchronize()

    # element size: mesh_size, five times finer near the melt|Inconel interface and the free surface
    field = gmsh.model.mesh.field
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    refine = []
    for curves in (interface, [14]):
        dist = field.add("Distance")
        field.setNumbers(dist, "CurvesList", curves)
        thr = field.add("Threshold")
        field.setNumber(thr, "InField", dist)
        field.setNumber(thr, "SizeMin", mesh_size / 5)
        field.setNumber(thr, "SizeMax", mesh_size)
        field.setNumber(thr, "DistMin", 0.002)
        field.setNumber(thr, "DistMax", 0.006)
        refine.append(thr)
    finest = field.add("Min")
    field.setNumbers(finest, "FieldsList", refine)
    field.setAsBackgroundMesh(finest)

    gmsh.model.mesh.generate(2)
    gmsh.write(fname)
    if show_gui:
        gmsh.option.setNumber("Geometry.Points", 0)
        gmsh.option.setNumber("Mesh.Points", 0)
        gmsh.fltk.run()
    gmsh.finalize()


if __name__ == "__main__":
    generate_mesh(show_gui="--gui" in sys.argv)
