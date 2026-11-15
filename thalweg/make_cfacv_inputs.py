#!/usr/bin/env python3
"""Write cfacv's label PDB and CV file from a CV definition JSON.

cfacv defines every CV on "groups" of atoms, named in a label PDB that is
congruent with the NAMD structure atom for atom: the beta column holds each
atom's group number (0 = no group) and the occupancy column its mass.  The CV
file then lists each CV as a type and 0-based group indices.  Writing the two
by hand is where a wrong atom gets into a CV, so this derives both from the
same JSON the analysis tools read (see common.py), and prints what it chose.

By default only the CVs named under "string" are written, in that order,
which is the order of the columns of every restraint file and of the string.
--all writes every CV in the file (for a measurement run).

A group is a set of atoms.  Two CVs that use the same atoms share a group.
cfacv does not allow an atom in two different groups; that is an error here.

Example:
  make_cfacv_inputs.py --cv t4l_cvs.json --psf t4l_closed.psf --pdb t4l_closed.pdb \\
      --label label.pdb --cvinp cv.inp
"""
import argparse
import sys

from common import CVSet


def serial_field(n, in_group=False):
    """PDB serial, columns 7-11: decimal up to 99999, then hexadecimal as VMD writes it.

    cfacv's reader takes the field as DECIMAL whenever it looks like a whole
    number, so a hexadecimal serial with no letter in it (148000 -> "24220")
    is read as a different atom, silently.  For an atom that is in a group
    that would put a CV on the wrong atom, so it is refused here."""
    if n <= 99999:
        return f"{n:5d}"
    s = f"{n:5x}"
    if in_group and s.strip().isdigit():
        raise ValueError(f"atom serial {n} is written '{s.strip()}' in hexadecimal, which cfacv would read as decimal "
                         f"atom {int(s)}: this atom cannot be used in a CV group through a label PDB. "
                         "Reorder the system so the CV atoms come first (serial <= 99999).")
    return s


def build_groups(cvset, universe):
    """Returns (groups, cv_lines).  groups: list of sorted atom-index tuples,
    in order of first use.  cv_lines: (type, [group indices]) per CV."""
    groups, cv_lines, owner = [], [], {}
    for c in cvset.cvs:
        idx = []
        for sel in c["atoms"]:
            g = universe.select_atoms(sel)
            if g.n_atoms == 0:
                raise ValueError(f"CV {c['name']}: selection '{sel}' matches no atoms")
            key = tuple(sorted(int(i) for i in g.indices))
            if key not in groups:
                for a in key:
                    if a in owner:
                        raise ValueError(
                            f"CV {c['name']}: atom index {a} ('{sel}') is already in a different group; "
                            "cfacv does not allow overlapping groups")
                for a in key:
                    owner[a] = len(groups)
                groups.append(key)
            idx.append(groups.index(key))
        cv_lines.append((c["type"], idx))
    return groups, cv_lines


def write_label_pdb(path, universe, groups):
    group_of = {a: n + 1 for n, g in enumerate(groups) for a in g}
    with open(path, "w") as fh:
        fh.write("REMARK cfacv label PDB: occupancy = mass of group atoms, beta = group number\n")
        for a in universe.atoms:
            g = group_of.get(int(a.index), 0)
            occ = float(a.mass) if g else 0.0
            name = a.name if len(a.name) == 4 else f" {a.name:<3s}"
            x, y, z = a.position
            fh.write(f"ATOM  {serial_field(a.index + 1, bool(g))} {name:4s} {a.resname:<4s}{(a.segid or 'X')[:1]}"
                     f"{a.resid % 10000:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{g:6.2f}"
                     f"      {a.segid[:4]:<4s}\n")
        fh.write("END\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", required=True)
    ap.add_argument("--psf", required=True)
    ap.add_argument("--pdb", required=True, help="coordinates congruent with the PSF (PDB, or NAMD .coor)")
    ap.add_argument("--all", action="store_true", help="write every CV, not only the string CVs")
    ap.add_argument("--label", default="label.pdb")
    ap.add_argument("--cvinp", default="cv.inp")
    a = ap.parse_args(argv)

    import MDAnalysis as mda
    full = CVSet.load(a.cv)
    cvset = full if a.all else full.subset(full.string)
    fmt = "NAMDBIN" if a.pdb.endswith(".coor") else None
    u = mda.Universe(a.psf, a.pdb, format=fmt) if fmt else mda.Universe(a.psf, a.pdb)
    groups, cv_lines = build_groups(cvset, u)

    write_label_pdb(a.label, u, groups)
    with open(a.cvinp, "w") as fh:
        fh.write("# written by make_cfacv_inputs.py; group indices are 0-based\n")
        for (typ, idx), name in zip(cv_lines, cvset.names):
            fh.write(f"# {name}\n{typ} {' '.join(str(i) for i in idx)}\n")

    print(f"{len(groups)} groups, {len(cv_lines)} CVs")
    for n, g in enumerate(groups):
        ats = ", ".join(f"{u.atoms[i].resname}{u.atoms[i].resid}:{u.atoms[i].name} (serial {i + 1})" for i in g)
        print(f"  group {n} (label beta {n + 1}): {ats}")
    for (typ, idx), name in zip(cv_lines, cvset.names):
        print(f"  {name:6s} {typ:5s} groups {idx}")
    print(f"wrote {a.label} and {a.cvinp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
