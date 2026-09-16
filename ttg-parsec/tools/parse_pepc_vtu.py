#!/usr/bin/env python3
"""Parses a PEPC pepc-gravity VTU output file (module_vtk.f90's "binary"
format: base64 of a length header followed immediately by the raw payload,
respecting the file's declared byte_order) and prints per-particle
positions and el_field (= acceleration for the gravity backend) sorted by
particle label, so they line up with pepc-ttg's particle ids (see
gen_shared_ic.py's comment: PEPC labels particles by file order, matching
our CSV's id column).
"""
import argparse
import base64
import re
import struct
import sys
import xml.etree.ElementTree as ET


def decode_array(text, byte_order, dtype):
    # module_vtk.f90 base64-encodes the 4-byte length header and the
    # payload as two SEPARATE base64 groups, concatenated with no
    # delimiter - a bare 4-byte quantity always encodes to exactly 8
    # base64 characters (with "==" padding), so that prefix is always the
    # header; the rest is the payload's own, independently-padded base64.
    text = text.strip()
    header_b64, payload_b64 = text[:8], text[8:]
    endian = ">" if byte_order == "BigEndian" else "<"
    header = base64.b64decode(header_b64)
    (payload_len,) = struct.unpack(endian + "I", header)
    payload = base64.b64decode(payload_b64)
    assert len(payload) == payload_len, (len(payload), payload_len)
    count = payload_len // struct.calcsize(endian + dtype)
    return struct.unpack(f"{endian}{count}{dtype}", payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtu")
    args = ap.parse_args()

    tree = ET.parse(args.vtu)
    root = tree.getroot()
    byte_order = root.attrib.get("byte_order", "LittleEndian")

    piece = root.find(".//Piece")
    npoints = int(piece.attrib["NumberOfPoints"])

    def get(name, dtype):
        arr = root.find(f".//DataArray[@Name='{name}']")
        return decode_array(arr.text, byte_order, dtype)

    xyz = get("xyz", "d")
    el_field = get("el_field", "d")
    mass = get("mass", "d")
    label = get("pelabel", "q")

    rows = []
    for i in range(npoints):
        rows.append(
            {
                "label": label[i],
                "pos": xyz[3 * i : 3 * i + 3],
                "accel": el_field[3 * i : 3 * i + 3],
                "mass": mass[i],
            }
        )
    rows.sort(key=lambda r: r["label"])

    for r in rows:
        ax, ay, az = r["accel"]
        print(f"label={r['label']:3d} mass={r['mass']:.6f} accel=({ax:+.9e}, {ay:+.9e}, {az:+.9e})")


if __name__ == "__main__":
    main()
