#!/usr/bin/env python3
"""Generates one shared set of initial conditions for comparing pepc-ttg
against PEPC's pepc-gravity frontend: the same particles (id, mass,
position, velocity), written in both formats.

pepc-ttg CSV: headerless id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z
PEPC binary:  56 bytes/particle, native-endian real*8 stream, no header:
              x(3), v(3), mass (matches read_particles() in
              src/frontends/pepc-gravity/module_helper.f90 - PEPC labels
              particles 0..tnp-1 in file order, so file order IS particle
              id/label for both sides).
"""
import argparse
import random
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spread", type=float, default=5.0)
    ap.add_argument("--vel-spread", type=float, default=0.0)
    ap.add_argument("--csv-out", default="shared_ic.csv")
    ap.add_argument("--pepc-out", default="shared_ic.pepc")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    particles = []
    for i in range(args.n):
        mass = rng.uniform(0.5, 1.5)
        pos = tuple(rng.uniform(-args.spread, args.spread) for _ in range(3))
        vel = tuple(rng.uniform(-args.vel_spread, args.vel_spread) for _ in range(3))
        particles.append((i, mass, pos, vel))

    with open(args.csv_out, "w") as f:
        for i, mass, pos, vel in particles:
            f.write(f"{i},{mass},{pos[0]},{pos[1]},{pos[2]},{vel[0]},{vel[1]},{vel[2]}\n")

    with open(args.pepc_out, "wb") as f:
        for _, mass, pos, vel in particles:
            f.write(struct.pack("<7d", pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], mass))

    print(f"wrote {args.n} particles to {args.csv_out} and {args.pepc_out}")


if __name__ == "__main__":
    main()
