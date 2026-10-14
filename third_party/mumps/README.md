# MUMPS 5.8.2 (third-party source, CeCILL-C)

`MUMPS_5.8.2.tar.gz` is the official MUMPS 5.8.2 source distribution, unchanged:

| | |
|---|---|
| source | https://mumps-solver.org/MUMPS_5.8.2.tar.gz |
| sha256 | `eb515aa688e6dbab414bb6e889ff4c8b23f1691a843c68da5230a33ac4db7039` (the same as conda-forge's mumps 5.8.2 recipe) |
| size | 4 476 609 bytes |
| license | CeCILL-C (`LICENSE` = the tarball's notice, `CeCILL-C_V1-en.txt` = its English license text; the French text is in the tarball's `doc/`), except the parts the notice names (BSD-3-Clause AMD variants and truncated RRQR, PORD under its own license in `PORD/README`) |
| copyright | 1991-2026 CERFACS, CNRS, ENS Lyon, INP Toulouse, Inria, Mumps Technologies, University of Bordeaux |

It is shipped in the repository so that a checkout builds offline (the cluster nodes and possibly
the judging machine have no internet). hsolve's default build (`tools/dev/build.sh`, P9
integration) compiles it with `tools/dev/build_mumps.sh` through `tools/dev/mumps_cache.sh`:
sequential MUMPS with OpenMP, conda-forge's portable compiler flags (`-O2 -march=nocona
-mtune=haswell`), against the project environment's METIS, SCOTCH and OpenBLAS, **modified by
`patches/mumps-5.8.2-l0-blocking.patch`** (two block-size controls under MUMPS' L0-OpenMP layer,
KEEP(498/499), read only when set; results/p9_factspeed.md §3.4). The checksum is verified before
every unpack. `HS_MUMPS=env tools/dev/build.sh` links the environment's unmodified conda-forge
`mumps-seq` instead.

The patch is a modification of MUMPS and is distributed under the same CeCILL-C license. Binaries
built from it carry the notice in `LICENSE` and the information that MUMPS was modified by that
patch (`THIRD_PARTY_NOTICES` §3.1; `hsolve --version` names the MUMPS install, whose
`BUILD-INFO.txt` lists the patch).
