#!/usr/bin/env python3
"""Generate the src/io test fixtures in tests/data/io/ (run from the project environment).

    python tests/data/make_io_fixtures.py [--out tests/data/io]

Every file is produced by the real producers (numpy.save / numpy.savez / scipy.sparse.save_npz /
Python zipfile) or, for MatrixMarket, written as plain text. expected.json records what the C++
readers must return:

  load    load_matrix cases: file, options, and either "error" (hs::to_string of the error code)
          or dtype, n, nnz, source_format, storage, the exact set of note tags, the dense matrix
          (re / im, row-major lists) and the expected Hermitian defect.
  npy     read_npy cases: raw dtype/shape/fortran_order and, if "convert" is set, the values in
          payload order after npy_to_vector<convert>.
  npz     read_npz cases: the member keys and per-key arrays (same fields as npy).
  errors  files that read_npz must reject, with the error code.
  golden  numpy-written .npy files that write_npy_vector / write_npy_matrix must reproduce
          byte for byte from the listed values (matrices: column-major values).
  generator  numpy / scipy / zlib versions used (deflate output depends on the zlib build).

Zip timestamps are pinned so that regenerating gives identical bytes (with the same zlib).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import struct
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent

# A small real symmetric indefinite matrix with a structurally zero diagonal entry (2, 2).
H_REAL = np.array([
    [4.0, -1.0, 0.0, 0.5, 0.0],
    [-1.0, -3.0, 2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, -1.5, 0.25],
    [0.5, 0.0, -1.5, 1.0, 0.0],
    [0.0, 0.0, 0.25, 0.0, -2.0],
])


def _complex_hermitian() -> np.ndarray:
    h = H_REAL.astype(np.complex128)
    for (i, j), v in {(0, 1): -1 + 0.5j, (2, 4): 0.25 - 2j, (1, 3): 0.75j}.items():
        h[i, j] = v
        h[j, i] = np.conj(v)
    return h


H_CPLX = _complex_hermitian()
# Non-Hermitian matrices for orientation checks (loaded with check_hermitian = false).
G_REAL = np.array([
    [1.0, 2.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 4.0],
    [5.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 6.0, 7.0],
])
G_CPLX = G_REAL + 1j * np.array([
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, -1.0],
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.25],
])


class _FixedClock:
    """Stand-in for the `time` module inside zipfile: constant member timestamps."""

    @staticmethod
    def time() -> float:
        return 1767225600.0  # 2026-01-01T00:00:00Z

    @staticmethod
    def localtime(t=None):
        return time.gmtime(1767225600.0 if t is None else t)


@contextlib.contextmanager
def _pinned_zip_time():
    saved = zipfile.time
    zipfile.time = _FixedClock
    try:
        yield
    finally:
        zipfile.time = saved


def _hermitian_defect(a: np.ndarray) -> float:
    nrm = np.linalg.norm(a)
    return 0.0 if nrm == 0 else float(np.linalg.norm(a - a.conj().T) / nrm)


def _hermitian_part(a: np.ndarray) -> np.ndarray:
    return 0.5 * a + 0.5 * a.conj().T


def _dense_fields(a: np.ndarray) -> dict:
    if np.iscomplexobj(a):
        return {"dtype": "complex128", "re": a.real.tolist(), "im": a.imag.tolist()}
    return {"dtype": "float64", "re": np.asarray(a, dtype=np.float64).tolist(), "im": None}


def _fortran_order(a: np.ndarray) -> bool:
    """The fortran_order numpy records in the header for `a`."""
    return bool(np.lib.format.header_data_from_array_1_0(a)["fortran_order"])


def _values(a: np.ndarray) -> dict:
    """Values in payload order; integers stay exact JSON integers (bools become 0/1)."""
    flat = a.ravel(order="F" if _fortran_order(a) else "C")
    if np.iscomplexobj(flat):
        return {"re": flat.real.astype(float).tolist(), "im": flat.imag.astype(float).tolist()}
    if flat.dtype.kind in "biu":
        return {"re": [int(x) for x in flat], "im": None}
    return {"re": flat.astype(float).tolist(), "im": None}


def _num(x) -> str:
    """Shortest round-trip text of a float (repr of a numpy scalar is 'np.float64(...)')."""
    return repr(float(x))


class Fixtures:
    def __init__(self, out: Path):
        self.out = out
        self.expected = {"generator": {"numpy": np.__version__, "scipy": scipy.__version__,
                                       "zlib": zlib.ZLIB_RUNTIME_VERSION},
                         "load": [], "npy": [], "npz": [], "errors": [], "golden": []}

    # -------------------------------------------------------------------------- helpers

    def path(self, name: str) -> Path:
        return self.out / name

    def load_ok(self, file: str, matrix: np.ndarray, *, fmt: str, storage: str = "full",
                notes: tuple[str, ...] = (), options: dict | None = None,
                defect: float | None = 0.0, name: str | None = None) -> None:
        """matrix: what load_matrix must return (dense); its nnz counts the nonzeros unless the
        case keeps explicit zeros (see keep_zero_nnz)."""
        entry = {"name": name or file, "file": file, "options": options or {},
                 "source_format": fmt, "storage": storage, "notes": sorted(notes),
                 "n": int(matrix.shape[0]), "nnz": int(np.count_nonzero(matrix)),
                 "defect": defect}
        entry.update(_dense_fields(matrix))
        self.expected["load"].append(entry)

    def load_error(self, file: str, error: str, *, options: dict | None = None,
                   name: str | None = None) -> None:
        self.expected["load"].append({"name": name or f"{file}:{error}", "file": file,
                                      "options": options or {}, "error": error})

    def save_npz(self, name: str, m, compressed: bool = True) -> str:
        with _pinned_zip_time():
            sp.save_npz(self.path(name), m, compressed=compressed)
        return name

    def savez(self, name: str, compressed: bool = True, **arrays) -> str:
        with _pinned_zip_time():
            (np.savez_compressed if compressed else np.savez)(self.path(name), **arrays)
        return name

    def npy(self, name: str, a: np.ndarray, convert: str | None, version=None) -> None:
        with open(self.path(name), "wb") as f:
            np.lib.format.write_array(f, a, version=version)
        entry = {"file": name, "dtype": a.dtype.str, "shape": list(a.shape),
                 "fortran_order": _fortran_order(a), "convert": convert}
        if convert is not None:
            entry.update(_values(a))
        self.expected["npy"].append(entry)

    # ----------------------------------------------------------------------- scipy npz

    def scipy_npz(self) -> None:
        a = H_REAL
        self.load_ok(self.save_npz("csr_matrix_f8.npz", sp.csr_matrix(a)), a, fmt="npz-csr")
        self.load_ok(self.save_npz("csr_array_f8.npz", sp.csr_array(a)), a, fmt="npz-csr")
        self.load_ok(self.save_npz("csr_matrix_f8_stored.npz", sp.csr_matrix(a), compressed=False),
                     a, fmt="npz-csr")
        self.load_ok(self.save_npz("csc_matrix_c16.npz", sp.csc_matrix(H_CPLX)), H_CPLX, fmt="npz-csc")
        self.load_ok(self.save_npz("csc_array_f8.npz", sp.csc_array(a)), a, fmt="npz-csc")
        self.load_ok(self.save_npz("coo_matrix_c16.npz", sp.coo_matrix(H_CPLX)), H_CPLX, fmt="npz-coo")

        # Orientation checks on non-Hermitian matrices (check disabled): csr/csc/coo must agree.
        unchecked = {"check_hermitian": False}
        for fmt, ctor in (("csr", sp.csr_matrix), ("csc", sp.csc_matrix), ("coo", sp.coo_matrix)):
            self.load_ok(self.save_npz(f"general_{fmt}_c16.npz", ctor(G_CPLX)), G_CPLX,
                         fmt=f"npz-{fmt}", notes=("unchecked",), options=unchecked, defect=None)
        self.load_ok(self.save_npz("general_csc_f8.npz", sp.csc_matrix(G_REAL)), G_REAL,
                     fmt="npz-csc", notes=("unchecked",), options=unchecked, defect=None)

        # Index and value dtypes.
        m = sp.csr_matrix(a)
        m.indices = m.indices.astype(np.int64)
        m.indptr = m.indptr.astype(np.int64)
        name = self.save_npz("csr_int64_indices_f8.npz", m)
        with np.load(self.path(name)) as z:
            assert z["indices"].dtype == np.int64 and z["indptr"].dtype == np.int64
        self.load_ok(name, a, fmt="npz-csr")
        self.load_ok(self.save_npz("csr_f4.npz", sp.csr_matrix(a.astype(np.float32))), a, fmt="npz-csr")
        self.load_ok(self.save_npz("csr_c8.npz", sp.csr_matrix(H_CPLX.astype(np.complex64))), H_CPLX,
                     fmt="npz-csr")
        self.load_ok(self.save_npz("csr_i8_values.npz", sp.csr_matrix(np.round(4 * a).astype(np.int64))),
                     np.round(4 * a), fmt="npz-csr")
        self.load_ok(self.save_npz("csr_complex_zero_imag.npz", sp.csr_matrix(a.astype(np.complex128))),
                     a, fmt="npz-csr", notes=("real",))

        # Triangular storage.
        upper = self.save_npz("csr_upper_f8.npz", sp.csr_matrix(np.triu(a)))
        self.load_ok(upper, a, fmt="npz-csr", storage="upper", notes=("storage",))
        self.load_ok(upper, a, fmt="npz-csr", storage="upper", notes=("storage",),
                     options={"storage": "upper"}, name="csr_upper_f8.npz:declared")
        self.load_error(upper, "invalid_input", options={"storage": "lower"})
        self.load_error(upper, "not_hermitian", options={"storage": "full"})
        self.load_ok(self.save_npz("csr_lower_c16.npz", sp.csr_matrix(np.tril(H_CPLX))), H_CPLX,
                     fmt="npz-csr", storage="lower", notes=("storage",))
        self.load_ok(self.save_npz("csc_upper_c16.npz", sp.csc_matrix(np.triu(H_CPLX))), H_CPLX,
                     fmt="npz-csc", storage="upper", notes=("storage",))
        self.load_error("csr_matrix_f8.npz", "invalid_input", options={"storage": "upper"})

        # Hermitian checks.
        bad = a.copy()
        bad[0, 3], bad[3, 0] = 0.5, -0.5
        self.load_error(self.save_npz("non_hermitian_f8.npz", sp.csr_matrix(bad)), "not_hermitian")
        near = a.copy()
        near[0, 1] += 3e-13  # symmetric pattern: fixed in place
        self.load_ok(self.save_npz("near_hermitian_f8.npz", sp.csr_matrix(near)), _hermitian_part(near),
                     fmt="npz-csr", notes=("hermitian",), defect=_hermitian_defect(near))
        near_asym = a.copy()
        near_asym[1, 4] = 1e-13  # (4, 1) structurally absent: needs the union pattern
        self.load_ok(self.save_npz("near_hermitian_asym_f8.npz", sp.csr_matrix(near_asym)),
                     _hermitian_part(near_asym), fmt="npz-csr", notes=("hermitian",),
                     defect=_hermitian_defect(near_asym))
        near_c = H_CPLX.copy()
        near_c[2, 4] += 2e-13j
        near_c[0, 0] += 1e-13j  # complex diagonal must become real
        self.load_ok(self.save_npz("near_hermitian_c16.npz", sp.csr_matrix(near_c)), _hermitian_part(near_c),
                     fmt="npz-csr", notes=("hermitian",), defect=_hermitian_defect(near_c))
        self.load_ok("non_hermitian_f8.npz", bad, fmt="npz-csr", notes=("unchecked",),
                     options={"check_hermitian": False}, defect=None, name="non_hermitian_f8.npz:unchecked")
        self.load_ok("near_hermitian_f8.npz", near, fmt="npz-csr", notes=("unchecked",),
                     options={"check_hermitian": False}, defect=None, name="near_hermitian_f8.npz:unchecked")
        self.load_error("near_hermitian_f8.npz", "not_hermitian", options={"hermitian_tol": 1e-15},
                        name="near_hermitian_f8.npz:tight_tol")

        # Non-canonical CSR: unsorted rows, duplicates, explicit zeros.
        data = np.array([-1.0, 4.0, 0.25, 0.25, 2.0, -3.0, -1.0, 0.0, 2.0, -1.5, 0.25, -1.0, 0.5, 1.0,
                         -1.5, -2.0, 0.25])
        indices = np.array([1, 0, 3, 3, 2, 1, 0, 2, 1, 3, 4, 3, 0, 3, 2, 4, 2], dtype=np.int32)
        indptr = np.array([0, 4, 7, 11, 15, 17], dtype=np.int32)
        # Unsorted rows; (0,3) = 0.25 + 0.25 is a duplicate; (2,2) is an explicit zero; (3,3) =
        # -1 + 1 is a duplicate that cancels. The result is H_REAL with a zero (3,3) entry.
        m = sp.csr_matrix((data, indices, indptr), shape=(5, 5))
        name = self.save_npz("csr_unsorted_dups_f8.npz", m)
        with np.load(self.path(name)) as z:
            assert np.array_equal(z["indices"], indices)
        want = m.toarray()
        self.load_ok(name, want, fmt="npz-csr", notes=("duplicates", "zeros"))
        keep = dict(self.expected["load"][-1])
        keep.update(name=f"{name}:keep_zeros", options={"drop_zeros": False}, notes=["duplicates"],
                    nnz=int(np.count_nonzero(want)) + 2)  # explicit (2,2) and cancelled (3,3)
        self.expected["load"].append(keep)

        # H_REAL's entries plus duplicates on the diagonal ((0,0) += 1, (1,1) += -3, (3,3) += 0,
        # (4,4) += 2 which cancels) and an explicit zero at (2,2), in a scrambled order.
        base = sp.coo_matrix(a)
        rows = np.concatenate([base.row, [0, 1, 3, 4, 2]])
        cols = np.concatenate([base.col, [0, 1, 3, 4, 2]])
        vals = np.concatenate([base.data, [1.0, -3.0, 0.0, 2.0, 0.0]])
        perm = np.random.default_rng(7).permutation(len(vals))
        coo = sp.coo_matrix((vals[perm], (rows[perm], cols[perm])), shape=(5, 5))
        name = self.save_npz("coo_dups_f8.npz", coo)
        self.load_ok(name, coo.toarray(), fmt="npz-coo", notes=("duplicates", "zeros"))

        # Variants of the npz layout.
        c = sp.csr_matrix(a)
        self.load_ok(self.savez("csr_format_u3.npz", format=np.array("csr"), shape=np.array(c.shape),
                                data=c.data, indices=c.indices, indptr=c.indptr), a, fmt="npz-csr")
        self.load_ok(self.savez("csr_extra_member.npz", format=b"csr", shape=np.array(c.shape),
                                data=c.data, indices=c.indices, indptr=c.indptr,
                                comment=np.arange(3)), a, fmt="npz-csr", notes=("ignored",))
        co = sp.coo_matrix(H_CPLX)
        self.load_ok(self.savez("coo_coords_c16.npz", format=b"coo", shape=np.array(co.shape),
                                data=co.data, coords=np.array([co.row, co.col])), H_CPLX, fmt="npz-coo")
        self.load_error(self.save_npz("coo_array_1d.npz", sp.coo_array(np.array([1.0, 0.0, 2.0]))),
                        "invalid_input")
        self.load_error(self.save_npz("bsr_f8.npz", sp.bsr_matrix(a)), "unsupported")
        self.load_error(self.save_npz("dia_f8.npz", sp.dia_matrix(a)), "unsupported")
        self.load_error(self.save_npz("csr_non_square.npz", sp.csr_matrix(np.ones((3, 4)))), "invalid_input")
        inf = a.copy()
        inf[1, 1] = np.inf
        self.load_error(self.save_npz("csr_inf.npz", sp.csr_matrix(inf)), "invalid_input")
        self.load_error(self.savez("plain_arrays.npz", x=np.arange(4.0)), "invalid_input")
        self.load_error(self.savez("csr_bad_indptr.npz", format=b"csr", shape=np.array([5, 5]),
                                   data=c.data, indices=c.indices, indptr=c.indptr[::-1].copy()),
                        "invalid_input")
        self.load_error(self.savez("csr_bad_column.npz", format=b"csr", shape=np.array([5, 5]),
                                   data=c.data, indices=np.where(c.indices == 4, 7, c.indices).astype(np.int32),
                                   indptr=c.indptr), "invalid_input")

    # ---------------------------------------------------------------- zip container edge cases

    def zip_variants(self) -> None:
        c = sp.csr_matrix(H_REAL)
        arrays = {"indices": c.indices, "indptr": c.indptr, "format": np.array(b"csr"),
                  "shape": np.array(c.shape), "data": c.data}
        # Python zipfile emits zip64 extras in the central directory and the zip64 end record only
        # beyond its limits; lowering them makes it write a fully zip64 archive for tiny members.
        saved = (zipfile.ZIP64_LIMIT, zipfile.ZIP_FILECOUNT_LIMIT)
        zipfile.ZIP64_LIMIT, zipfile.ZIP_FILECOUNT_LIMIT = 0, 0
        try:
            with _pinned_zip_time(), zipfile.ZipFile(self.path("zip64_deflate.npz"), "w",
                                                     compression=zipfile.ZIP_DEFLATED) as z:
                z.comment = b"hsolve io fixture: zip64 + deflate"
                for key, arr in arrays.items():
                    buf = io.BytesIO()
                    np.save(buf, arr)
                    with z.open(key + ".npy", "w", force_zip64=True) as f:
                        f.write(buf.getvalue())
        finally:
            zipfile.ZIP64_LIMIT, zipfile.ZIP_FILECOUNT_LIMIT = saved
        raw = self.path("zip64_deflate.npz").read_bytes()
        assert raw.rfind(b"PK\x06\x06") > 0 and raw.rfind(b"PK\x06\x07") > 0
        self.load_ok("zip64_deflate.npz", H_REAL, fmt="npz-csr")

        # Saturate the classic end record so that only the zip64 record holds the truth.
        end = raw.rfind(b"PK\x05\x06")
        rec = bytearray(raw[end:end + 22])
        struct.pack_into("<HHII", rec, 8, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        sat = raw[:end] + bytes(rec) + raw[end + 22:]
        self.path("zip64_saturated_eocd.npz").write_bytes(sat)
        with zipfile.ZipFile(self.path("zip64_saturated_eocd.npz")) as z:  # Python agrees
            assert z.testzip() is None
        self.load_ok("zip64_saturated_eocd.npz", H_REAL, fmt="npz-csr")

        # An archive comment containing the end-record signature (Python's reader trips on it).
        comment = b"comment with a stray PK\x05\x06 signature inside"
        rec = bytearray(raw[end:end + 22])
        struct.pack_into("<H", rec, 20, len(comment))
        self.path("comment_with_signature.npz").write_bytes(raw[:end] + bytes(rec) + comment)
        self.load_ok("comment_with_signature.npz", H_REAL, fmt="npz-csr")

        # Corruption: a flipped payload byte in a stored member (CRC) and in a deflated one.
        def flip_member_byte(src: str, dst: str, member: str, at_end: bool) -> None:
            data = bytearray(self.path(src).read_bytes())
            with zipfile.ZipFile(self.path(src)) as z:
                info = z.getinfo(member)
            nl, el = struct.unpack_from("<HH", data, info.header_offset + 26)
            start = info.header_offset + 30 + nl + el
            pos = start + info.compress_size - 1 if at_end else start + info.compress_size // 2
            data[pos] ^= 0x55
            self.path(dst).write_bytes(bytes(data))

        flip_member_byte("csr_matrix_f8_stored.npz", "bad_crc.npz", "data.npy", at_end=True)
        self.expected["errors"].append({"file": "bad_crc.npz", "reader": "read_npz", "error": "io"})
        self.load_error("bad_crc.npz", "io")
        flip_member_byte("csr_matrix_f8.npz", "bad_deflate.npz", "data.npy", at_end=False)
        self.expected["errors"].append({"file": "bad_deflate.npz", "reader": "read_npz", "error": "io"})
        self.load_error("bad_deflate.npz", "io")

    # ------------------------------------------------------------------------------ dense npy

    def dense_npy(self) -> None:
        self.npy("dense_c_f8.npy", H_REAL, "float64")
        self.load_ok("dense_c_f8.npy", H_REAL, fmt="npy-dense")
        self.npy("dense_f_c16.npy", np.asfortranarray(H_CPLX), "complex128")
        self.load_ok("dense_f_c16.npy", H_CPLX, fmt="npy-dense")
        self.npy("dense_be_f8.npy", H_REAL.astype(">f8"), "float64")
        self.load_ok("dense_be_f8.npy", H_REAL, fmt="npy-dense")
        self.npy("dense_f_be_c8.npy", np.asfortranarray(H_CPLX.astype(">c8")), "complex128")
        self.load_ok("dense_f_be_c8.npy", H_CPLX, fmt="npy-dense")

        rect = np.arange(12.0).reshape(3, 4) - 5.5
        self.npy("npy_c_f8.npy", rect, "float64")
        self.npy("npy_f_f8.npy", np.asfortranarray(rect), "float64")
        self.load_error("npy_c_f8.npy", "invalid_input", name="npy_c_f8.npy:non_square")
        self.npy("npy_be_c16.npy", np.array([1 + 2j, -0.5 - 4j, 3e-300 + 1e300j], dtype=">c16"), "complex128")
        self.npy("npy_be_i4.npy", np.array([-7, 0, 2**31 - 1], dtype=">i4"), "int32")
        self.npy("npy_v2_f8.npy", np.array([1.5, -2.5]), "float64", version=(2, 0))
        self.npy("npy_v3_c16.npy", np.array([1 - 1j]), "complex128", version=(3, 0))
        self.npy("npy_scalar_f8.npy", np.array(3.25), "float64")
        self.load_error("npy_scalar_f8.npy", "invalid_input", name="npy_scalar_f8.npy:not_2d")
        self.npy("npy_b1.npy", np.array([True, False, True]), "int64")
        self.npy("npy_u1.npy", np.array([0, 200, 255], dtype=np.uint8), "int32")
        self.npy("npy_u8.npy", np.array([0, 2**40], dtype=np.uint64), "int64")
        self.npy("npy_c8.npy", np.array([0.5 - 0.25j], dtype=np.complex64), "complex128")
        self.npy("npy_f4.npy", np.array([0.1, -3.0], dtype=np.float32), "float64")
        self.npy("npy_i8.npy", np.array([-(2**53), 7], dtype=np.int64), "float64")
        self.npy("npy_u3.npy", np.array(["ab", "xyz"]), None)
        self.npy("npy_s3.npy", np.array([b"csr"]), None)
        self.npy("npy_empty_f8.npy", np.zeros((0, 3)), "float64")

    # ----------------------------------------------------------------------------- read_npz

    def npz_archives(self) -> None:
        arrays = {"a": np.arange(6.0).reshape(2, 3), "b": np.array([1 + 1j, 2 - 2j]),
                  "c": np.asfortranarray(np.arange(6, dtype=np.int32).reshape(3, 2)),
                  "flag": np.array(True)}
        for name, compressed in (("multi_stored.npz", False), ("multi_deflate.npz", True)):
            self.savez(name, compressed=compressed, **arrays)
            with _pinned_zip_time(), zipfile.ZipFile(self.path(name), "a") as z:
                z.writestr("readme.txt", "not an array: skipped by read_npz")
            self.expected["npz"].append({
                "file": name, "keys": sorted(arrays),
                "arrays": {k: {"dtype": v.dtype.str, "shape": list(v.shape),
                               "fortran_order": _fortran_order(v), "convert": "complex128",
                               **_values(v)} for k, v in arrays.items()}})

    # ---------------------------------------------------------------------------- MatrixMarket

    def write_mtx(self, name: str, banner: str, size: str, lines: list[str], newline: str = "\n") -> str:
        text = [f"%%MatrixMarket matrix {banner}", "% generated by tests/data/make_io_fixtures.py", "%",
                size, *lines]
        self.path(name).write_bytes((newline.join(text) + newline).encode())
        return name

    def mtx(self) -> None:
        a, h = H_REAL, H_CPLX
        n = a.shape[0]

        def coo_lines(m, keep) -> list[str]:
            out = []
            for j in range(n):  # column-major order, as most writers do
                for i in range(n):
                    if m[i, j] != 0 and keep(i, j):
                        v = m[i, j]
                        out.append(f"{i + 1} {j + 1} " +
                                   (f"{_num(v.real)} {_num(v.imag)}" if np.iscomplexobj(m) else _num(v)))
            return out

        def size(lines):
            return f"{n} {n} {len(lines)}"

        lines = coo_lines(a, lambda i, j: True)
        lines.insert(3, "")  # blank lines are tolerated between entries
        self.load_ok(self.write_mtx("real_general.mtx", "coordinate real general",
                                    f"{n} {n} {len(lines) - 1}", lines), a, fmt="mtx")
        lines = coo_lines(a, lambda i, j: i >= j)
        name = self.write_mtx("real_symmetric.mtx", "coordinate real symmetric", size(lines), lines)
        self.load_ok(name, a, fmt="mtx", storage="lower", notes=("storage",))
        self.load_error(name, "invalid_input", options={"storage": "full"}, name=f"{name}:storage_full")
        self.load_error(name, "invalid_input", options={"storage": "upper"}, name=f"{name}:storage_upper")
        lines = coo_lines(a, lambda i, j: i <= j)
        self.load_ok(self.write_mtx("real_symmetric_upper.mtx", "coordinate double symmetric", size(lines),
                                    lines), a, fmt="mtx", storage="upper", notes=("storage",))
        lines = coo_lines(h, lambda i, j: i >= j)
        self.load_ok(self.write_mtx("complex_hermitian.mtx", "coordinate complex hermitian", size(lines),
                                    lines), h, fmt="mtx", storage="lower", notes=("storage",))
        lines = [ln.rsplit(" ", 1)[0] for ln in coo_lines(a, lambda i, j: i >= j)]
        self.load_ok(self.write_mtx("pattern_symmetric.mtx", "coordinate pattern symmetric", size(lines),
                                    lines), (a != 0).astype(float), fmt="mtx", storage="lower",
                     notes=("storage",))
        ints = np.array([[2, -1, 0], [-1, 5, 3], [0, 3, -4]], dtype=float)
        lines = ["1 1 1", "2 1 -1", "1 2 -1", "2 2 5", "3 2 3", "2 3 3", "3 3 -4", "1 1 +1"]
        self.load_ok(self.write_mtx("integer_general_crlf.mtx", "coordinate integer general",
                                    f"3 3 {len(lines)}", lines, newline="\r\n"),
                     ints, fmt="mtx", notes=("duplicates",))
        dense = [_num(a[i, j]) for j in range(n) for i in range(n)]
        self.load_ok(self.write_mtx("array_general.mtx", "array real general", f"{n} {n}", dense), a, fmt="mtx")
        dense = [_num(a[i, j]) for j in range(n) for i in range(j, n)]
        self.load_ok(self.write_mtx("array_symmetric.mtx", "array real symmetric", f"{n} {n}", dense), a,
                     fmt="mtx", storage="lower", notes=("storage",))
        dense = [f"{_num(h[i, j].real)} {_num(h[i, j].imag)}" for j in range(n) for i in range(j, n)]
        self.load_ok(self.write_mtx("array_complex_hermitian.mtx", "array complex hermitian", f"{n} {n}",
                                    dense), h, fmt="mtx", storage="lower", notes=("storage",))
        # complex symmetric (A = A^T): Hermitian only when every entry is real.
        lines = [f"{ln} 0.0" for ln in coo_lines(a, lambda i, j: i >= j)]
        self.load_ok(self.write_mtx("complex_symmetric_real.mtx", "coordinate complex symmetric", size(lines),
                                    lines), a, fmt="mtx", storage="lower", notes=("real", "storage"))
        cs = a.astype(complex)
        cs[1, 0] = cs[0, 1] = -1 + 1j
        lines = coo_lines(cs, lambda i, j: i >= j)
        self.load_error(self.write_mtx("complex_symmetric.mtx", "coordinate complex symmetric", size(lines),
                                       lines), "not_hermitian")
        self.load_ok("complex_symmetric.mtx", np.tril(cs) + np.tril(cs, -1).T, fmt="mtx", storage="lower",
                     notes=("storage", "unchecked"), options={"check_hermitian": False}, defect=None,
                     name="complex_symmetric.mtx:unchecked")
        self.load_error(self.write_mtx("skew_symmetric.mtx", "coordinate real skew-symmetric", "2 2 1",
                                       ["2 1 -1.0"]), "invalid_input")
        self.load_error(self.write_mtx("rectangular.mtx", "coordinate real general", "2 3 1", ["1 3 1.0"]),
                        "invalid_input")
        self.load_error(self.write_mtx("truncated.mtx", "coordinate real general", "3 3 4",
                                       ["1 1 1.0", "2 2 1.0"]), "io")
        self.load_error(self.write_mtx("bad_index.mtx", "coordinate real general", "3 3 1", ["4 1 1.0"]), "io")

    # ------------------------------------------------------------------------- golden writers

    def golden(self) -> None:
        def vec(name, v, type_):
            np.save(self.path(name), v)
            self.expected["golden"].append({"file": name, "kind": "vector", "type": type_,
                                            "rows": int(v.size), "cols": 1, **_values(v)})

        def mat(name, m, type_):
            np.save(self.path(name), np.asfortranarray(m))
            col_major = m.ravel(order="F")
            vals = ({"re": col_major.real.tolist(), "im": col_major.imag.tolist()} if np.iscomplexobj(m)
                    else {"re": col_major.tolist(), "im": None})
            self.expected["golden"].append({"file": name, "kind": "matrix", "type": type_,
                                            "rows": int(m.shape[0]), "cols": int(m.shape[1]), **vals})

        vec("golden_vec_f8.npy", np.array([1.0, -2.5, 3e-300, 1e300, -0.0]), "float64")
        vec("golden_vec_c16.npy", np.array([1 - 2j, 0.5j, -3.25]), "complex128")
        vec("golden_vec_f4.npy", np.array([0.5, -1.25, 3.0], dtype=np.float32), "float32")
        vec("golden_vec_i4.npy", np.array([-(2**31), 0, 2**31 - 1], dtype=np.int32), "int32")
        vec("golden_vec_i8.npy", np.array([-(2**63), 42, 2**63 - 1], dtype=np.int64), "int64")
        vec("golden_vec_long_f8.npy", np.arange(1000) * 0.25 - 7.0, "float64")
        vec("golden_vec_empty_f8.npy", np.zeros(0), "float64")
        mat("golden_mat_f8.npy", np.arange(6.0).reshape(3, 2) - 1.5, "float64")
        mat("golden_mat_c16.npy", (np.arange(6.0) + 1j * np.arange(6.0)[::-1]).reshape(2, 3), "complex128")
        mat("golden_mat_col_f8.npy", np.array([[1.0], [2.0], [3.0], [4.0]]), "float64")
        mat("golden_mat_row_c16.npy", np.array([[1 + 1j, 2 + 0j, -3j]]), "complex128")
        mat("golden_mat_empty_f8.npy", np.zeros((0, 3)), "float64")
        mat("golden_mat_wide_f8.npy", np.arange(24.0).reshape(2, 12), "float64")

    def write(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        for old in self.out.iterdir():
            if old.suffix in (".npy", ".npz", ".mtx", ".json"):
                old.unlink()
        self.scipy_npz()
        self.zip_variants()
        self.dense_npy()
        self.npz_archives()
        self.mtx()
        self.golden()
        names = [e["name"] for e in self.expected["load"]]
        assert len(names) == len(set(names)), "load case names must be unique"
        (self.out / "expected.json").write_text(json.dumps(self.expected, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "io", help="output directory")
    args = ap.parse_args()
    Fixtures(args.out).write()


if __name__ == "__main__":
    main()
