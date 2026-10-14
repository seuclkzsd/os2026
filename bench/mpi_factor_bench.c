/* bench/mpi_factor_bench.c - P9 "mpi": upper bound of hybrid MPI + OpenMP MUMPS for hsolve's
 * factorizations (the certificate's inertia-only counts and the main LDL^T with its block solves).
 *
 * A standalone C program (not part of the hsolve build: CMake globs only the .cpp files of bench/), built twice
 * by tools/bench/mpi_build.sh from this one source:
 *   build/mpi/mpi_factor_bench      against the MPI MUMPS 5.8.2 of the os2026-ref env (Open MPI 5,
 *                                   ScaLAPACK, PT-SCOTCH, ParMETIS, OpenMP), run under mpiexec;
 *   build/mpi/seq_factor_bench      against the sequential MUMPS 5.8.2 hsolve links (os2026 env,
 *                                   libseq MPI stubs; -DHS_SEQ_MUMPS) - hsolve's 1 x 16 path.
 * Both use hsolve's MUMPS controls (src/factor/mumps_factorization.cpp, driver defaults): SYM = 2
 * (LDL^T with 2x2 pivots), PAR = 1, ICNTL(8) = 7 scaling, ICNTL(13) = 1 (root front without
 * ScaLAPACK: the exact inertia - see --icntl13), ICNTL(14) = 35, ICNTL(16) = threads, ICNTL(24) = 1
 * with CNTL(3) = 1e-8 (null-pivot detection), ICNTL(48) = 1 (L0-OpenMP), no BLR, in core, and for
 * the counts ICNTL(31) = 1 (factors discarded; read at the analysis).
 *
 * Input: K(sigma) as written by tools/bench/mpi_dump_kernel.py (upper triangle, 1-based COO;
 * magic "HSKMAT1\0", int64 n, int64 nnz, int32 irn[], int32 jcn[], float64 a[]) and optionally a
 * pivot order (magic "HSPERM1\0", int64 n, int32 perm[n]: perm[i] = pivot position of variable i,
 * MUMPS' SYM_PERM / PERM_IN convention; --write-perm writes one from an analysis).
 *
 * Per mode in --modes (each a fresh MUMPS instance, run one after the other in the process):
 *   count   analysis with ICNTL(31) = 1, then --reps inertia-only numeric factorizations
 *   full    analysis, --reps numeric factorizations keeping the factors, then --solve-reps block
 *           solves of --nrhs right-hand sides (centralized dense RHS, like hsolve) and the
 *           relative residual of the first column
 * Every phase starts and ends at an MPI barrier; rank 0 prints one JSON line per phase to stdout:
 *   {"event":"analysis"|"factorize"|"solve"|"run", "s":..., "neg":INFOG(12), "null":INFOG(28),
 *    "flops":RINFOG(3), "mem_max_mb":INFOG(21), "mem_total_mb":INFOG(22), ...}
 * "run" (last) has the per-rank peak RSS (max and sum of ru_maxrss) and --expect-neg's verdict.
 *
 * usage:
 *   [mpiexec -n R ... tools/bench/mpi_bind.sh] build/mpi/mpi_factor_bench --matrix K.bin
 *       [--perm P.bin | --ordering 3|5|... | --par-analysis 1|2] [--modes count,full] [--reps 1]
 *       [--nrhs 16] [--solve-reps 3] [--threads T] [--dist 0|3] [--icntl13 1] [--l0 1]
 *       [--write-perm OUT.bin] [--expect-neg N] [--label TEXT] [--icntl I=V]... [--cntl I=V]...
 *   --ordering   ICNTL(7) of a sequential analysis (ICNTL(28) = 1) when no --perm is given
 *                (3 SCOTCH, 5 METIS; default 5)
 *   --par-analysis  ICNTL(28) = 2 with ICNTL(29) = 1 (PT-SCOTCH) or 2 (ParMETIS); MPI build only
 *   --dist       ICNTL(18): 0 centralized matrix on rank 0 (hsolve), 3 distributed (each rank reads
 *                its slice of the entries)
 *   --threads    ICNTL(16) OpenMP threads per rank (default OMP_NUM_THREADS, else 1)
 * The binding of ranks to cores is the launcher's job: tools/bench/mpi_bind.sh pins each rank to its
 * block of the inherited cpuset (the guard's slot cores); tools/bench/mpi_factor_runs.sh runs the
 * P9 layouts 1x16 / 2x8 / 4x4 / 8x2 / 16x1 interleaved, each under its own guard.
 *
 * The P9 measurements (results/p9_mpi.md) - from the work tree on the node, the outer guard holding
 * the slot:
 *   bash tools/bench/mpi_build.sh
 *   env/run.sh python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 -- \
 *       bash tools/bench/mpi_factor_runs.sh OUTDIR CASE.bin PERM.bin REPS
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

#include "dmumps_c.h"
#include "mpi.h"

#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654
#define ICNTL(I) icntl[(I)-1]
#define CNTL(I) cntl[(I)-1]
#define INFOG(I) infog[(I)-1]
#define INFO(I) info[(I)-1]
#define RINFOG(I) rinfog[(I)-1]

/* ------------------------------------------------------------------ MPI (stubs for libseq) */

static int g_rank = 0, g_size = 1;

#ifdef HS_SEQ_MUMPS
/* libseq's mpi.h provides MPI_Init / MPI_Comm_rank / MPI_Finalize / MPI_Wtime only. */
static void barrier(void) {}
static void reduce_sum_d(double* v, int count) { (void)v; (void)count; }
static void reduce_max_d(double* v, int count) { (void)v; (void)count; }
static void bcast_d(double* v, int64_t count) { (void)v; (void)count; }
static void comm_size(void) { g_size = 1; }
#else
static void barrier(void) { MPI_Barrier(MPI_COMM_WORLD); }
/* In-place sum / max onto rank 0. */
static void reduce_op(double* v, int count, MPI_Op op) {
  if (g_rank == 0)
    MPI_Reduce(MPI_IN_PLACE, v, count, MPI_DOUBLE, op, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(v, NULL, count, MPI_DOUBLE, op, 0, MPI_COMM_WORLD);
}
static void reduce_sum_d(double* v, int count) { reduce_op(v, count, MPI_SUM); }
static void reduce_max_d(double* v, int count) { reduce_op(v, count, MPI_MAX); }
static void bcast_d(double* v, int64_t count) {
  const int64_t chunk = 1 << 26;
  for (int64_t off = 0; off < count; off += chunk) {
    const int64_t m = count - off < chunk ? count - off : chunk;
    MPI_Bcast(v + off, (int)m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}
static void comm_size(void) { MPI_Comm_size(MPI_COMM_WORLD, &g_size); }
#endif

static double now(void) { return MPI_Wtime(); }

static void die(const char* msg) {
  fprintf(stderr, "mpi_factor_bench (rank %d): %s\n", g_rank, msg);
  fflush(stderr);
  exit(2);
}

/* ------------------------------------------------------------------ options */

#define MAX_EXTRA 16
typedef struct {
  const char* matrix;
  const char* perm;
  const char* write_perm;
  const char* label;
  const char* modes;
  int ordering, par_analysis, reps, nrhs, solve_reps, threads, dist, icntl13, l0;
  long long expect_neg;
  int n_icntl, icntl_idx[MAX_EXTRA], icntl_val[MAX_EXTRA];
  int n_cntl, cntl_idx[MAX_EXTRA];
  double cntl_val[MAX_EXTRA];
} Options;

static int parse_int(const char* s, const char* what) {
  char* end = NULL;
  errno = 0;
  const long v = strtol(s, &end, 10);
  if (errno || end == s || *end) {
    fprintf(stderr, "mpi_factor_bench: bad integer for %s: '%s'\n", what, s);
    exit(2);
  }
  return (int)v;
}

static Options parse_options(int argc, char** argv) {
  Options o;
  memset(&o, 0, sizeof o);
  o.modes = "count,full";
  o.ordering = 5;
  o.reps = 1;
  o.nrhs = 16;
  o.solve_reps = 3;
  o.icntl13 = 1;
  o.l0 = 1;
  o.expect_neg = -1;
  o.label = "";
  const char* env_threads = getenv("OMP_NUM_THREADS");
  o.threads = env_threads ? atoi(env_threads) : 1;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    const char* v = i + 1 < argc ? argv[i + 1] : NULL;
#define TAKE(name) (strcmp(a, name) == 0 && v && (++i, 1))
    if (TAKE("--matrix")) o.matrix = v;
    else if (TAKE("--perm")) o.perm = v;
    else if (TAKE("--write-perm")) o.write_perm = v;
    else if (TAKE("--label")) o.label = v;
    else if (TAKE("--modes")) o.modes = v;
    else if (TAKE("--ordering")) o.ordering = parse_int(v, a);
    else if (TAKE("--par-analysis")) o.par_analysis = parse_int(v, a);
    else if (TAKE("--reps")) o.reps = parse_int(v, a);
    else if (TAKE("--nrhs")) o.nrhs = parse_int(v, a);
    else if (TAKE("--solve-reps")) o.solve_reps = parse_int(v, a);
    else if (TAKE("--threads")) o.threads = parse_int(v, a);
    else if (TAKE("--dist")) o.dist = parse_int(v, a);
    else if (TAKE("--icntl13")) o.icntl13 = parse_int(v, a);
    else if (TAKE("--l0")) o.l0 = parse_int(v, a);
    else if (TAKE("--expect-neg")) o.expect_neg = atoll(v);
    else if (TAKE("--icntl") || TAKE("--cntl")) {
      const int is_icntl = strcmp(a, "--icntl") == 0;
      const char* eq = strchr(v, '=');
      if (!eq) die("--icntl / --cntl take I=V");
      char idx[16] = {0};
      const size_t len = (size_t)(eq - v) < sizeof idx - 1 ? (size_t)(eq - v) : sizeof idx - 1;
      memcpy(idx, v, len);
      if (is_icntl) {
        if (o.n_icntl == MAX_EXTRA) die("too many --icntl");
        o.icntl_idx[o.n_icntl] = parse_int(idx, a);
        o.icntl_val[o.n_icntl++] = parse_int(eq + 1, a);
      } else {
        if (o.n_cntl == MAX_EXTRA) die("too many --cntl");
        o.cntl_idx[o.n_cntl] = parse_int(idx, a);
        o.cntl_val[o.n_cntl++] = atof(eq + 1);
      }
    } else {
      fprintf(stderr, "mpi_factor_bench: unknown or incomplete argument '%s' (see the header of "
                      "bench/mpi_factor_bench.c)\n", a);
      exit(2);
    }
#undef TAKE
  }
  if (!o.matrix) die("--matrix is required");
  if (o.threads < 1) o.threads = 1;
  if (o.reps < 1) o.reps = 1;
  if (o.dist != 0 && o.dist != 3) die("--dist must be 0 or 3");
#ifdef HS_SEQ_MUMPS
  if (o.par_analysis) die("--par-analysis needs the MPI build");
#endif
  return o;
}

/* ------------------------------------------------------------------ input */

typedef struct {
  int64_t n, nnz;          /* global */
  int64_t nnz_here, first; /* entries held by this rank: [first, first + nnz_here) */
  int* irn;
  int* jcn;
  double* a;
} Matrix;

static void read_exact(FILE* f, void* p, size_t size, size_t count, const char* what) {
  if (count && fread(p, size, count, f) != count) die(what);
}

static void* xmalloc(size_t bytes) {
  void* p = malloc(bytes ? bytes : 1);
  if (!p) die("out of memory");
  return p;
}

/* Rank 0 reads everything (dist 0), or every rank its contiguous slice of the entries (dist 3). */
static Matrix read_matrix(const char* path, int dist) {
  Matrix m;
  memset(&m, 0, sizeof m);
  FILE* f = fopen(path, "rb");
  if (!f) die("cannot open --matrix");
  char magic[8];
  int64_t hdr[2];
  read_exact(f, magic, 1, 8, "short matrix header");
  if (memcmp(magic, "HSKMAT1", 8) != 0) die("not an HSKMAT1 file");
  read_exact(f, hdr, sizeof(int64_t), 2, "short matrix header");
  m.n = hdr[0];
  m.nnz = hdr[1];
  if (dist == 0) {
    m.first = 0;
    m.nnz_here = g_rank == 0 ? m.nnz : 0;
  } else {
    m.first = m.nnz * g_rank / g_size;
    m.nnz_here = m.nnz * (g_rank + 1) / g_size - m.first;
  }
  m.irn = xmalloc((size_t)m.nnz_here * sizeof(int));
  m.jcn = xmalloc((size_t)m.nnz_here * sizeof(int));
  m.a = xmalloc((size_t)m.nnz_here * sizeof(double));
  const long base = 8 + 16;
  if (m.nnz_here) {
    if (fseek(f, base + (long)(m.first * 4), SEEK_SET)) die("seek irn");
    read_exact(f, m.irn, sizeof(int), (size_t)m.nnz_here, "short irn");
    if (fseek(f, base + (long)(m.nnz * 4 + m.first * 4), SEEK_SET)) die("seek jcn");
    read_exact(f, m.jcn, sizeof(int), (size_t)m.nnz_here, "short jcn");
    if (fseek(f, base + (long)(m.nnz * 8 + m.first * 8), SEEK_SET)) die("seek a");
    read_exact(f, m.a, sizeof(double), (size_t)m.nnz_here, "short a");
  }
  fclose(f);
  return m;
}

static int* read_perm(const char* path, int64_t n) {
  FILE* f = fopen(path, "rb");
  if (!f) die("cannot open --perm");
  char magic[8];
  int64_t pn;
  read_exact(f, magic, 1, 8, "short perm header");
  if (memcmp(magic, "HSPERM1", 8) != 0) die("not an HSPERM1 file");
  read_exact(f, &pn, sizeof pn, 1, "short perm header");
  if (pn != n) die("pivot order and matrix have different orders");
  int* p = xmalloc((size_t)n * sizeof(int));
  read_exact(f, p, sizeof(int), (size_t)n, "short perm");
  fclose(f);
  return p;
}

static void write_perm(const char* path, const int* p, int64_t n) {
  FILE* f = fopen(path, "wb");
  if (!f) die("cannot open --write-perm");
  fwrite("HSPERM1", 1, 8, f);
  fwrite(&n, sizeof n, 1, f);
  fwrite(p, sizeof(int), (size_t)n, f);
  if (fclose(f)) die("write --write-perm");
}

/* ------------------------------------------------------------------ MUMPS */

static double decode_count(int v) { return v >= 0 ? (double)v : -(double)v * 1e6; }

static void run_job(DMUMPS_STRUC_C* id, int job, const char* what) {
  id->job = job;
  dmumps_c(id);
  if (id->INFOG(1) < 0) {
    char msg[200];
    snprintf(msg, sizeof msg, "%s failed: INFOG(1)=%d INFOG(2)=%d", what, id->INFOG(1), id->INFOG(2));
    die(msg);
  }
}

static void common_controls(DMUMPS_STRUC_C* id, const Options* o) {
  id->ICNTL(1) = -1;
  id->ICNTL(2) = -1;
  id->ICNTL(3) = -1;
  id->ICNTL(4) = 0;
  id->ICNTL(5) = 0;
  id->ICNTL(8) = 7;
  id->ICNTL(13) = o->icntl13;
  id->ICNTL(14) = 35;
  id->ICNTL(16) = o->threads;
  id->ICNTL(18) = o->dist;
  id->ICNTL(20) = 0;
  id->ICNTL(21) = 0;
  id->ICNTL(22) = 0;
  id->ICNTL(23) = 0;
  id->ICNTL(24) = 1;
  id->CNTL(3) = 1e-8;
  id->ICNTL(35) = 0;
  id->ICNTL(48) = o->l0;
  for (int i = 0; i < o->n_icntl; ++i) id->ICNTL(o->icntl_idx[i]) = o->icntl_val[i];
  for (int i = 0; i < o->n_cntl; ++i) id->CNTL(o->cntl_idx[i]) = o->cntl_val[i];
}

static void print_common(const Options* o, const char* mode) {
  printf("\"label\":\"%s\",\"mode\":\"%s\",\"ranks\":%d,\"threads\":%d,\"dist\":%d", o->label, mode,
         g_size, o->threads, o->dist);
}

/* y += K x for the entries this rank holds (upper triangle, symmetric). */
static void spmv_upper(const Matrix* m, const double* x, double* y) {
  for (int64_t e = 0; e < m->nnz_here; ++e) {
    const int64_t i = m->irn[e] - 1, j = m->jcn[e] - 1;
    y[i] += m->a[e] * x[j];
    if (i != j) y[j] += m->a[e] * x[i];
  }
}

static double norm2(const double* v, int64_t n) {
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += v[i] * v[i];
  return sqrt(s);
}

/* Runs one mode; returns INFOG(12) of its first factorization, *ok = the inertia was the same in
 * every repetition and equals --expect-neg (when given). */
static int run_mode(const Options* o, const Matrix* m, int* perm, const char* mode, double* t_total, int* ok) {
  const int count = strcmp(mode, "count") == 0;
  if (!count && strcmp(mode, "full") != 0) die("--modes takes count and/or full");
  DMUMPS_STRUC_C id;
  memset(&id, 0, sizeof id);
  id.comm_fortran = USE_COMM_WORLD;
  id.par = 1;
  id.sym = 2;
  run_job(&id, JOB_INIT, "initialization");
  common_controls(&id, o);
  id.ICNTL(31) = count ? 1 : 0;
  if (o->par_analysis) {
    id.ICNTL(28) = 2;
    id.ICNTL(29) = o->par_analysis;
  } else {
    id.ICNTL(28) = 1;
    id.ICNTL(7) = perm ? 1 : o->ordering;
  }
  if (g_rank == 0) id.n = (int)m->n;
  if (o->dist == 0) {
    if (g_rank == 0) {
      id.nnz = m->nnz;
      id.irn = m->irn;
      id.jcn = m->jcn;
      id.a = m->a;
    }
  } else {
    id.n = (int)m->n;
    id.nnz_loc = m->nnz_here;
    id.irn_loc = m->irn;
    id.jcn_loc = m->jcn;
    id.a_loc = m->a;
  }
  if (perm && g_rank == 0 && !o->par_analysis) id.perm_in = perm;

  barrier();
  double t0 = now();
  run_job(&id, 1, "analysis");
  barrier();
  double t = now() - t0;
  *t_total += t;
  if (g_rank == 0) {
    printf("{\"event\":\"analysis\",");
    print_common(o, mode);
    printf(",\"s\":%.6f,\"ordering\":%d,\"par_analysis\":%d,\"perm_in\":%d,\"flops_est\":%.6e,"
           "\"entries_est\":%.0f,\"incore_max_mb\":%d,\"incore_total_mb\":%d}\n",
           t, id.INFOG(7), o->par_analysis, perm ? 1 : 0, id.RINFOG(1), decode_count(id.INFOG(20)),
           id.INFOG(16), id.INFOG(17));
    fflush(stdout);
    if (o->write_perm && id.sym_perm) write_perm(o->write_perm, id.sym_perm, m->n);
  }

  int neg_first = -1, inertia_stable = 1;
  for (int r = 0; r < o->reps; ++r) {
    barrier();
    t0 = now();
    run_job(&id, 2, "factorization");
    barrier();
    t = now() - t0;
    *t_total += t;
    if (r == 0) neg_first = id.INFOG(12);
    else if (id.INFOG(12) != neg_first) inertia_stable = 0;
    if (g_rank == 0) {
      printf("{\"event\":\"factorize\",");
      print_common(o, mode);
      printf(",\"rep\":%d,\"s\":%.6f,\"neg\":%d,\"null\":%d,\"delayed\":%d,\"flops\":%.6e,"
             "\"entries\":%.0f,\"mem_max_mb\":%d,\"mem_total_mb\":%d,\"alloc_max_mb\":%d,"
             "\"alloc_total_mb\":%d}\n",
             r, t, id.INFOG(12), id.INFOG(28), id.INFOG(13), id.RINFOG(3), decode_count(id.INFOG(9)),
             id.INFOG(21), id.INFOG(22), id.INFOG(18), id.INFOG(19));
      fflush(stdout);
    }
  }

  if (!count && o->nrhs > 0 && o->solve_reps > 0) {
    const int64_t n = m->n;
    double* rhs = NULL;
    double* b0 = xmalloc((size_t)n * sizeof(double));
    for (int s = 0; s < o->solve_reps; ++s) {
      if (g_rank == 0) {
        if (!rhs) rhs = xmalloc((size_t)n * o->nrhs * sizeof(double));
        uint64_t state = 0x9E3779B97F4A7C15ull;
        for (int64_t i = 0; i < n * o->nrhs; ++i) {
          state = state * 6364136223846793005ull + 1442695040888963407ull;
          rhs[i] = (double)(state >> 11) / 9007199254740992.0 - 0.5;
        }
        memcpy(b0, rhs, (size_t)n * sizeof(double));
        id.rhs = rhs;
        id.nrhs = o->nrhs;
        id.lrhs = (int)n;
      }
      barrier();
      t0 = now();
      run_job(&id, 3, "solve");
      barrier();
      t = now() - t0;
      *t_total += t;
      /* Relative residual of the first column: ||K x - b|| / (||b|| + ||K||_est ||x||) is not needed;
       * ||K x - b|| / ||b|| tells a correct solve from a broken one. */
      double* x = xmalloc((size_t)n * sizeof(double));
      double* y = calloc((size_t)n, sizeof(double));
      if (!y) die("out of memory");
      if (g_rank == 0) memcpy(x, rhs, (size_t)n * sizeof(double));
      if (o->dist == 3) bcast_d(x, n);
      spmv_upper(m, x, y);
      if (o->dist == 3) {
        const int64_t chunk = 1 << 26;
        for (int64_t off = 0; off < n; off += chunk) reduce_sum_d(y + off, (int)(n - off < chunk ? n - off : chunk));
      }
      if (g_rank == 0) {
        for (int64_t i = 0; i < n; ++i) y[i] -= b0[i];
        const double res = norm2(y, n) / norm2(b0, n);
        printf("{\"event\":\"solve\",");
        print_common(o, mode);
        printf(",\"rep\":%d,\"nrhs\":%d,\"s\":%.6f,\"rel_residual\":%.3e}\n", s, o->nrhs, t, res);
        fflush(stdout);
      }
      free(x);
      free(y);
    }
    free(rhs);
    free(b0);
  }
  *ok = inertia_stable && (o->expect_neg < 0 || (long long)neg_first == o->expect_neg);
  run_job(&id, JOB_END, "termination");
  return neg_first;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  comm_size();
  const Options o = parse_options(argc, argv);
  double t_load = now();
  Matrix m = read_matrix(o.matrix, o.dist);
  int* perm = NULL;
  if (o.perm && g_rank == 0) perm = read_perm(o.perm, m.n);
  barrier();
  t_load = now() - t_load;

  char modes[64];
  snprintf(modes, sizeof modes, "%s", o.modes);
  double t_total = 0;
  int all_ok = 1;
  int neg_seen = -1;
  for (char* tok = strtok(modes, ","); tok; tok = strtok(NULL, ",")) {
    int ok = 0;
    const int neg = run_mode(&o, &m, perm, tok, &t_total, &ok);
    if (!ok || (neg_seen >= 0 && neg != neg_seen)) all_ok = 0;
    if (neg_seen < 0) neg_seen = neg;
  }

  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  double rss[2] = {ru.ru_maxrss / 1048576.0, ru.ru_maxrss / 1048576.0}; /* GiB: max, sum */
  reduce_max_d(&rss[0], 1);
  reduce_sum_d(&rss[1], 1);
  if (g_rank == 0) {
    printf("{\"event\":\"run\",\"label\":\"%s\",\"ranks\":%d,\"threads\":%d,\"dist\":%d,\"load_s\":%.3f,"
           "\"timed_s\":%.3f,\"rank_peak_gib_max\":%.3f,\"rank_peak_gib_sum\":%.3f,\"neg\":%d,"
           "\"expect_neg\":%lld,\"inertia_ok\":%s}\n",
           o.label, g_size, o.threads, o.dist, t_load, t_total, rss[0], rss[1], neg_seen, o.expect_neg,
           all_ok ? "true" : "false");
    fflush(stdout);
  }
  free(m.irn);
  free(m.jcn);
  free(m.a);
  free(perm);
  MPI_Finalize();
  return all_ok ? 0 : 1;
}
