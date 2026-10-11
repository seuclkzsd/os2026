// JSON serialization of the driver's public structs: SolverConfig (config.json, both ways) and
// SolveReport (report.json). Schemas: docs/FORMATS.md §2 and §3. Key names equal field names.
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "hsolve/error.hpp"
#include "hsolve/solver.hpp"

namespace hs {

namespace {

using json = nlohmann::json;

template <class E>
struct EnumNames;

template <>
struct EnumNames<TargetMode> {
  static constexpr std::pair<TargetMode, const char*> kList[] = {
      {TargetMode::Nearest, "nearest"},
      {TargetMode::Interval, "interval"},
      {TargetMode::Smallest, "smallest"},
      {TargetMode::Largest, "largest"}};
};
template <>
struct EnumNames<ProblemType> {
  static constexpr std::pair<ProblemType, const char*> kList[] = {
      {ProblemType::Auto, "auto"},
      {ProblemType::Standard, "standard"},
      {ProblemType::GeneralizedDefinite, "generalized_definite"},
      {ProblemType::GeneralizedIndefinite, "generalized_indefinite"}};
};
template <>
struct EnumNames<Method> {
  static constexpr std::pair<Method, const char*> kList[] = {
      {Method::Auto, "auto"},
      {Method::ShiftInvertLanczos, "shift_invert_lanczos"},
      {Method::ShiftInvertKrylovSchur, "shift_invert_krylov_schur"},
      {Method::Lanczos, "lanczos"}};
};
template <>
struct EnumNames<FactorPrecision> {
  static constexpr std::pair<FactorPrecision, const char*> kList[] = {
      {FactorPrecision::Double, "double"}, {FactorPrecision::Single, "single"}};
};
template <>
struct EnumNames<Toggle> {
  static constexpr std::pair<Toggle, const char*> kList[] = {
      {Toggle::Auto, "auto"}, {Toggle::On, "on"}, {Toggle::Off, "off"}};
};

template <class E>
const char* enum_name(E e) {
  for (const auto& [v, s] : EnumNames<E>::kList)
    if (v == e) return s;
  return "unknown";
}

template <class E>
E enum_parse(const json& j, const char* key) {
  if (!j.is_string()) HS_THROW(InvalidInput, "config: '" << key << "' must be a string");
  const auto s = j.get<std::string>();
  for (const auto& [v, name] : EnumNames<E>::kList)
    if (s == name) return v;
  std::string allowed;
  for (const auto& [v, name] : EnumNames<E>::kList) allowed += std::string(allowed.empty() ? "" : ", ") + name;
  HS_THROW(InvalidInput, "config: '" << key << "' = \"" << s << "\" is not one of {" << allowed << "}");
}

double get_number(const json& j, const char* key) {
  if (!j.is_number()) HS_THROW(InvalidInput, "config: '" << key << "' must be a number");
  return j.get<double>();
}

template <class I>
I get_integer(const json& j, const char* key) {
  if (!j.is_number_integer() && !j.is_number_unsigned())
    HS_THROW(InvalidInput, "config: '" << key << "' must be an integer");
  return j.get<I>();
}

// JSON has no infinities; the certificate uses +-inf for one-sided windows.
json finite_or_null(double x) { return std::isfinite(x) ? json(x) : json(nullptr); }

json factor_options_json(const FactorOptions& o) {
  return {{"precision", enum_name(o.precision)}, {"ordering", o.ordering},       {"scaling", o.scaling},
          {"threads", o.threads},                {"mem_relax_pct", o.mem_relax_pct},
          {"max_memory_mb", o.max_memory_mb},    {"blr", o.blr},                 {"blr_tol", o.blr_tol},
          {"blr_estimates", o.blr_estimates},    {"out_of_core", o.out_of_core}, {"ooc_dir", o.ooc_dir},
          {"null_pivot_tol", o.null_pivot_tol},  {"verbosity", o.verbosity},
          {"deterministic", o.deterministic}};
}

json factor_stats_json(const FactorStats& s) {
  return {{"analyze_s", s.analyze_s},
          {"factorize_s", s.factorize_s},
          {"solve_s", s.solve_s},
          {"factorizations", s.factorizations},
          {"solves", s.solves},
          {"rhs_solved", s.rhs_solved},
          {"estimate",
           {{"factor_entries", s.estimate.factor_entries},
            {"incore_mb", s.estimate.incore_mb},
            {"ooc_mb", s.estimate.ooc_mb},
            {"blr_incore_mb", s.estimate.blr_incore_mb},
            {"blr_ooc_mb", s.estimate.blr_ooc_mb},
            {"flops", s.estimate.flops}}},
          {"factor_entries", s.factor_entries},
          {"factor_mb_used", s.factor_mb_used},
          {"null_pivots", s.null_pivots},
          {"delayed_pivots", s.delayed_pivots},
          {"workspace_retries", s.workspace_retries},
          {"last_info1", s.last_info1},
          {"last_info2", s.last_info2}};
}

}  // namespace

SolverConfig parse_config(const std::string& json_text, const SolverConfig& base) {
  json doc;
  try {
    doc = json::parse(json_text);
  } catch (const json::parse_error& e) {
    HS_THROW(InvalidInput, "config: invalid JSON: " << e.what());
  }
  if (!doc.is_object()) HS_THROW(InvalidInput, "config: top level must be a JSON object");
  SolverConfig c = base;
  for (const auto& [key, v] : doc.items()) {
    const char* k = key.c_str();
    if (key == "target") {
      if (!v.is_object()) HS_THROW(InvalidInput, "config: 'target' must be an object");
      for (const auto& [tk, tv] : v.items()) {
        const std::string full = "target." + tk;
        if (tk == "mode") c.target.mode = enum_parse<TargetMode>(tv, full.c_str());
        else if (tk == "sigma") c.target.sigma = get_number(tv, full.c_str());
        else if (tk == "k") c.target.k = get_integer<index_t>(tv, full.c_str());
        else if (tk == "lower") c.target.lower = get_number(tv, full.c_str());
        else if (tk == "upper") c.target.upper = get_number(tv, full.c_str());
        else HS_THROW(InvalidInput, "config: unknown key '" << full << "'");
      }
    } else if (key == "problem") c.problem = enum_parse<ProblemType>(v, k);
    else if (key == "method") c.method = enum_parse<Method>(v, k);
    else if (key == "tol") c.tol = get_number(v, k);
    else if (key == "inner_tol_factor") c.inner_tol_factor = get_number(v, k);
    else if (key == "hermitian_tol") c.hermitian_tol = get_number(v, k);
    else if (key == "ncv") c.ncv = get_integer<index_t>(v, k);
    else if (key == "slice_size") c.slice_size = get_integer<index_t>(v, k);
    else if (key == "max_restarts") c.max_restarts = get_integer<index_t>(v, k);
    else if (key == "seed") c.seed = get_integer<std::uint64_t>(v, k);
    else if (key == "threads") c.threads = get_integer<int>(v, k);
    else if (key == "memory_budget_gb") c.memory_budget_gb = get_number(v, k);
    else if (key == "factor_precision") c.factor_precision = enum_parse<FactorPrecision>(v, k);
    else if (key == "ordering") c.ordering = get_integer<int>(v, k);
    else if (key == "deterministic") {
      if (!v.is_boolean()) HS_THROW(InvalidInput, "config: 'deterministic' must be true/false");
      c.deterministic = v.get<bool>();
    }
    else if (key == "blr") c.blr = enum_parse<Toggle>(v, k);
    else if (key == "out_of_core") c.out_of_core = enum_parse<Toggle>(v, k);
    else if (key == "inertia_check") c.inertia_check = enum_parse<Toggle>(v, k);
    else if (key == "zero_filter_tol") c.zero_filter_tol = get_number(v, k);
    else if (key == "complex_tol") c.complex_tol = get_number(v, k);
    else if (key == "max_shift_retries") c.max_shift_retries = get_integer<index_t>(v, k);
    else HS_THROW(InvalidInput, "config: unknown key '" << key << "'");
  }
  // Semantic validation (independent of the problem).
  if (c.target.k < 1 && c.target.mode != TargetMode::Interval) HS_THROW(InvalidInput, "config: target.k must be >= 1");
  if (c.target.mode == TargetMode::Interval && !(c.target.lower < c.target.upper))
    HS_THROW(InvalidInput, "config: interval needs target.lower < target.upper");
  if (!(c.tol > 0.0) || !(c.tol < 1.0)) HS_THROW(InvalidInput, "config: tol must be in (0, 1)");
  if (!(c.inner_tol_factor > 0.0) || c.inner_tol_factor > 1.0)
    HS_THROW(InvalidInput, "config: inner_tol_factor must be in (0, 1]");
  if (c.ncv < 0 || c.slice_size < 0 || c.max_restarts < 1 || c.threads < 0 || c.max_shift_retries < 0)
    HS_THROW(InvalidInput, "config: ncv, slice_size, threads, max_shift_retries must be >= 0 and max_restarts >= 1");
  if (!(c.memory_budget_gb > 0.0)) HS_THROW(InvalidInput, "config: memory_budget_gb must be > 0");
  if (c.ordering < -1 || c.ordering > 7) HS_THROW(InvalidInput, "config: ordering must be -1 (adaptive) or a MUMPS ICNTL(7) value 0..7");
  return c;
}

std::string config_to_json(const SolverConfig& c, int indent) {
  json j = {{"target",
             {{"mode", enum_name(c.target.mode)},
              {"sigma", c.target.sigma},
              {"k", c.target.k},
              {"lower", c.target.lower},
              {"upper", c.target.upper}}},
            {"problem", enum_name(c.problem)},
            {"method", enum_name(c.method)},
            {"tol", c.tol},
            {"inner_tol_factor", c.inner_tol_factor},
            {"hermitian_tol", c.hermitian_tol},
            {"ncv", c.ncv},
            {"slice_size", c.slice_size},
            {"max_restarts", c.max_restarts},
            {"seed", c.seed},
            {"threads", c.threads},
            {"memory_budget_gb", c.memory_budget_gb},
            {"factor_precision", enum_name(c.factor_precision)},
            {"ordering", c.ordering},
            {"deterministic", c.deterministic},
            {"blr", enum_name(c.blr)},
            {"out_of_core", enum_name(c.out_of_core)},
            {"inertia_check", enum_name(c.inertia_check)},
            {"zero_filter_tol", c.zero_filter_tol},
            {"complex_tol", c.complex_tol},
            {"max_shift_retries", c.max_shift_retries}};
  return j.dump(indent);
}

std::string SolveReport::to_json(int indent) const {
  json pairs_j = json::array();
  for (const auto& p : pairs) {
    pairs_j.push_back({{"lambda_re", p.lambda_re},
                       {"lambda_im", p.lambda_im},
                       {"abs_residual", p.abs_residual},
                       {"rel_residual", p.rel_residual},
                       {"backward_error", p.backward_error},
                       {"accepted", p.accepted}});
  }
  json slices_j = json::array();
  for (const auto& sl : slices) {
    slices_j.push_back({{"lower", finite_or_null(sl.lower)},
                        {"upper", finite_or_null(sl.upper)},
                        {"sigma", sl.sigma},
                        {"which", sl.which},
                        {"count_inertia", sl.count_inertia},
                        {"count_found", sl.count_found},
                        {"nev", sl.nev},
                        {"reruns", sl.reruns},
                        {"restarts", sl.restarts},
                        {"op_applies", sl.op_applies},
                        {"passed", sl.passed}});
  }
  json j = {{"schema_version", 1},
            {"n", n},
            {"nnz_a", nnz_a},
            {"nnz_b", nnz_b},
            {"complex_scalars", complex_scalars},
            {"problem", enum_name(problem)},
            {"norm2_a", norm2_a},
            {"norm2_b", norm2_b},
            {"scale", scale},
            {"hermitian_defect_a", hermitian_defect_a},
            {"hermitian_defect_b", hermitian_defect_b},
            {"method", enum_name(method)},
            {"sigma_requested", sigma_requested},
            {"sigma_used", sigma_used},
            {"shift_retries", shift_retries},
            {"factor_options", factor_options_json(factor_options)},
            {"factor_stats", factor_stats_json(factor_stats)},
            {"restarts", restarts},
            {"op_applies", op_applies},
            {"time_total_s", time_total_s},
            {"time_setup_s", time_setup_s},
            {"time_factor_s", time_factor_s},
            {"time_iterate_s", time_iterate_s},
            {"time_verify_s", time_verify_s},
            {"peak_rss_bytes", peak_rss_bytes},
            {"pairs", pairs_j},
            {"certificate",
             {{"performed", certificate.performed},
              {"lower", finite_or_null(certificate.lower)},
              {"upper", finite_or_null(certificate.upper)},
              {"count_inertia", certificate.count_inertia},
              {"count_found", certificate.count_found},
              {"endpoint_moves", certificate.endpoint_moves},
              {"passed", certificate.passed},
              {"note", certificate.note}}},
            {"slices", slices_j},
            {"zero_cluster_dim", zero_cluster_dim},
            {"zero_modes_filtered", zero_modes_filtered},
            {"success", success},
            {"warnings", warnings}};
  return j.dump(indent);
}

}  // namespace hs
