// Re-evaluates the auto choice of the completeness check (hs::certificate_cost, verify.hpp) of one
// solve with other measured times - e.g. a 16-thread run's factorization and solve times - for
// the module certint's "what would auto pick at 16 threads" estimate.
//
//   bench_certint_model --event LOG.jsonl [--factorize-s F --solve-call-s C [--rhs-per-call Q]]
//                       [--cmp results/cmp-p16-best.json --case NAME [--solver hsolve-nocert]]
//                       [--rhs-ratio R] [--stream-bw B]
//
// LOG.jsonl: the driver's event log of an auto (or randomized) run (bench_certint_e2e --log, or the
// CLI's --log); its first "certificate_choice" event carries every input of certificate_cost
// (in_* fields): problem size, deflated pairs, eps of the randomized check, kappa, the single-count
// rule's verdict, the analysis model, and the run's own measured times. The window and so eps,
// the deflation and the rule do not depend on the thread count; the times do. They are replaced
// by --factorize-s (one numeric factorization) and --solve-call-s (one solve call with
// --rhs-per-call right-hand sides, default: the logged run's block width), or taken from a
// tools/bench/compare.py result (--cmp: the solver's median phases, factorize = numeric_s /
// factorizations, solve call = solve_s / (solves / rhs_per_call), solves = right-hand sides).
// The analysis model is thread-independent (MUMPS' analysis is sequential) and stays.
// Prints one JSON line: the logged prediction, the inputs used and the re-evaluated prediction.
// Run it on a cluster node like every bench; it is O(1) work, so the build/test lane (default
// cores, not the measurement slot) serves:
//   python3 tools/run/guard.py --skip-busy-check --mem-gb 1 --timeout 60 --
//     build/release/bench_certint_model --event E.jsonl --cmp results/cmp-p16-best.json --case C
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "hsolve/verify.hpp"

namespace {

using json = nlohmann::json;

json first_choice(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("certificate_choice") == std::string::npos) continue;
    const json j = json::parse(line, nullptr, false);
    if (!j.is_discarded() && j.value("event", "") == "certificate_choice" && j.contains("in_n")) return j;
  }
  throw std::runtime_error(path + ": no certificate_choice event with inputs (an auto or randomized run of "
                                  "this module's driver is needed)");
}

hs::CertificateCostInputs inputs_from(const json& e) {
  hs::CertificateCostInputs in;
  in.n = e.at("in_n").get<hs::index_t>();
  in.deflated = e.at("in_deflated").get<hs::index_t>();
  in.complex_scalars = e.at("in_complex").get<bool>();
  in.factorize_s = e.at("in_factorize_s").get<double>();
  in.counter_analysis_s = e.at("in_counter_analysis_s").get<double>();
  in.single_analysis_s = e.at("in_single_analysis_s").get<double>();
  in.count_ratio = e.at("in_count_ratio").get<double>();
  in.single_ratio = e.at("in_single_ratio").get<double>();
  in.single_valid = e.at("in_single_valid").get<bool>();
  in.apply1_s = e.at("in_apply1_s").get<double>();
  in.rhs_ratio = e.at("in_rhs_ratio").get<double>();
  in.stream_bytes_per_s = e.at("in_stream_bytes_per_s").get<double>();
  in.epsilon = e.at("in_epsilon").get<double>();
  in.delta = e.at("in_delta").get<double>();
  in.kappa = e.at("in_kappa").get<double>();
  in.max_krylov_vectors = e.at("in_max_krylov_vectors").get<hs::index_t>();
  in.max_memory_bytes = e.at("in_max_memory_bytes").get<double>();
  in.preference = e.at("in_preference").get<double>();
  return in;
}

json estimate_json(const hs::CertificateCostEstimate& e) {
  auto fin = [](double x) { return std::isfinite(x) ? json(x) : json(nullptr); };
  return {{"exact_s", fin(e.exact_s)},
          {"exact_single_s", fin(e.exact_single_s)},
          {"randomized_s", fin(e.randomized_s)},
          {"block_size", e.block_size},
          {"block_steps", e.block_steps},
          {"failure_bound", e.failure_bound},
          {"recommended", hs::certificate_method_name(e.recommended)}};
}

}  // namespace

int main(int argc, char** argv) {
  std::string event_path, cmp_path, case_name, solver = "hsolve-nocert";
  double factorize_s = -1.0, solve_call_s = -1.0, rhs_per_call = -1.0, rhs_ratio = -1.0, stream_bw = -1.0;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      auto need = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
        return argv[++i];
      };
      if (k == "--event") event_path = need();
      else if (k == "--cmp") cmp_path = need();
      else if (k == "--case") case_name = need();
      else if (k == "--solver") solver = need();
      else if (k == "--factorize-s") factorize_s = std::stod(need());
      else if (k == "--solve-call-s") solve_call_s = std::stod(need());
      else if (k == "--rhs-per-call") rhs_per_call = std::stod(need());
      else if (k == "--rhs-ratio") rhs_ratio = std::stod(need());
      else if (k == "--stream-bw") stream_bw = std::stod(need());
      else throw std::runtime_error("unknown argument " + k);
    }
    if (event_path.empty() || (cmp_path.empty() != case_name.empty()))
      throw std::runtime_error(
          "usage: bench_certint_model --event LOG.jsonl [--factorize-s F --solve-call-s C [--rhs-per-call Q]] "
          "[--cmp FILE --case NAME [--solver S]] [--rhs-ratio R] [--stream-bw B]");
    const json ev = first_choice(event_path);
    hs::CertificateCostInputs in = inputs_from(ev);
    const hs::CertificateCostEstimate logged = hs::certificate_cost(in);
    if (rhs_per_call <= 0.0) rhs_per_call = std::max(1.0, ev.value("rhs_per_call", 1.0));
    json source = {{"event", event_path}};
    if (!cmp_path.empty()) {
      std::ifstream f(cmp_path);
      if (!f) throw std::runtime_error("cannot read " + cmp_path);
      const json cmp = json::parse(f);
      const json* row = nullptr;
      for (const json& r : cmp.at("rows"))
        if (r.value("case", "") == case_name) row = &r;
      if (!row) throw std::runtime_error(cmp_path + ": no case " + case_name);
      const json& ph = row->at("solvers").at(solver).at("phases");
      factorize_s = ph.at("numeric_s").get<double>() / ph.at("factorizations").get<double>();
      const double calls = ph.at("solves").get<double>() / rhs_per_call;
      solve_call_s = ph.at("solve_s").get<double>() / calls;
      source["cmp"] = cmp_path;
      source["case"] = case_name;
      source["solver"] = solver;
      source["cores"] = cmp.value("ncores", 0);
    }
    if (rhs_ratio >= 0.0) in.rhs_ratio = rhs_ratio;
    if (stream_bw > 0.0) in.stream_bytes_per_s = stream_bw;
    if (factorize_s > 0.0) in.factorize_s = factorize_s;
    if (solve_call_s > 0.0) in.apply1_s = solve_call_s / (1.0 + in.rhs_ratio * (rhs_per_call - 1.0));
    const hs::CertificateCostEstimate now = hs::certificate_cost(in);
    json out = {{"exp", "certint_model"},
                {"source", source},
                {"logged", estimate_json(logged)},
                {"inputs",
                 {{"n", in.n},
                  {"deflated", in.deflated},
                  {"factorize_s", in.factorize_s},
                  {"apply1_s", in.apply1_s},
                  {"rhs_per_call", rhs_per_call},
                  {"counter_analysis_s", in.counter_analysis_s},
                  {"single_analysis_s", in.single_analysis_s},
                  {"single_valid", in.single_valid},
                  {"epsilon", in.epsilon},
                  {"kappa", in.kappa},
                  {"delta", in.delta}}},
                {"estimate", estimate_json(now)}};
    std::cout << out.dump() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
