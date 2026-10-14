// GPU module: CPU affinity of the engine's threads (affinity.hpp).
#include "gpu/affinity.hpp"

#include <omp.h>
#include <pthread.h>

#include <vector>

namespace hs::gpu {

bool openmp_places_cpuset(cpu_set_t* out) noexcept {
  CPU_ZERO(out);
  const int places = omp_get_num_places();
  if (places <= 0) return false;
  std::vector<int> ids;
  for (int p = 0; p < places; ++p) {
    const int n = omp_get_place_num_procs(p);
    if (n <= 0) continue;
    ids.resize(static_cast<std::size_t>(n));
    omp_get_place_proc_ids(p, ids.data());
    for (const int id : ids)
      if (id >= 0 && id < CPU_SETSIZE) CPU_SET(id, out);
  }
  return CPU_COUNT(out) > 0;
}

int thread_cpu_count() noexcept {
  cpu_set_t s;
  CPU_ZERO(&s);
  if (pthread_getaffinity_np(pthread_self(), sizeof s, &s) != 0) return 0;
  return CPU_COUNT(&s);
}

ScopedPlacesAffinity::ScopedPlacesAffinity() noexcept {
  CPU_ZERO(&saved_);
  cpu_set_t want;
  if (!openmp_places_cpuset(&want)) return;
  if (pthread_getaffinity_np(pthread_self(), sizeof saved_, &saved_) != 0) return;
  // Only widen: a thread bound to a place (a subset of the union). An unbound thread (OMP_PLACES
  // without binding) keeps its mask, which already holds every place.
  cpu_set_t common;
  CPU_AND(&common, &saved_, &want);
  if (!CPU_EQUAL(&common, &saved_) || CPU_EQUAL(&saved_, &want)) return;
  active_ = pthread_setaffinity_np(pthread_self(), sizeof want, &want) == 0;
}

ScopedPlacesAffinity::~ScopedPlacesAffinity() {
  if (active_) pthread_setaffinity_np(pthread_self(), sizeof saved_, &saved_);
}

}  // namespace hs::gpu
