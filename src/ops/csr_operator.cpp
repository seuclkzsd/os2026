// CsrOperator: Op = A through the sparse kernels (docs/DESIGN.md §3.4b).
#include "hsolve/error.hpp"
#include "hsolve/operator.hpp"

namespace hs {

template <class T>
CsrOperator<T>::CsrOperator(const CsrMatrix<T>& a, KernelPath kernels) : a_(&a), kernels_(kernels) {
  if (!a.square())
    HS_THROW(InvalidInput,
             "CsrOperator: matrix is " << a.nrows << "x" << a.ncols << ", not square");
  resolve_kernel_path(kernels);  // an unavailable path fails here, not at the first apply
}

template <class T>
void CsrOperator<T>::do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) {
  spmm(*a_, x, ldx, y, ldy, p, T(1), T(0), KernelOptions{kernels_, &scratch_});
}

#define HS_INSTANTIATE_CSR_OPERATOR(T) template class CsrOperator<T>;
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_CSR_OPERATOR)
#undef HS_INSTANTIATE_CSR_OPERATOR

}  // namespace hs
