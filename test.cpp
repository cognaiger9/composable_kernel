#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/utility/literals.hpp"

#include <hip/hip_runtime.h>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using I8  = std::int8_t;
using I16  = std::int16_t;
using I32 = std::int32_t;
using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

struct DeviceMem
{
    DeviceMem() = delete;

    DeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~DeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

struct LimitInt8Range {

    LimitInt8Range(F32 alpha, F32 beta) : alpha(alpha), beta(beta){};

    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D& d) const;
    
    template <>
    __host__ __device__ constexpr void
    operator()<I8, I32, I8>(I8& e, const I32& c, const I8& d) const
    {
        const F32 c_scale = ck::type_convert<F32>(c) * alpha;
        const F32 d_scale = ck::type_convert<F32>(d) * beta;
        F32 temp = c_scale + d_scale;
        temp = temp < -127 ? -127 : temp;
        temp = temp > 127 ? 127 : temp;
        e = ck::type_convert<I8>(temp);
    }
        
    F32 alpha;
    F32 beta;
};

// Function input and output using std::vector
std::vector<I8> linear_relu_abde_i8(
  const std::vector<I8>& A, 
  const std::vector<I8>& B, 
  const std::vector<I8>& D, 
  int M, int N, int K, 
  float alpha, 
  float beta)
{
    int batch_count = 1;

    int stride_A = K;
    // int stride_B = K;
    int stride_B = N;
    int stride_D0 = N;
    int stride_E = N;

    int batch_stride_A = M * K;
    int batch_stride_B = K * N;
    int batch_stride_D0 = M * N;
    int batch_stride_E = M * N;

    std::cout << std::endl;
    std::cout << "batch_count: " << batch_count << std::endl;
    std::cout << "stride_A: " << stride_A << std::endl;
    std::cout << "stride_B: " << stride_B << std::endl;
    std::cout << "stride_D0: " << stride_D0 << std::endl;
    std::cout << "stride_E: " << stride_E << std::endl;

    std::cout << "batch_stride_A: " << batch_stride_A << std::endl;
    std::cout << "batch_stride_B: " << batch_stride_B << std::endl;
    std::cout << "batch_stride_D0: " << batch_stride_D0 << std::endl;
    std::cout << "batch_stride_E: " << batch_stride_E << std::endl;

    std::vector<I8> E(M * N);

    using ADataType        = I8;
    using BDataType        = I8;
    using AccDataType      = I32;
    using CShuffleDataType = I32;
    using D0DataType       = I8;
    using DsDataType       = ck::Tuple<D0DataType>;
    using EDataType        = I8;

    using ALayout  = ck::tensor_layout::gemm::RowMajor;
    using BLayout  = ck::tensor_layout::gemm::RowMajor;
    using D0Layout = ck::tensor_layout::gemm::RowMajor;
    using DsLayout = ck::Tuple<D0Layout>;
    using ELayout  = ck::tensor_layout::gemm::RowMajor;

    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;
    using CDEElementOp = LimitInt8Range;
    
    // Allocate GPU memory
    DeviceMem a_device_buf(sizeof(ADataType) * batch_count * M * K);
    DeviceMem b_device_buf(sizeof(BDataType) * batch_count * K * N);
    DeviceMem d_device_buf(sizeof(D0DataType) * batch_count * M * N);
    DeviceMem e_device_buf(sizeof(EDataType) * batch_count * M * N);

    // Copy data from host to device
    auto _ = hipMemcpy(a_device_buf.GetDeviceBuffer(), A.data(), sizeof(ADataType) * M * K, hipMemcpyHostToDevice);
    _ = hipMemcpy(b_device_buf.GetDeviceBuffer(), B.data(), sizeof(BDataType) * K * N, hipMemcpyHostToDevice);
    _ = hipMemcpy(d_device_buf.GetDeviceBuffer(), D.data(), sizeof(D0DataType) * M * N, hipMemcpyHostToDevice);

    static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

    using DeviceOpInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl<
        ALayout, BLayout, DsLayout, ELayout,
        ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,
        AElementOp, BElementOp, CDEElementOp,
        GemmDefault,1,   256,   256,   128,    64,  16,  16,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,              16>;
    auto device_op = DeviceOpInstance{};
    auto invoker = device_op.MakeInvoker();

    auto argument = device_op.MakeArgument(
        a_device_buf.GetDeviceBuffer(),
        b_device_buf.GetDeviceBuffer(),
        {d_device_buf.GetDeviceBuffer()},
        e_device_buf.GetDeviceBuffer(),
        M, N, K,
        batch_count,
        stride_A, stride_B, {stride_D0}, stride_E,
        batch_stride_A, batch_stride_B, {batch_stride_D0}, batch_stride_E,
        AElementOp{}, BElementOp{}, CDEElementOp{alpha, beta});

    // Run the GEMM operation
    invoker.Run(argument, StreamConfig{nullptr, 0});
    _ = hipMemcpy(E.data(), e_device_buf.GetDeviceBuffer(), sizeof(EDataType) * batch_count * M * N, hipMemcpyDeviceToHost);
    return E;
}

int main() {
    int M = 4;
    int N = 4;
    int K = 4;
    float alpha = 1.0f;
    float beta = 0.0f;

    std::vector<I8> A(M * K);
    std::vector<I8> B(K * N);
    std::vector<I8> D(M * N);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-5, 5);

    for (int i = 0; i < M * K; ++i) {
        A[i] = static_cast<I8>(1); // static_cast<I8>(dist(rng));
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<I8>(1); // static_cast<I8>(dist(rng));
    }
    for (int i = 0; i < M * N; ++i) {
        D[i] = static_cast<I8>(0); // static_cast<I8>(dist(rng));
    }

    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << static_cast<int>(A[i * K + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nMatrix B:" << std::endl;
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << static_cast<int>(B[i * N + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nMatrix D:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << static_cast<int>(D[i * N + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::vector<I8> E = linear_relu_abde_i8(A, B, D, M, N, K, alpha, beta);

    std::cout << "\nResult of Matrix E:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << static_cast<int>(E[i * N + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
