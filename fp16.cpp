#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include <hip/hip_runtime.h>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using FP16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

std::vector<FP16> naive_cpu_matmul(std::vector<FP16> _A, std::vector<FP16> _B, std::vector<FP16> _C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += _A[i * K + k] * _B[k * N + j];
            }
            _C[i * N + j] = static_cast<FP16>(sum);
        }
    }
}

void zero_vec(std::vector<FP16> &_vec)
{
    for (int i = 0; i < _vec.size(); ++i)
    {
        _vec[i] = 0.0f;
    }
}

void check_matmul(const std::vector<FP16> &A, const std::vector<FP16> &B, const std::vector<FP16> &C, int M, int N, int K)
{
    printf("Validating...\n");

    std::vector<FP16> C_ans(M * N, 0);

#pragma omp parallel for num_threads(20)
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_ans[i * N + j] = static_cast<FP16>(sum);
        }
    }

    bool is_valid = true;
    int cnt = 0, thr = 10;
    float eps = 1e-3;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float c = C[i * N + j];
            float c_ans = C_ans[i * N + j];
            if (fabsf(c - c_ans) > eps &&
                (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps))
            {
                ++cnt;
                if (cnt <= thr)
                    printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j,
                           c_ans, c);
                if (cnt == thr + 1)
                    printf("Too many error, only first %d values are printed.\n", thr);
                is_valid = false;
            }
        }
    }

    if (is_valid)
    {
        printf("Result: VALID\n");
    }
    else
    {
        printf("Result: INVALID\n");
    }
}

struct DeviceMem
{
    DeviceMem() = delete;

    DeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void **>(&p_mem_), mem_size);
    }

    void *GetDeviceBuffer() { return p_mem_; }

    ~DeviceMem() { (void)hipFree(p_mem_); }

    void *p_mem_;
};

std::vector<FP16> fp16_fp16_fp32(
    const std::vector<FP16> &A,
    const std::vector<FP16> &B,
    std::vector<FP16> &C,
    int M, int N, int K)
{
    // assumption: all vector is row-major
    int stride_A = K;
    int stride_B = N;
    int stride_C = N;

    using ADataType = FP16;
    using BDataType = FP16;
    using AccDataType = float;
    using CDataType = FP16;

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::RowMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    using CElementOp = PassThrough;

    // Allocate memory on device
    DeviceMem a_buf(sizeof(ADataType) * M * K);
    DeviceMem b_buf(sizeof(BDataType) * K * N);
    DeviceMem c_buf(sizeof(CDataType) * M * N);

    hipMemcpy(
        a_buf.GetDeviceBuffer(), A.data(), sizeof(ADataType) * M * K, hipMemcpyHostToDevice);
    hipMemcpy(
        b_buf.GetDeviceBuffer(), B.data(), sizeof(BDataType) * K * N, hipMemcpyHostToDevice);

    // Create a GEMM object
    static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

    // clang-format off
    using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmXdl
    // ######|     AData|     BData|     CData|     AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
    // ######|      Type|      Type|      Type|        Type|        |        |        | Elementwise| Elementwise| Elementwise|Spacialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
    // ######|          |          |          |            |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
    // ######|          |          |          |            |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
             < ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,   GemmDefault,   256,   256,   128,     4,  8,   64,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,      true,               7,               1>;
    // // clang-format on

    auto gemm = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    auto argument = gemm.MakeArgument(
        static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
        static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(nullptr),
        static_cast<CDataType*>(c_buf.GetDeviceBuffer()),
        M,
        N,
        K,
        stride_A,
        stride_B,
        stride_C,
        AElementOp{},
        BElementOp{},
        CElementOp{});

    // Run the GEMM operation
    invoker.Run(argument, StreamConfig{nullptr, 0});
    _ = hipMemcpy(C.data(), c_buf.GetDeviceBuffer(), sizeof(CDataType) * M * N, hipMemcpyDeviceToHost);
    return C;
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:")) != -1) {
    switch (c) {
      case 'p':
        print_matrix = true;
        break;
      case 'v':
        validation = true;
        break;
      case 'n':
        num_iterations = atoi(optarg);
        break;
      case 'h':
      default:
        print_help(argv[0]);
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: M = atoi(argv[i]); break;
      case 1: N = atoi(argv[i]); break;
      case 2: K = atoi(argv[i]); break;
      default: break;
    }
  }
  printf("Options:\n");
  printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

static bool print_matrix = false;
static bool validation = false;
static int M = 8, N = 8, K = 8;
static int num_iterations = 1;

int main(int argc, char* argv[]) {
    parse_opt(argc, argv);

    float alpha = 1.0f;
    float beta = 0.0f;

    std::vector<FP16> A(M * K);
    std::vector<FP16> B(K * N);
    std::vector<FP16> D(M * N);

    // Initialize matrix
    for (int i = 0; i < M * K; ++i) {
        A[i] = static_cast<FP16>(rand()) / RAND_MAX - 0.5;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<FP16>(rand()) / RAND_MAX - 0.5;
    }

    float total_elapsed = 0.0f;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    for (int i = 0; i < num_iterations; ++i) {
        float elapsed = 0.0f;
        std::cout << "Calculate iteration " << i << std::endl;
        zero_vec(D);

        if (i != 0) // remove warm up
        {
            hipEventRecord(start, 0);
        }

        std::vector<FP16> E; //
        
        if (i != 0)
        {
            hipEventRecord(stop, 0);
            hipThreadSynchronize();
            hipEventElapsedTime(&elapsed, start, stop);
            total_elapsed += elapsed;
        }
    }

    // Verify the result compared to GPU
    if (validation) {
        check_matmul(A, B, E, M, N, K);
    }

    // Perf
    double elapsed_time_avg = total_elapsed / (num_iterations - 1);
    printf("Avg. time: %f sec\n", elapsed_time_avg);
    printf("Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);

    return 0;
}
