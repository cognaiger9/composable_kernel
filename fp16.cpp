#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>
#include <stdio.h>
#include <unistd.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"

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

void naive_cpu_matmul(const std::vector<FP16>& _A, const std::vector<FP16>& _B, std::vector<FP16>& _C, int M, int N, int K)
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
                sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * N + j]);
            }
            C_ans[i * N + j] = static_cast<FP16>(sum);
        }
    }

    // print random numbers
    std::cout << "c_ans = " << static_cast<float>(C_ans[0]) << "c res = " << static_cast<float>(C[0]) << std::endl;
    std::cout << "c_ans = " << static_cast<float>(C_ans[1]) << "c res = " << static_cast<float>(C[1]) << std::endl;

    bool is_valid = true;
    auto distance = std::distance(C.begin(), C.end());

    int cnt = 0, thr = M * N / 10;
    float eps = 1e-2;
    float err = 0.0f;

    #pragma omp parallel for num_threads(20)
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            FP16 c = C[i * N + j];
            FP16 c_ans = C_ans[i * N + j];
            float diff = std::abs(static_cast<float>(c) - static_cast<float>(c_ans));
            float denominator = (static_cast<float>(c_ans) + static_cast<float>(c)) * (static_cast<float>(c_ans) + static_cast<float>(c));
            err += (diff / denominator);
            if (diff > eps)
            {
                ++cnt;
                if (cnt <= thr)
                    printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j,
                           static_cast<float>(c_ans), static_cast<float>(c));
                if (cnt == thr + 1)
                {
                    printf("Too many error, only first %d values are printed.\n", thr);
                    is_valid = false;
                }
            }
        }
    }

    if (err < eps)
    {
        printf("Result: VALID\n");
    }
    else
    {
        printf("Result: INVALID, error = %f, threshold = %f\n", err, eps);
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

void fp16_fp16_fp32(
    const std::vector<FP16> &A,
    const std::vector<FP16> &B,
    std::vector<FP16> &C,
    int M, int N, int K, int iter, float& total_elapsed)
{
    // assumption: all vector is row-major
    int stride_A = K;
    int stride_B = N;
    int stride_C = N;

    using ADataType = FP16;
    using BDataType = FP16;
    using AccDataType = float;
    using CDataType = FP16;
    using CShuffleDataType = FP16;

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
    using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle
    // ######| ALayout| BLayout| CLayout|     AData|     BData|     CData|     AccData|         CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
    // ######|        |        |        |      Type|      Type|      Type|        Type|         DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
    // ######|        |        |        |          |          |          |            |                 |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
    // ######|        |        |        |          |          |          |            |                 |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
             < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  CElementOp,    GemmDefault,        1 ,   256,  128,   128,    32,   8,   2,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              4,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              2,         0,           1,           2,              S<1, 16, 1, 16>,               8, ck::LoopScheduler::Interwave, ck::PipelineVersion::v1>;
    // clang-format on

    auto gemm = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    auto argument = gemm.MakeArgument(
        static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
        static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
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

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Run the GEMM operation
    for (int i = 0; i < iter; i++)
    {
        float elapsed = 0.0f;
        std::cout << "Calculate iteration " << i << std::endl;
        //zero_vec(D);

        if (i != 0) // remove warm up
        {
            hipEventRecord(start, 0);
        }

        invoker.Run(argument, StreamConfig{nullptr, 0});
        
        if (i != 0)
        {
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&elapsed, start, stop);
            total_elapsed += elapsed;
        }
    }

    hipMemcpy(C.data(), c_buf.GetDeviceBuffer(), sizeof(CDataType) * M * N, hipMemcpyDeviceToHost);
}

static bool print_matrix = false;
static bool validation = false;
static int M = 8, N = 8, K = 8;
static int num_iterations = 1;

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
        //print_help(argv[0]);
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

int main(int argc, char* argv[]) {
    parse_opt(argc, argv);

    std::vector<FP16> A(M * K);
    std::vector<FP16> B(K * N);
    std::vector<FP16> C(M * N);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the range of the floating-point numbers
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Initialize matrix
    for (int i = 0; i < M * K; ++i) {
        A[i] = static_cast<FP16>(dist(gen));
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<FP16>(dist(gen));
    }

    float total_elapsed = 0.0f;
    fp16_fp16_fp32(A, B, C, M, N, K, num_iterations, total_elapsed);

    // Verify the result compared to GPU
    if (validation) {
        check_matmul(A, B, C, M, N, K);
    }

    // Perf
    double elapsed_time_avg = total_elapsed / (num_iterations - 1);
    printf("Avg. time: %f sec\n", elapsed_time_avg);
    printf("Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);

    return 0;
}
