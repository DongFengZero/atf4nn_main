/*
 * benchmark.cu  —  Nsight Compute profiling benchmark
 * =====================================================
 * Three composite GPU operators, no external libraries.
 *
 * Interface:
 *   ./benchmark --op flash_attn  --tile <S> --reps <N>
 *   ./benchmark --op cbr         --tile <S> --reps <N>
 *   ./benchmark --op layer_norm  --tile <S> --reps <N>
 *
 *   --tile : fast-memory tile size in cache lines (64..384)
 *            1 cache line = 128 bytes  →  tile_bytes = S * 128
 *
 * Tile → dimension mapping
 * ─────────────────────────
 *   flash_attn  : head_dim D=64 (fixed)
 *                 block_rows Br = tile_bytes / (D * sizeof(half))
 *                 seq_len    L  = Br * 8
 *                 tensors: Q[Br,D]  K[L,D]  V[L,D]  S[Br,L]  O[Br,D]
 *
 *   cbr         : channels C = tile_bytes / (H*W*sizeof(float))
 *                 spatial H=W=8
 *                 tensors: feat[C,H,W]  weight[C,9]
 *
 *   layer_norm  : rows  R = tile_bytes / sizeof(float)  (aligned 32)
 *                 cols  C = R * 8  (hidden dim = 8x tile rows)
 *                 tensors: X[R,C]  gamma[C]  beta[C]  Y[R,C]
 *                 3 kernels: row_mean -> row_var -> norm_apply
 *                 Total I/O ~= 4*R*C floats  proportional to S^2 (monotone)
 *
 * Build:
 *   make
 *   nvcc -O2 -arch=sm_80 -o benchmark benchmark.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            exit(1);                                                          \
        }                                                                     \
    } while(0)

static inline size_t tile_bytes(int S) { return (size_t)S * 128; }

// ---------------------------------------------------------------------------
// Tiled fp16 GEMM  —  used by flash_attn only
// ---------------------------------------------------------------------------
#define TK 16

template<int BM, int BN>
__global__ void tiled_gemm_h(const __half* __restrict__ A,
                              const __half* __restrict__ B,
                              __half*       __restrict__ C,
                              int M, int N, int K, float scale)
{
    __shared__ float sA[BM][TK];
    __shared__ float sB[TK][BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (K + TK - 1) / TK; ++t) {
        int k0 = t * TK;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && k0 + (int)threadIdx.x < K)
            ? __half2float(A[row * K + k0 + threadIdx.x]) : 0.f;
        sB[threadIdx.y][threadIdx.x] =
            (k0 + (int)threadIdx.y < K && col < N)
            ? __half2float(B[(k0 + threadIdx.y) * N + col]) : 0.f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TK; ++i) acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = __float2half(acc * scale);
}

// ---------------------------------------------------------------------------
// Row-wise softmax (fp16, in-place)  —  used by flash_attn only
// ---------------------------------------------------------------------------
__global__ void row_softmax_h(__half* __restrict__ X, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    __half* x = X + (size_t)row * N;
    float mx = __half2float(x[0]);
    for (int j = 1; j < N; ++j) mx = fmaxf(mx, __half2float(x[j]));
    float s = 0.f;
    for (int j = 0; j < N; ++j) s += expf(__half2float(x[j]) - mx);
    float inv = 1.f / s;
    for (int j = 0; j < N; ++j)
        x[j] = __float2half(expf(__half2float(x[j]) - mx) * inv);
}

// ===========================================================================
// OPERATOR 1: flash_attn  (3 kernels)
// ===========================================================================
namespace flash_attn_op {

void run(int S, int reps)
{
    const int D = 64;
    int Br = (int)(tile_bytes(S) / (D * sizeof(__half)));
    if (Br < TK) Br = TK;
    Br = (Br / TK) * TK;
    int L  = Br * 8;

    size_t q_sz  = (size_t)Br * D * sizeof(__half);
    size_t kv_sz = (size_t)L  * D * sizeof(__half);
    size_t s_sz  = (size_t)Br * L * sizeof(__half);

    __half *d_Q, *d_K, *d_V, *d_S, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, q_sz));
    CUDA_CHECK(cudaMalloc(&d_K, kv_sz));
    CUDA_CHECK(cudaMalloc(&d_V, kv_sz));
    CUDA_CHECK(cudaMalloc(&d_S, s_sz));
    CUDA_CHECK(cudaMalloc(&d_O, q_sz));
    CUDA_CHECK(cudaMemset(d_Q, 1, q_sz));
    CUDA_CHECK(cudaMemset(d_K, 1, kv_sz));
    CUDA_CHECK(cudaMemset(d_V, 1, kv_sz));

    float inv_sqrt_D = 1.f / sqrtf((float)D);
    dim3  blk(TK, TK);
    dim3  grd_qk((L +TK-1)/TK, (Br+TK-1)/TK);
    dim3  grd_pv((D +TK-1)/TK, (Br+TK-1)/TK);
    int   sm_grd = (Br + 255) / 256;

    for (int r = 0; r < reps; ++r) {
        tiled_gemm_h<TK,TK><<<grd_qk,blk>>>(d_Q, d_K, d_S, Br, L,  D, inv_sqrt_D);
        row_softmax_h       <<<sm_grd,256>>>(d_S, Br, L);
        tiled_gemm_h<TK,TK><<<grd_pv,blk>>>(d_S, d_V, d_O, Br, D, L, 1.f);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S); cudaFree(d_O);
    fprintf(stderr, "[flash_attn]  tile=%d  D=%d  Br=%d  L=%d  "
                    "Q=%.0fKB  S=%.0fKB\n",
            S, D, Br, L, q_sz/1024.f, s_sz/1024.f);
}
} // namespace flash_attn_op


// ===========================================================================
// OPERATOR 2: cbr  (3 kernels: depthwise conv3x3 -> batchnorm -> relu)
// ===========================================================================
namespace cbr_op {

__global__ void depthwise_conv3x3(const float* __restrict__ X,
                                   const float* __restrict__ W,
                                   float*       __restrict__ Y,
                                   int C, int H, int Wd)
{
    int c = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || h >= H || w >= Wd) return;
    float acc = 0.f;
    for (int kh = 0; kh < 3; ++kh)
        for (int kw = 0; kw < 3; ++kw) {
            int ih = h+kh-1, iw = w+kw-1;
            acc += (ih>=0 && ih<H && iw>=0 && iw<Wd)
                   ? X[c*H*Wd + ih*Wd + iw] * W[c*9 + kh*3 + kw] : 0.f;
        }
    Y[c*H*Wd + h*Wd + w] = acc;
}

__global__ void batchnorm(float* __restrict__ X,
                           const float* __restrict__ gamma,
                           const float* __restrict__ beta,
                           int C, int HW)
{
    int c = blockIdx.x;
    if (c >= C) return;
    float* xc = X + c*HW;
    float mean=0.f, var=0.f;
    for (int i=0;i<HW;++i) mean+=xc[i]; mean/=HW;
    for (int i=0;i<HW;++i){float d=xc[i]-mean; var+=d*d;}
    float inv = rsqrtf(var/HW + 1e-5f) * gamma[c];
    for (int i=0;i<HW;++i) xc[i]=(xc[i]-mean)*inv+beta[c];
}

__global__ void relu_inplace(float* __restrict__ X, int N)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) X[i] = fmaxf(0.f, X[i]);
}

void run(int S, int reps)
{
    const int H=8, W=8, HW=H*W;
    int C = (int)(tile_bytes(S) / (HW * sizeof(float)));
    if (C < 4) C = 4;

    size_t feat_sz = (size_t)C * HW * sizeof(float);
    size_t kern_sz = (size_t)C * 9  * sizeof(float);
    size_t ch_sz   = (size_t)C      * sizeof(float);

    float *d_X, *d_W, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_X,     feat_sz));
    CUDA_CHECK(cudaMalloc(&d_W,     kern_sz));
    CUDA_CHECK(cudaMalloc(&d_gamma, ch_sz));
    CUDA_CHECK(cudaMalloc(&d_beta,  ch_sz));
    CUDA_CHECK(cudaMemset(d_X,     1, feat_sz));
    CUDA_CHECK(cudaMemset(d_W,     1, kern_sz));
    CUDA_CHECK(cudaMemset(d_gamma, 1, ch_sz));
    CUDA_CHECK(cudaMemset(d_beta,  0, ch_sz));

    dim3 blk(8,8);
    dim3 grd(1, 1, C);

    for (int r=0; r<reps; ++r) {
        depthwise_conv3x3<<<grd,blk>>>(d_X, d_W, d_X, C, H, W);
        batchnorm<<<C,1>>>(d_X, d_gamma, d_beta, C, HW);
        relu_inplace<<<(C*HW+255)/256, 256>>>(d_X, C*HW);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaFree(d_X); cudaFree(d_W); cudaFree(d_gamma); cudaFree(d_beta);
    fprintf(stderr, "[cbr]         tile=%d  C=%d  H=%d  W=%d  "
                    "feat=%.0fKB\n",
            S, C, H, W, feat_sz/1024.f);
}
} // namespace cbr_op


// ===========================================================================
// OPERATOR 3: layer_norm  (3 kernels)
//
//   R = tile_bytes / sizeof(float)  aligned to 32   -- rows in fast memory
//   C = R * 8                                        -- hidden dim (cols)
//
//   Kernel 1: mean[R]  = row_mean(X[R,C])
//   Kernel 2: var[R]   = row_var(X[R,C], mean)
//   Kernel 3: Y[R,C]   = (X - mean) / sqrt(var+eps) * gamma + beta
//
//   Total DRAM I/O:
//     K1 reads  X              = R*C * 4 B
//     K2 reads  X, mean        = (R*C + R) * 4 B
//     K3 reads  X,mean,var,g,b = (R*C + 3R + 2C) * 4 B
//        writes Y              = R*C * 4 B
//     Sum ~ 4*R*C * 4 B  ∝ S^2   (monotone, no noise)
// ===========================================================================
namespace layer_norm_op {

// Kernel 1: per-row mean
__global__ void row_mean(const float* __restrict__ X,
                          float*       __restrict__ mean,
                          int R, int C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= R) return;
    const float* x = X + (size_t)row * C;
    float s = 0.f;
    for (int j = 0; j < C; ++j) s += x[j];
    mean[row] = s / C;
}

// Kernel 2: per-row variance
__global__ void row_var(const float* __restrict__ X,
                         const float* __restrict__ mean,
                         float*       __restrict__ var,
                         int R, int C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= R) return;
    const float* x = X + (size_t)row * C;
    float m = mean[row], v = 0.f;
    for (int j = 0; j < C; ++j) { float d = x[j] - m; v += d*d; }
    var[row] = v / C;
}

// Kernel 3: normalize and apply affine transform
__global__ void norm_apply(const float* __restrict__ X,
                            const float* __restrict__ mean,
                            const float* __restrict__ var,
                            const float* __restrict__ gamma,
                            const float* __restrict__ beta,
                            float*       __restrict__ Y,
                            int R, int C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= R || col >= C) return;
    float inv = rsqrtf(var[row] + 1e-5f);
    Y[(size_t)row * C + col] =
        (X[(size_t)row * C + col] - mean[row]) * inv * gamma[col] + beta[col];
}

void run(int S, int reps)
{
    int R = (int)(tile_bytes(S) / sizeof(float));
    if (R < 32) R = 32;
    R = (R / 32) * 32;
    int C = R * 8;

    size_t x_sz     = (size_t)R * C * sizeof(float);
    size_t stat_sz  = (size_t)R     * sizeof(float);
    size_t param_sz = (size_t)C     * sizeof(float);

    float *d_X, *d_Y, *d_mean, *d_var, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_X,     x_sz));
    CUDA_CHECK(cudaMalloc(&d_Y,     x_sz));
    CUDA_CHECK(cudaMalloc(&d_mean,  stat_sz));
    CUDA_CHECK(cudaMalloc(&d_var,   stat_sz));
    CUDA_CHECK(cudaMalloc(&d_gamma, param_sz));
    CUDA_CHECK(cudaMalloc(&d_beta,  param_sz));
    CUDA_CHECK(cudaMemset(d_X,     1, x_sz));
    CUDA_CHECK(cudaMemset(d_gamma, 1, param_sz));
    CUDA_CHECK(cudaMemset(d_beta,  0, param_sz));

    int blk1 = 256;
    int grd1  = (R + blk1 - 1) / blk1;
    dim3 blk3(32, 8);
    dim3 grd3((C + 31) / 32, (R + 7) / 8);

    for (int r = 0; r < reps; ++r) {
        row_mean  <<<grd1, blk1>>>(d_X, d_mean, R, C);
        row_var   <<<grd1, blk1>>>(d_X, d_mean, d_var, R, C);
        norm_apply<<<grd3, blk3>>>(d_X, d_mean, d_var, d_gamma, d_beta, d_Y, R, C);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaFree(d_X);  cudaFree(d_Y);
    cudaFree(d_mean); cudaFree(d_var);
    cudaFree(d_gamma); cudaFree(d_beta);
    fprintf(stderr, "[layer_norm]  tile=%d  R=%d  C=%d  "
                    "X=%.0fKB  total_IO~=%.0fKB\n",
            S, R, C, x_sz/1024.f, 4*x_sz/1024.f);
}
} // namespace layer_norm_op


// ===========================================================================
// main
// ===========================================================================
static void usage(const char* p) {
    fprintf(stderr,
        "Usage: %s --op <OP> --tile <S> --reps <N>\n"
        "  --op    flash_attn | cbr | layer_norm\n"
        "  --tile  cache-line tile size 64..384  (1 CL = 128 bytes)\n"
        "  --reps  number of kernel launches\n"
        "Example:\n"
        "  %s --op layer_norm --tile 256 --reps 8\n", p, p);
    exit(1);
}

int main(int argc, char** argv)
{
    const char* op = nullptr;
    int tile = 0, reps = 0;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i],"--op")  && i+1<argc) op   = argv[++i];
        else if (!strcmp(argv[i],"--tile")&& i+1<argc) tile = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--reps")&& i+1<argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")) usage(argv[0]);
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); }
    }
    if (!op || tile <= 0 || reps <= 0) {
        fprintf(stderr, "Missing or invalid args.\n"); usage(argv[0]);
    }

    fprintf(stderr, "[benchmark] op=%s  tile=%d  reps=%d  tile_bytes=%zu\n",
            op, tile, reps, tile_bytes(tile));

    if      (!strcmp(op,"flash_attn")) flash_attn_op::run(tile, reps);
    else if (!strcmp(op,"cbr"))        cbr_op::run(tile, reps);
    else if (!strcmp(op,"layer_norm")) layer_norm_op::run(tile, reps);
    else { fprintf(stderr,"Unknown op: %s\n", op); usage(argv[0]); }

    fprintf(stderr, "[benchmark] done.\n");
    return 0;
}
