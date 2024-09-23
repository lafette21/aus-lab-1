#include "ransac.hh"


// CUDA Kernel for calculating distances and checking inliers
__global__ void calculate_distances(
    const nova::Vec3f* points,
    std::size_t points_size,
    nova::Vec3f S0,
    float r,
    float threshold,
    float* distances,
    bool* is_inliers,
    unsigned int* inliers_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= points_size) return;

    // Calculate distance
    nova::Vec3f tmp_v3 = points[idx] - S0;
    nova::Vec2f tmp_v2 = {tmp_v3.x(), tmp_v3.y()};
    float dist = fabs(tmp_v2.length() - r);

    distances[idx] = dist;
    is_inliers[idx] = dist < threshold ? 1 : 0;

    // Count inliers using atomic addition
    if (is_inliers[idx]) {
        atomicAdd(inliers_count, 1);
    }
}

auto calculate_RANSAC_diffs_cuda(
    const nova::Vec3f* points,
    std::size_t points_size,
    const nova::Vec4f& cylinder,
    float threshold
)
        -> RANSAC_diffs_cuda
{
    const nova::Vec3f S0 { cylinder.x(), cylinder.y(), cylinder.z() };
    const float r = cylinder.w();

    // Allocate memory for device and host
    nova::Vec3f* d_points;
    float* d_distances;
    bool* d_is_inliers;
    unsigned int* d_inliers_count;
    unsigned int h_inliers_count = 0;

    cudaMalloc(&d_points, points_size * sizeof(nova::Vec3f));
    cudaMalloc(&d_distances, points_size * sizeof(float));
    cudaMalloc(&d_is_inliers, points_size * sizeof(bool));
    cudaMalloc(&d_inliers_count, sizeof(unsigned int));

    cudaMemcpy(d_points, points, points_size * sizeof(nova::Vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inliers_count, &h_inliers_count, sizeof(unsigned int), cudaMemcpyHostToDevice);

    int blockSize = 256; // You can adjust the block size for optimization
    int numBlocks = (points_size + blockSize - 1) / blockSize;

    // Launch CUDA kernel
    calculate_distances<<<numBlocks, blockSize>>>(d_points, points_size, S0, r, threshold, d_distances, d_is_inliers, d_inliers_count);

    // Copy back results to host
    std::vector<float> distances(points_size);
    std::vector<char> is_inliers(points_size);
    cudaMemcpy(distances.data(), d_distances, points_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(is_inliers.data(), d_is_inliers, points_size * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_inliers_count, d_inliers_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_distances);
    cudaFree(d_is_inliers);
    cudaFree(d_inliers_count);

    return {
        .num_inliers = h_inliers_count,
        .distances = distances,
        .is_inliers = is_inliers,
    };
}
