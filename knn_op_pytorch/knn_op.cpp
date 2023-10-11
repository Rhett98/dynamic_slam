#include <cstdlib>
#include <ctime>
#include <iostream>
#include <torch/extension.h>

#include "nanoflann.hpp"
#include "utils.h"

using torch::Tensor;

// ref_pts: (N1, 3)
// query_pts: (N2, 3)
template <typename num_t>
Tensor knn_search(const Tensor &ref_pts, const Tensor &query_pts, const size_t k)
{
    static_assert(
        std::is_standard_layout<nanoflann::ResultItem<num_t, size_t>>::value,
        "Unexpected memory layout for nanoflann::ResultItem");
    
    Tensor distances = torch::zeros({query_pts.size(0)}, ref_pts.options());

    PointCloud<num_t> ref_cloud;
    PointCloud<num_t> query_cloud;
    initPointCloudFromTensor(ref_cloud, ref_pts);
    initPointCloudFromTensor(query_cloud, query_pts);
    
    size_t num_query_pts = query_cloud.kdtree_get_point_count();

    // construct a kd-tree index:
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>,
        PointCloud<num_t>, 3 /* dim */>;

    // dump_mem_usage();

    my_kd_tree_t index(3 /*dim*/, ref_cloud, {10 /* max leaf */});

    // dump_mem_usage();
    for(size_t query_pt_idx = 0; query_pt_idx < num_query_pts; ++query_pt_idx)
    {
        // do a knn search
        size_t ret_index[k];
        num_t out_dist_sqr[k];
        num_t k_mean_dist = 0;
        const float query_pt[3] = {query_cloud.kdtree_get_pt(query_pt_idx, 0),
                                   query_cloud.kdtree_get_pt(query_pt_idx, 1),
                                   query_cloud.kdtree_get_pt(query_pt_idx, 2)
                                   };

        nanoflann::KNNResultSet<num_t> resultSet(k);
        resultSet.init(ret_index, out_dist_sqr);
        index.findNeighbors(resultSet, &query_pt[0]);
        for(size_t neighbor_idx = 0; neighbor_idx < k; ++neighbor_idx)
        {
            k_mean_dist += out_dist_sqr[neighbor_idx];
        }
        distances[query_pt_idx] = k_mean_dist/k;
    }
    return distances;
}


Tensor knn_loss(const Tensor feats, const Tensor points, const size_t k)
{
    return knn_search<float>(feats, points, k);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("knn_loss", &knn_loss);
}

