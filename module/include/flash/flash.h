#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>
#include <set>
#include <map>
#include <complex>
#include <omp.h>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
// #include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>

#include <clipper/clipper.h>
#include <clipper/utils.h>

// #include "flash/sh.hpp"
#include "flash/sph_harm.hpp"
#include "flash/parameter.hpp"
#include "flash/external/hungarian.h"


using PointType = pcl::PointXYZL;
using CloudPtr = pcl::PointCloud<PointType>::Ptr;
// using Basis = std::map<std::pair<int, int>, std::complex<float>>;
using Basis = std::map<std::pair<int, int>, std::vector<std::complex<double>>>;


double ComputeElapsed(const std::chrono::high_resolution_clock::time_point& start,
                         const std::chrono::high_resolution_clock::time_point& end);

Eigen::Matrix4d toTransformation(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

std::vector<int> SampleUniqueIndices(int total, int num_samples, std::mt19937& rng);

std::vector<double> RPE(Eigen::Matrix4d est_mat, Eigen::Matrix4d gt_mat);

namespace FLASH
{
    struct Vertex
    {
        Eigen::Vector3f point;
        int label;

        Vertex(const Eigen::Vector3f& _point, int _label)
        : point(_point), label(_label) {}
    };

    struct Edge
    {
        int src;
        int dst;

        Edge(int _src, int _dst)
        : src(_src), dst(_dst) {}
    };

    struct SemanticGraph
    {
        std::vector<Vertex> vertices;
        std::vector<Edge> edges;

        SemanticGraph(const std::vector<Vertex>& _vertices, const std::vector<Edge> _edges)
        : vertices(_vertices), edges(_edges) {}

        // float ComputeSimilarity();
        float ComputeSimilarity(const SemanticGraph& other, const Eigen::Matrix4d& T, const Eigen::MatrixXi& inliers) const;
        float ComputeSimilarity(const SemanticGraph& other, const Eigen::MatrixXi& inliers) const;
        std::pair<Eigen::Matrix3Xd, std::vector<int>> Convert() const;

        // geometry verification
        // std::pair<Eigen::Matrix4d, float> Align(const SemanticGraph& other, int max_iter = 100, float th = 0.3);
        clipper::CLIPPER BuildCLIPPER(double sigma, double epsilon, bool parallelize) const;
        Eigen::MatrixXi CLIPPERMatch(const SemanticGraph& other) const;
        // std::tuple<Eigen::Matrix4d, float> SVDReg(const SemanticGraph& other, const Eigen::MatrixXi& corres);
        Eigen::Matrix4d SVDReg(const SemanticGraph& other, const Eigen::MatrixXi& corres) const;

        // std::tuple<Eigen::Matrix4d, Eigen::MatrixXi> SVDRANSAC(const SemanticGraph& other, const Eigen::MatrixXi& corres, int max_iter = 100, float th = 0.3);
        std::tuple<Eigen::Matrix4d, Eigen::MatrixXi, float> SVDRANSAC(const SemanticGraph& other, const Eigen::MatrixXi& corres, int max_iter = 100, float th = 0.3);
        Eigen::Matrix4d KabschUmeyama(const Eigen::Matrix3Xd& P, const Eigen::Matrix3Xd& Q) const;
        Eigen::MatrixXi ComputeInliers(const Eigen::Matrix3Xd& P, const std::vector<int>& Pl,
                                        const Eigen::Matrix3Xd& Q, const std::vector<int>& Ql,
                                        const Eigen::Matrix4d& T, float th = 0.3);

        // void Transform(const Eigen::Matrix4d& t)
        // {
        //     // std::vector<Vertex> vertices;
        //     // Eigen::Vector3f point;

        // }

    };


    struct Descriptor
    {
        std::vector<float> values;

        Descriptor(const std::vector<float>& _values)
        : values(_values) {}

        float ComputeSimilarity(const Descriptor& other) const
        {
            auto d1 = values;
            auto d2 = other.values;

            if (d1.size() != d2.size() || d1.empty())
                return 0.0f;

            float dot = std::inner_product(d1.begin(), d1.end(),
                                        d2.begin(), 0.0f);

            float n1  = std::inner_product(d1.begin(), d1.end(), d1.begin(), 0.0f);
            float n2  = std::inner_product(d2.begin(), d2.end(), d2.begin(), 0.0f);

            float denom = std::sqrt(n1 * n2);
            float sim = (denom > 0.0f) ? dot / denom : 0.0f;

            return sim; 
        }

    };

    struct MatchingResult
    {
        int dst;
        float sim;
        Eigen::Matrix4d T;
        Eigen::MatrixXi inliers;

        MatchingResult()
        : dst(-1), sim(0.0f), T(Eigen::Matrix4d::Identity()), inliers(Eigen::MatrixXi(0, 2)) {}
    };  


    struct ResultSet
    {
        int src;
        std::vector<MatchingResult> candidates;

        ResultSet(int k) {
            for (int i=0; i<k; i++)
                candidates.push_back(MatchingResult());
        }

        void Clear()
        {
            candidates.clear();
        }

        void Sorting()
        {
            // std::stable_sort(candidates.begin(), candidates.end(),
            //     [](const MatchingResult& a, const MatchingResult& b)
            //     {
            //         if (a.dst < 0) return false;
            //         if (b.dst < 0) return true;
            //         return a.similarity > b.similarity;
            //     });

            // std::stable_sort(candidates.begin(), candidates.end(),
            //     [](const MatchingResult& a, const MatchingResult& b)
            //     {
            //         if (a.dst < 0) return false;
            //         if (b.dst < 0) return true;
            //         return a.inliers.rows() > b.inliers.rows();
            //     });
            
            std::stable_sort(candidates.begin(), candidates.end(),
                [](const MatchingResult& a, const MatchingResult& b)
                {
                    if (a.dst < 0) return false;
                    if (b.dst < 0) return true;

                    auto na = a.inliers.rows();
                    auto nb = b.inliers.rows();
                    if (na != nb) return na > nb;        
                    return a.sim > b.sim;  
                });

        }
    };

    class Generator
    {
        private:
            Params _params;

            pcl::VoxelGrid<PointType> _voxel;
            std::vector<Eigen::Vector3f> _anchors; // valid fibonacci points in LiDAR FoV
            // Eigen::MatrixXf _anchors;
            int _num_valid_anchors;

            std::vector<double> _thetas, _phis;
            Basis _ylms;
            HungarianAlgorithm _hungarian;
            std::unordered_map<std::string, int> _global_wl_map;
            int _global_wl_id;
            int _debug_idx = 0;


        public:
            Generator();
            void UpdateParams(const Params& params);
            

            void WLInit();
            std::unordered_map<std::string, int> GetWL();
            void SetWL(const std::unordered_map<std::string, int>& map);
            void LoadWL();
            void SaveWL();            
            std::vector<Eigen::Vector3f> GetValidAnchor();

            // precomputed vals
            // Eigen::MatrixXf FibonacciLattice(int N);
            std::vector<Eigen::Vector3f> FibonacciLattice(int N);
            std::vector<Eigen::Vector3f> FibonacciLattice(int N, float fov_l, float fov_r, float fov_d, float fov_u);
            void computeSphericalCoordinates();
            void SHBasis();


            std::pair<SemanticGraph, Descriptor> Generate(const CloudPtr& cloud_in);

            // funtions
            CloudPtr VoxelDownsample(const CloudPtr& cloud_in);
            std::pair<CloudPtr, CloudPtr> DivideCloud(const CloudPtr& cloud_in);
            std::vector<Vertex> ExtractVertices(const CloudPtr& cloud_in);
            std::pair<std::vector<Edge>, Eigen::MatrixXi> ConnectEdges(const std::vector<Vertex>& vertices);

            std::vector<float> WLColoring(const std::vector<int>& labels, const Eigen::MatrixXi& adjacency);
            // std::vector<int> LinearSumAssignment(const std::vector<Vertex>& vertices, const std::vector<Eigen::Vector3f>& anchors);
            std::vector<int> LinearSumAssignment(const std::vector<Eigen::Vector3f>& vertices, const std::vector<Eigen::Vector3f>& anchors);
            std::vector<int> NearestAssignment(const std::vector<Eigen::Vector3f>& vertices, const std::vector<Eigen::Vector3f>& anchors);
            
            Eigen::MatrixXf ForegroundEncoding(const std::vector<Vertex>& vertices, const Eigen::MatrixXi& adjacency);
            Eigen::MatrixXf BackgroundEncoding(const CloudPtr& bg);
            std::vector<float> DescriptorGeneration(const Eigen::MatrixXf& f_signal, const Eigen::MatrixXf& b_signal);



            

    };

}; // namespace FLASH       