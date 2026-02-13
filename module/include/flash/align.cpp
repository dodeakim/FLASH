#include "flash/flash.h"

Eigen::Matrix4d toTransformation(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) 
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

std::vector<int> SampleUniqueIndices(int total, int num_samples, std::mt19937& rng) 
{
    std::set<int> indices;
    std::uniform_int_distribution<int> distrib(0, total - 1);
    while (indices.size() < num_samples)
        indices.insert(distrib(rng));
    return std::vector<int>(indices.begin(), indices.end());
}


float MeasureOverlap()
{
    
    return 0.0;
}


namespace FLASH
{

float SemanticGraph::ComputeSimilarity(const SemanticGraph& other,
                                       const Eigen::Matrix4d& T, 
                                       const Eigen::MatrixXi& inliers) const
{

    auto [P, Pl] = Convert();           
    auto [Q, Ql] = other.Convert();     

    Eigen::Matrix3Xd Ptrans(3, P.cols());
    for (int i = 0; i < P.cols(); ++i)
    {
        Eigen::Vector4d ph;
        ph.head<3>() = P.col(i);
        ph(3)        = 1.0;            
        Ptrans.col(i) = (T * ph).head<3>();
    }

    const int N = inliers.rows();       
    if (N == 0) return 0.0f;            

    double mse = 0.0;                   
    for (int i = 0; i < N; ++i)
    {
        int idx_p = inliers(i, 0);      
        int idx_q = inliers(i, 1);      
        double dist_sq = (Ptrans.col(idx_p) - Q.col(idx_q)).squaredNorm();

        mse += dist_sq;
    }
    mse /= static_cast<double>(N); 
    return static_cast<float>(std::exp(-mse));
}

float SemanticGraph::ComputeSimilarity(const SemanticGraph& other, const Eigen::MatrixXi& inliers) const
{

    int N = static_cast<int>(vertices.size());
    int M = static_cast<int>(other.vertices.size());
    int I = static_cast<int>(inliers.rows());
    // return static_cast<float>(I) / static_cast<float>(std::min(N, M));
    return static_cast<float>(I) / static_cast<float>(N + M - I);
}


std::pair<Eigen::Matrix3Xd, std::vector<int>> SemanticGraph::Convert() const
{
    int N = static_cast<int>(vertices.size());
    
    // converting
    Eigen::Matrix3Xd P(3, N);
    std::vector<int> Pl(N);
    for (int i = 0; i < N; ++i) {
        P.col(i) = vertices[i].point.cast<double>();
        Pl[i] = vertices[i].label; 
    }
    return {P, Pl};
}

clipper::CLIPPER SemanticGraph::BuildCLIPPER(double sigma, double epsilon, bool parallelize) const
{
    // instantiate the invariant function that will be used to score associations
    clipper::invariants::EuclideanDistance::Params iparams;
    iparams.sigma = sigma;
    iparams.epsilon = epsilon;
    clipper::invariants::EuclideanDistancePtr invariant;
    invariant.reset(new clipper::invariants::EuclideanDistance(iparams));

    clipper::Params params;
    clipper::CLIPPER clipper(invariant, params);
    clipper.setParallelize(parallelize);

    return clipper;
}


Eigen::MatrixXi SemanticGraph::CLIPPERMatch(const SemanticGraph& other) const
{
    auto [P, Pl] = Convert();
    auto [Q, Ql] = other.Convert();
    int N = static_cast<int>(Pl.size());
    int M = static_cast<int>(Ql.size());

    // putative association
    std::vector<Eigen::Vector2i> pairs;
    for (int i = 0; i < N; ++i) {
        int li = vertices[i].label;

        for (int j = 0; j < M; ++j) {
            int lj = other.vertices[j].label;

            if (li == lj) {
                pairs.emplace_back(Eigen::Vector2i(i, j));
            }
        }
    }

    clipper::Association A = clipper::Association(pairs.size(), 2);
    for (size_t k = 0; k < pairs.size(); ++k) 
    {
        A(k, 0) = pairs[k][0];
        A(k, 1) = pairs[k][1];
    }

    // clipper
    auto clipper = BuildCLIPPER(0.2, 0.5, true);
    clipper.scorePairwiseConsistency(P, Q, A);
    clipper.solve();
    const clipper::Association Ain = clipper.getSelectedAssociations();

    return Ain;
}

// std::tuple<Eigen::Matrix4d, float> SemanticGraph::SVDReg(const SemanticGraph& other, const Eigen::MatrixXi& corres)
// {
//     int N = corres.rows();
//     if (N < 3) // Too few correspondences to perform matching
//     {
//         return {Eigen::Matrix4d::Identity(), 0.0f};
//     }

//     auto [P, Pl] = Convert();
//     auto [Q, Ql] = other.Convert();
    
//     Eigen::Matrix3Xd src(3, N);
//     Eigen::Matrix3Xd tgt(3, N);

//     for (int i=0; i<N; i++)
//     {
//         int pi = corres(i, 0); 
//         int qi = corres(i, 1); 
//         src.col(i) = P.col(pi);
//         tgt.col(i) = Q.col(qi);
//     }
//     Eigen::Matrix4d T = KabschUmeyama(src, tgt);
//     float sim = ComputeSimilarity(other, T, corres);
//     return {T, sim};
// }


Eigen::Matrix4d SemanticGraph::SVDReg(const SemanticGraph& other, const Eigen::MatrixXi& corres) const
{
    int N = corres.rows();
    if (N < 3) // Too few correspondences to perform matching
    {
        return Eigen::Matrix4d::Identity();
    }

    auto [P, Pl] = Convert();
    auto [Q, Ql] = other.Convert();
    
    Eigen::Matrix3Xd src(3, N);
    Eigen::Matrix3Xd tgt(3, N);

    for (int i=0; i<N; i++)
    {
        int pi = corres(i, 0); 
        int qi = corres(i, 1); 
        src.col(i) = P.col(pi);
        tgt.col(i) = Q.col(qi);
    }
    Eigen::Matrix4d T = KabschUmeyama(src, tgt);
    return T;
}


// i -> j
Eigen::Matrix4d SemanticGraph::KabschUmeyama(const Eigen::Matrix3Xd& P, const Eigen::Matrix3Xd& Q) const
{
    Eigen::Vector3d meanP = P.rowwise().mean();
    Eigen::Vector3d meanQ = Q.rowwise().mean();

    Eigen::Matrix3Xd Phat = P.colwise() - meanP;
    Eigen::Matrix3Xd Qhat = Q.colwise() - meanQ;

    Eigen::Matrix3d H = Phat * Qhat.transpose();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if ((V * U.transpose()).determinant() < 0) {
        D(2, 2) = -1;
    }

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    R = V * D * U.transpose();
    t = meanQ - R * meanP;

    auto T = toTransformation(R, t);
    return T;
}


}; // namespace FLASH