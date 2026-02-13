#include "flash/manager.h"




namespace FLASH
{

Manager::Manager() 
{
    _gen = std::make_unique<Generator>();
    _tree = nullptr;

}

int Manager::Size()
{
    return static_cast<int>(_graphs.size());
}

void Manager::LoadParams(const YAML::Node& cfg)
{
    const YAML::Node& cfg_ = cfg["FLASH"];
    
    _params.Clear();
    auto& p = _params;

    auto get_f = [&](const std::string& key) { return cfg_[key].as<float>(); };
    auto get_i = [&](const std::string& key) { return cfg_[key].as<int>(); };
    auto get_b = [&](const std::string& key) { return cfg_[key].as<bool>(); };
    auto get_s = [&](const std::string& key) { return cfg_[key].as<std::string>(); };
    
    p.fov_left          = get_f("fov_left");
    p.fov_right         = get_f("fov_right");
    p.fov_down          = get_f("fov_down");
    p.fov_up            = get_f("fov_up");

    p.leaf_size        = get_f("leaf_size");
    p.edge_th          = get_f("edge_th");
    p.num_hop          = get_i("num_hop");
    p.num_anchor       = get_i("num_anchor");
    p.l                = get_i("l");
    p.inlier_th   = get_f("inlier_th");
    p.similarity_th = get_f("similarity_th");

    // p.desc_type = get_i("desc_type");

    p.use_precomputed_wl_map   = get_b("use_precomputed_wl_map");
    p.precomputed_wl_map_path  = get_s("precomputed_wl_map_path");

    // foreground_label
    for (const auto& it : cfg_["foreground_label"]) {
        int label = it.first.as<int>();
        std::string name = it.second.as<std::string>();
        p.foreground_label[label] = name;
    }

    // background_label
    for (const auto& it : cfg_["background_label"]) {
        int label = it.first.as<int>();
        std::string name = it.second.as<std::string>();
        p.background_label[label] = name;
        // std::cout << "label : " << label << ", name : "<<  name << std::endl;

    }

    // eps_per_label
    for (const auto& it : cfg_["eps_per_label"]) {
        int label = it.first.as<int>();
        float eps = it.second.as<float>();
        p.eps_per_label[label] = eps;
    }

    // min_pts_per_label
    for (const auto& it : cfg_["min_pts_per_label"]) {
        int label = it.first.as<int>();
        int min_pts = it.second.as<int>();
        p.min_pts_per_label[label] = min_pts;
    }

    // color_map
    for (const auto& it : cfg_["color_map"]) {
        int label = it.first.as<int>();
        std::vector<int> rgb_vec = it.second.as<std::vector<int>>();
        p.color_map[label] = { rgb_vec[0], rgb_vec[1], rgb_vec[2] };
    }

    // Update generator parameters from configured values
    _gen->UpdateParams(_params);
    
    std::cout << "Manager Parameter Load Complete!" << std::endl;
    std::cout << "FoV Up : " << _params.fov_up << std::endl;
    std::cout << "FoV Down : " << _params.fov_down << std::endl;
    std::cout << "FoV Right : " << _params.fov_right << std::endl;
    std::cout << "FoV Left : " << _params.fov_left << std::endl;
}

void Manager::LoadWL()
{
    _gen->LoadWL();
}

void Manager::SaveWL()
{
    _gen->SaveWL();
}

Params Manager::GetParams()
{
    return _params;
}

std::vector<Eigen::Vector3f> Manager::GetValidAnchor()
{
    return _gen->GetValidAnchor();
}

SemanticGraph Manager::GetGraph(int idx)
{
    return _graphs[idx];
}

Descriptor Manager::GetDesc(int idx)
{
    return _descs[idx];
}

std::pair<SemanticGraph, Descriptor> Manager::GetQuery()
{   
    int query_idx = Size() - 1;
    auto graph = GetGraph(query_idx);
    auto desc = GetDesc(query_idx);
    return {graph, desc};
}



std::pair<SemanticGraph, Descriptor> Manager::Generate(const CloudPtr& cloud_in)
{
    return _gen->Generate(cloud_in);
}

void Manager::Save(const SemanticGraph& graph, const Descriptor& desc)
{
    _graphs.push_back(graph);
    _descs.push_back(desc);
}

void Manager::GenerateAndSave(const CloudPtr& cloud_in)
{
    auto [graph, desc] = _gen->Generate(cloud_in);
    Save(graph, desc);
}

std::vector<size_t> Manager::Retrieve(const Descriptor& desc, int k)
{
    std::vector<float> qd = desc.values;
    const int N = static_cast<int>(_descs.size());
    
    if (k > N)
    {
        k = N;
    }

    if (_tree == nullptr) 
    {
        _descs_tree.clear();
        _descs_tree.reserve(N);
        for (const auto& d : _descs) _descs_tree.push_back(d.values);
        const size_t dim = _descs_tree.front().size();
        // std::cout << "dim : " << dim << std::endl;
        _tree = std::make_unique<DescTree>(dim, _descs_tree, 10 /* max leaf */);
    }

    std::vector<size_t> knn_indices( k ); 
    std::vector<float> knn_distances( k );
    nanoflann::KNNResultSet<float> knn_result(k);
    knn_result.init( &knn_indices[0], &knn_distances[0] );
    _tree->index->findNeighbors(knn_result, &qd[0], nanoflann::SearchParameters(10));
    return knn_indices;
}

MatchingResult Manager::VerifyOne(int cand_idx, const SemanticGraph& qgraph, const Descriptor& qdesc) const
{
    MatchingResult out;
    out.dst     = cand_idx; // if you need candidate id, activate 
    const auto& cgraph = _graphs[cand_idx];
    const auto& cdesc  = _descs[cand_idx];

    const auto corres = qgraph.CLIPPERMatch(cgraph);
    if (corres.rows() < _params.inlier_th || corres.rows() < 3) return out;

    const auto  T         = qgraph.SVDReg(cgraph, corres);
    // const float graph_sim = qgraph.ComputeSimilarity(cgraph, T, corres);
    const float graph_sim = qgraph.ComputeSimilarity(cgraph, corres);
    const float desc_sim  = qdesc.ComputeSimilarity(cdesc);

    const float fs = graph_sim * desc_sim;
    // const float fs = std::sqrt(graph_sim * desc_sim);
    // const float fs = 2.f * graph_sim * desc_sim / (graph_sim + desc_sim); 
    // const float fs = 0.5f * (graph_sim + desc_sim);
    // const float fs = (graph_sim * desc_sim) / (graph_sim * desc_sim + (1 - graph_sim) * (1 - desc_sim));
    
    if (fs < _params.similarity_th) return out;
    out.dst     = cand_idx;
    out.sim     = fs;
    out.T       = T;
    out.inliers = corres;
    return out;
}


ResultSet Manager::Verify(int idx, const SemanticGraph& graph, const Descriptor& desc, const std::vector<size_t>& topk)
{

    int k = static_cast<int>(topk.size());
    ResultSet result(k);
    result.src = idx;

    const auto& qgraph = graph;
    const auto& qdesc = desc;
    for (int i = 0; i < k; ++i)
    {
        int cand_idx = static_cast<int>(topk[i]);
        result.candidates[i] = VerifyOne(cand_idx, qgraph, qdesc);
    }
    result.Sorting();
    return result;
}


// scancontext-like
ResultSet Manager::OnlineLCD(int k, int num_exclude_recent)
{
    ResultSet result(k); // init result

    // query
    int size = Size();
    int query_idx = size - 1;
    result.src = query_idx;
    if (size <= num_exclude_recent) {return result;}

    auto [qgraph, qdesc] = GetQuery();
    std::vector<float> qd = qdesc.values;
    
    // database subset
    std::vector<SemanticGraph> db_graphs(_graphs.begin(), _graphs.end() - num_exclude_recent);
    std::vector<Descriptor> db_descs(_descs.begin(), _descs.end() - num_exclude_recent);

    const int N = static_cast<int>(db_descs.size());
    _descs_tree.clear();
    _descs_tree.reserve(N);
    for (const auto& d : db_descs) _descs_tree.push_back(d.values);
    const size_t dim = _descs_tree.front().size();

    // k-d tree update
    _tree = std::make_unique<DescTree>(dim, _descs_tree, 10 /* max leaf */);

    int valid_k = std::min(k, N);
    std::vector<size_t> knn_indices(k, size_t(-1));
    std::vector<float>  knn_distances(k, std::numeric_limits<float>::max());
    nanoflann::KNNResultSet<float> knn_result(valid_k);
    knn_result.init( &knn_indices[0], &knn_distances[0] );
    _tree->index->findNeighbors(knn_result, &qd[0], nanoflann::SearchParameters(10));

    // geometry verification
    for (int i = 0; i < k; ++i)
    {
        if (knn_indices[i] == size_t(-1)) continue;

        int cand_idx = static_cast<int>(knn_indices[i]);
        result.candidates[i] = VerifyOne(cand_idx, qgraph, qdesc);
    }

    
    result.Sorting();
    return result;
}



ResultSet Manager::PR(const CloudPtr& cloud_in, int k)
{
    // generate FLASH graph and desc
    auto [graph, desc] = Generate(cloud_in);

    // prepare place recognition results
    ResultSet result(k); // init result

    // retreive by descriptor
    auto knn_indices = Retrieve(desc, k); 

    // geometry verification
    for (int i = 0; i < k; ++i)
    {
        if (knn_indices[i] == size_t(-1)) continue;

        int cand_idx = static_cast<int>(knn_indices[i]);
        result.candidates[i] = VerifyOne(cand_idx, graph, desc);
    }

    result.Sorting();
    return result;
}


ResultSet Manager::PR(const SemanticGraph& graph, const Descriptor& desc, int k)
{
    // prepare place recognition results
    ResultSet result(k); // init result

    // retreive by descriptor
    auto knn_indices = Retrieve(desc, k); 

    // geometry verification
    for (int i = 0; i < k; ++i)
    {
        if (knn_indices[i] == size_t(-1)) continue;

        int cand_idx = static_cast<int>(knn_indices[i]);
        result.candidates[i] = VerifyOne(cand_idx, graph, desc);
    }

    result.Sorting();
    return result;
}


float Manager::PairwiseMatching(const std::pair<SemanticGraph, Descriptor>& src, const std::pair<SemanticGraph, Descriptor>& dst)
{
    auto src_graph = src.first;
    auto src_desc = src.second;
    auto dst_graph = dst.first;
    auto dst_desc = dst.second;

    const auto corres = src_graph.CLIPPERMatch(dst_graph);
    if (corres.rows() < _params.inlier_th || corres.rows() < 3) return 0.0;

    const auto  T         = src_graph.SVDReg(dst_graph, corres);
    const float graph_sim = src_graph.ComputeSimilarity(dst_graph, corres);
    const float desc_sim  = src_desc.ComputeSimilarity(dst_desc);

    const float fs = graph_sim * desc_sim;
    return fs;
}

}; // namespace FLASH