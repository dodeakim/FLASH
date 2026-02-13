#include "flash/flash.h"



double ComputeElapsed(const std::chrono::high_resolution_clock::time_point& start,
                         const std::chrono::high_resolution_clock::time_point& end)
{
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();  // seconds
}

float rad2deg(float radians)
{
    return radians * 180.0 / M_PI;
}

float deg2rad(float degrees)
{
    return degrees * M_PI / 180.0;
}


float xy2theta( const float & _x, const float & _y )
{
    if ( _x >= 0 & _y >= 0) 
        return (180/M_PI) * atan(_y / _x);

    if ( _x < 0 & _y >= 0) 
        return 180 - ( (180/M_PI) * atan(_y / (-_x)) );

    if ( _x < 0 & _y < 0) 
        return 180 + ( (180/M_PI) * atan(_y / _x) );

    if ( _x >= 0 & _y < 0)
        return 360 - ( (180/M_PI) * atan((-_y) / _x) );
} // xy2theta






namespace FLASH
{


Generator::Generator() 
{
    // std::cout << "FLASH Basic Construction " << std::endl;
    if (_params.leaf_size != 0.0)
        _voxel.setLeafSize(_params.leaf_size, _params.leaf_size, _params.leaf_size);

    // precompute fibonacci lattice
    // _anchors = FibonacciLattice(_params.num_anchor);
    _anchors = FibonacciLattice(_params.num_anchor,
        _params.fov_left, _params.fov_right, _params.fov_down, _params.fov_up);
    _num_valid_anchors = static_cast<int>(_anchors.size());

    computeSphericalCoordinates();


    SHBasis();
    WLInit();

}

void Generator::UpdateParams(const Params& params)
{
    // _params = params;
    _params.Update(params);
    
    // update
    if (_params.leaf_size != 0.0)
        _voxel.setLeafSize(_params.leaf_size, _params.leaf_size, _params.leaf_size);
    // _anchors = FibonacciLattice(_params.num_anchor);
    _anchors = FibonacciLattice(_params.num_anchor,
        _params.fov_left, _params.fov_right, _params.fov_down, _params.fov_up);
    _num_valid_anchors = static_cast<int>(_anchors.size());

    computeSphericalCoordinates();
    SHBasis();

    if(_params.use_precomputed_wl_map)
    {
        LoadWL();
    }
    // std::cout << "Generator Parameter Update Complete! " << std::endl;
}

void Generator::WLInit()
{
    _global_wl_map.clear();
    _global_wl_id = 1;
    // std::cout << "WL init complete => " << "GLOBAL ID : " << _global_wl_id << std::endl;
}

void Generator::SetWL(const std::unordered_map<std::string, int>& map)
{
    _global_wl_map = map;
    int max_id = 0;
    for (const auto& [_, val] : _global_wl_map) 
    {
        if (val > max_id) max_id = val;
    }
    _global_wl_id = max_id + 1;
}

std::unordered_map<std::string, int> Generator::GetWL()
{
    return _global_wl_map;
}

void Generator::SaveWL()
{
    auto path = _params.precomputed_wl_map_path;

    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "WL cant save " << path << std::endl;
        return;
    }

    const auto& wl = GetWL();

    for (const auto& [key, val] : wl) {
        out << key << " " << val << "\n";
    }

    out.close();
    std::cout << "SAVE COMPLETE WL MAP" << std::endl;

}

void Generator::LoadWL()
{
    auto path = _params.precomputed_wl_map_path;

    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "WL cant read " << path << std::endl;
        return;
    }

    std::unordered_map<std::string, int> wl;
    std::string key;
    int value;

    while (in >> key >> value) {
        wl[key] = value;
    }

    in.close();
    SetWL(wl); 
    std::cout << "WL load complete => " << "GLOBAL ID : " << _global_wl_id << std::endl;
}


std::vector<Eigen::Vector3f> Generator::GetValidAnchor()
{
    return _anchors;
}

std::pair<SemanticGraph, Descriptor> Generator::Generate(const CloudPtr& cloud_in)
{
    // std::cout << "\n" << std::endl;
    // auto start = std::chrono::high_resolution_clock::now();
    
    // preprocessing
    CloudPtr cloud; 
    if (_params.leaf_size > 0.0) {
        cloud = VoxelDownsample(cloud_in);
        // std::cout << "Downsample : " << cloud_in->size() << " => " << cloud->size() << std::endl;
    } else {
        cloud = cloud_in;
    }

    auto [fg, bg] = DivideCloud(cloud);
    // std::cout << "Divide Foreground : " << fg->size() << std::endl;
    // std::cout << "Divide Background : " << bg->size() << std::endl;

    // semantic graph construction
    auto vertices = ExtractVertices(fg);
    auto [edges, adjacency] = ConnectEdges(vertices);
    // std::cout << "Num Vertices : " << vertices.size() << std::endl;
    // std::cout << "Num Edges : " << edges.size() << std::endl;

    // signal encoding
    // Eigen::MatrixXf f_signal;
    auto f_signal = ForegroundEncoding(vertices, adjacency); // 0.05
    auto b_signal = BackgroundEncoding(bg);                  // ~

    // descriptor generation    
    auto desc = DescriptorGeneration(f_signal, b_signal);
    // auto desc = DescriptorGenerationTest(f_signal, b_signal);
    // std::cout << desc.size() << std::endl;
    
    return {SemanticGraph(vertices, edges), Descriptor(desc)};
}



// funtions 
// x-forward, y-left, z-up
std::vector<Eigen::Vector3f> Generator::FibonacciLattice(int N)
{
    std::vector<Eigen::Vector3f> points;
    points.reserve(N);

    const float phi = (std::sqrt(5.0f) + 1.0f) / 2.0f;
    const float golden_angle = 2.0f * M_PI / phi;

    for (int i = 0; i < N; ++i)
    {
        float z = 1.0f - 2.0f * static_cast<float>(i) / (N - 1);
        float radius = std::sqrt(1.0f - z * z);
        float theta = golden_angle * i;

        float x = radius * std::cos(theta);
        float y = radius * std::sin(theta);

        points.emplace_back(x, y, z);
    }

    return points;
}

std::vector<Eigen::Vector3f> Generator::FibonacciLattice(
    int N, float fov_left, float fov_right, float fov_down, float fov_up)
{
    std::vector<Eigen::Vector3f> points_all = FibonacciLattice(N); 
    std::vector<Eigen::Vector3f> points;

    float left   = deg2rad(fov_left);
    float right  = deg2rad(fov_right);
    float down   = deg2rad(fov_down);
    float up     = deg2rad(fov_up);

    for (const auto& p : points_all)
    {
        float x = p.x();
        float y = p.y();
        float z = p.z();

        float azim = std::atan2(y, x); // [-pi, pi]
        float elev = std::asin(z);     // [-pi/2, pi/2]

        if (azim >= left && azim <= right && elev >= down && elev <= up)
            points.push_back(p);
    }

    // std::cout << "generated fibonacci points : " << N << std::endl;
    // std::cout << "generated valid points : " << points.size() << std::endl;

    return points;
}

void Generator::computeSphericalCoordinates() 
{
    size_t N = _anchors.size();
    _thetas.reserve(N);  
    _phis.reserve(N);    

    for (const auto& p : _anchors) {
        double theta = std::acos(p.z() / 1.0); 
        double phi = std::atan2(p.y(), p.x());
        if (phi < 0) phi += 2 * M_PI;

        _thetas.push_back(theta);
        _phis.push_back(phi);
    }
}

void Generator::SHBasis()
{
    // using Basis = std::map<std::pair<int, int>, std::complex<double>>;
    // Basis _ymls;
    // std::vector<double> thetas, phis;

    const int l = _params.l;
    // const int N = _params.num_anchor;
    const int N = _num_valid_anchors;

    _ylms.clear();

    for (int ell = 0; ell <= l; ++ell) 
    {
        for (int m = -ell; m <= ell; ++m) 
        {
            std::vector<std::complex<double>> Ylm_k;
            Ylm_k.reserve(N);

            for (int k = 0; k < N; ++k) 
            {
                // sph_harm(m, n, theta=azimuth, phi=polar)
                std::complex<double> Ylm = sph_harm(m, ell, _phis[k], _thetas[k]);
                Ylm_k.push_back(Ylm);
            }

            _ylms[{ell, m}] = std::move(Ylm_k);
        }
    }
}

CloudPtr Generator::VoxelDownsample(const CloudPtr& cloud_in)
{
    CloudPtr cloud_out(new pcl::PointCloud<PointType>);
    _voxel.setInputCloud(cloud_in);
    _voxel.filter(*cloud_out);

    // work 
    // std::cout << cloud_in->size() << std::endl;
    // std::cout << cloud_out->size() << std::endl;
    return cloud_out;
}

std::pair<CloudPtr, CloudPtr> Generator::DivideCloud(const CloudPtr& cloud_in)
{
    CloudPtr fg(new pcl::PointCloud<PointType>());
    CloudPtr bg(new pcl::PointCloud<PointType>());

    for (const auto& p : cloud_in->points)
    {
        int label = p.label;

        if (_params.foreground_label.count(label))
            fg->push_back(p);

        if (_params.background_label.count(label))
            bg->push_back(p);
    }

    return {fg, bg};
}

std::vector<Vertex> Generator::ExtractVertices(const CloudPtr& cloud_in)
{

    std::unordered_map<int, CloudPtr> cloud_map;
    for (const auto& p : cloud_in->points)
    {
        int label = p.label;
        if (!cloud_map[label]) {
            cloud_map[label] = CloudPtr(new pcl::PointCloud<PointType>);
        }
        cloud_map[label]->push_back(p);
    }

    std::vector<std::pair<int, CloudPtr>> label_clouds(cloud_map.begin(), cloud_map.end());

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<Vertex>> thread_local_vertices(num_threads);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(label_clouds.size()); ++i)
    {
        int thread_id = omp_get_thread_num();
        auto& local_vertices = thread_local_vertices[thread_id];

        int label = label_clouds[i].first;
        CloudPtr cloud = label_clouds[i].second;

        if (cloud->empty()) continue;

        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
        tree->setInputCloud(cloud);

        pcl::EuclideanClusterExtraction<PointType> ec;
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.setClusterTolerance(_params.eps_per_label[label]);
        ec.setMinClusterSize(_params.min_pts_per_label[label]);

        std::vector<pcl::PointIndices> cluster_indices;
        ec.extract(cluster_indices);

        for (const auto& cluster : cluster_indices)
        {
            if (cluster.indices.empty()) continue;

            Eigen::Vector3f point = Eigen::Vector3f::Zero();
            for (int idx : cluster.indices)
                point += cloud->points[idx].getVector3fMap();

            point /= static_cast<float>(cluster.indices.size());
            local_vertices.emplace_back(point, label);
        }
    }

    std::vector<Vertex> vertices;
    for (const auto& local : thread_local_vertices) {
        vertices.insert(vertices.end(), local.begin(), local.end());
    }

    return vertices;
}

std::pair<std::vector<Edge>, Eigen::MatrixXi> Generator::ConnectEdges(const std::vector<Vertex>& vertices)
{
    int N = static_cast<int>(vertices.size());

    std::vector<Edge> edges;
    Eigen::MatrixXi adjacency = Eigen::MatrixXi::Zero(N, N);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->resize(N);
    for (int i = 0; i < N; ++i)
    {
        const auto& v = vertices[i];
        (*cloud)[i].x = v.point.x();
        (*cloud)[i].y = v.point.y();
        (*cloud)[i].z = v.point.z();
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);

    for (int i = 0; i < N; ++i)
    {
        std::vector<int> neighbors;
        std::vector<float> sq_dists;

        tree.radiusSearch((*cloud)[i], _params.edge_th, neighbors, sq_dists);

        for (int j : neighbors)
        {
            if (j <= i) continue;
            edges.push_back(Edge(i, j));
            adjacency(i, j) = 1;
        }
    }
    adjacency += adjacency.transpose().eval();
    return {edges, adjacency};
}

// std::vector<int> Generator::LinearSumAssignment(const std::vector<Vertex>& vertices, const std::vector<Eigen::Vector3f>& anchors)
std::vector<int> Generator::LinearSumAssignment(const std::vector<Eigen::Vector3f>& vertices, const std::vector<Eigen::Vector3f>& anchors)
{

    // int N = 54; => 0.027

    int N = static_cast<int>(vertices.size());
    int M = static_cast<int>(anchors.size());

    std::vector<std::vector<double>> cost(N, std::vector<double>(M));

    // for (int i = 0; i < 54; ++i)
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            
            // const Eigen::Vector3f& v_i = vertices[i].xyz;
            // cost[i][j] = static_cast<double>((v_i - anchors[j]).norm());
            // cost[i][j] = static_cast<double>((v_i - anchors[j]).squaredNorm());
            cost[i][j] = static_cast<double>((vertices[i] - anchors[j]).norm());
        }
    }

    std::vector<int> assignment;
    // double total_cost = _hungarian.Solve(cost, assignment);
    // std::cout << "Total cost: " << total_cost << std::endl;
    (void)_hungarian.Solve(cost, assignment);    

    return assignment;
}


std::vector<int> Generator::NearestAssignment(const std::vector<Eigen::Vector3f>& vertices, const std::vector<Eigen::Vector3f>& anchors)
{
    int N = static_cast<int>(vertices.size());
    int M = static_cast<int>(anchors.size());
    std::vector<int> assignment(N);

    for (int i = 0; i < N; ++i)
    {
        Eigen::Vector3f p = vertices[i];
        Eigen::Vector3f _p_unit = p.normalized(); 
        int anchor_idx = 0;
        float max_dot = -1.0f;

        for (int j =0; j<M; ++j)
        {            
            float dot = anchors[j].dot(_p_unit);
            if (dot > max_dot) 
            {
                max_dot = dot;
                anchor_idx = j;
            }
        }

        assignment[i] = anchor_idx;
    }
    return assignment;
}


std::vector<float> Generator::WLColoring(const std::vector<int>& labels, const Eigen::MatrixXi& adjacency)
{
    int N = static_cast<int>(labels.size());
    std::vector<int> codes(labels);

    const int num_hop = _params.num_hop;
    for (int hop = 0; hop <= num_hop; ++hop)
    {
        std::vector<int> new_code(N, 0);

        for (int i = 0; i < N; ++i)
        {
            std::vector<int> multiset;
            for (int j = 0; j < N; ++j)
            {
                if (adjacency(i, j) != 0)
                    multiset.push_back(codes[j]);
            }

            std::sort(multiset.begin(), multiset.end());

            std::string key = std::to_string(codes[i]) + "|";
            for (int c : multiset)
            {
                key += std::to_string(c);
                key += ",";
            }

            auto it = _global_wl_map.find(key);
            if (it != _global_wl_map.end())
            {
                new_code[i] = it->second;
            }
            else
            {
                _global_wl_map[key] = _global_wl_id;
                new_code[i] = _global_wl_id;
                ++_global_wl_id;
            }
        }

        codes.swap(new_code);
    }

    std::vector<float> _codes(codes.begin(), codes.end());
    // if (!_codes.empty()) {
    //     float max_code = *std::max_element(_codes.begin(), _codes.end());
    //     if (max_code > 0.f)
    //         for (float& v : _codes) v /= max_code;
    // }

    return _codes;
}

Eigen::MatrixXf Generator::ForegroundEncoding(const std::vector<Vertex>& vertices, const Eigen::MatrixXi& adjacency)
{
    int N = static_cast<int>(vertices.size());

    const int num_foreground_label = static_cast<int>(_params.foreground_label.size());
    // const int num_anchor = _params.num_anchor;
    const int num_anchor = _num_valid_anchors;
    Eigen::MatrixXf f_signal = Eigen::MatrixXf::Zero(num_foreground_label, num_anchor);

    std::vector<Eigen::Vector3f> points;
    std::vector<int> labels;
    points.reserve(N);
    labels.reserve(N);
    
    for (const auto& vertex : vertices)
    {
        points.emplace_back(vertex.point);  
        labels.emplace_back(vertex.label);
    }

    auto codes = WLColoring(labels, adjacency);
    auto assignment = LinearSumAssignment(points, _anchors);
    // auto assignment = NearestAssignment(points, _anchors);
    // std::cout << "Num assignment : " << assignment.size() << std::endl;

    for (int i = 0; i<N; ++i)
    {

        int label = labels[i];
        int label_index = _params.foreground_label_to_index[label];
        int anchor_index = assignment[i];

        f_signal(label_index, anchor_index) = codes[i];
        // f_signal(label_index, anchor_index) = 1; // just directional information
    }

    return f_signal;
}

Eigen::MatrixXf Generator::BackgroundEncoding(const CloudPtr& bg)
{
    const int num_background_label = static_cast<int>(_params.background_label.size());
    // const int num_anchor = _params.num_anchor;
    const int num_anchor = _num_valid_anchors;

    Eigen::MatrixXf b_signal = Eigen::MatrixXf::Zero(num_background_label, num_anchor);

    // omp_set_num_threads(4); 
    int num_threads = omp_get_max_threads();
    
    std::vector<Eigen::MatrixXf> thread_buffers(num_threads, Eigen::MatrixXf::Zero(num_background_label, num_anchor));

    #pragma omp parallel for
    for (const auto& p : bg->points)
    {
        int label = p.label;
        int label_index = _params.background_label_to_index[label];

        Eigen::Vector3f _p(p.x, p.y, p.z);
        Eigen::Vector3f _p_unit = _p.normalized(); 

        int anchor_idx = 0;
        float max_dot = -1.0f;

        for (int j = 0; j < num_anchor; ++j) {
            float dot = _anchors[j].dot(_p_unit);
            if (dot > max_dot) {
                max_dot = dot;
                anchor_idx = j;
            }
        }

        int tid = omp_get_thread_num();
        thread_buffers[tid](label_index, anchor_idx) += 1.0f;
    }

    for (const auto& buffer : thread_buffers) 
    {
        b_signal += buffer;
    }
    

    return b_signal;
}


std::vector<float> Generator::DescriptorGeneration(const Eigen::MatrixXf& f_signal, const Eigen::MatrixXf& b_signal)
{
    const int num_foreground_label = f_signal.rows();
    const int num_background_label = b_signal.rows();
    const int l_max = _params.l;
    // const int num_anchor = _params.num_anchor;
    const int num_anchor = _num_valid_anchors;
    const double w = 4.0 * M_PI / num_anchor;

    auto compute_descriptor = [&](const Eigen::MatrixXf& signal, int num_label) -> std::vector<float>
    {
        std::vector<std::vector<float>> energy_thread(num_label);

        #pragma omp parallel for
        for (int i = 0; i < num_label; ++i)
        {
            const auto& f = signal.row(i);
            std::vector<float> energy_l;

            for (int ell = 0; ell <= l_max; ++ell) 
            {
                int m_count = 2 * ell + 1;
                std::vector<float> mags(m_count, 0.f);
                std::vector<float> phases(m_count, 0.f);

                for (int m = -ell; m <= ell; ++m) 
                {
                    int m_idx = m + ell;
                    const auto& Ylm = _ylms.at({ell, m});
                    std::complex<double> Alm = 0.0;

                    for (int k = 0; k < num_anchor; ++k)
                    {
                        double fk = static_cast<double>(f(k));
                        if (fk == 0.0) continue;
                        Alm += fk * std::conj(Ylm[k]) * w;
                    }

                    mags[m_idx] = static_cast<float>(std::abs(Alm));
                    // phases[m_idx] = static_cast<float>(std::arg(Alm));
                }

                float sum_mag = 0.f;
                for (int m_idx = 0; m_idx < m_count; ++m_idx) 
                {
                    float mag = mags[m_idx];
                    // float phase = phases[m_idx];
                    sum_mag += mag * mag;
                }

                energy_l.push_back(sum_mag);
            }
            energy_thread[i] = std::move(energy_l);
        }

        // flatten
        std::vector<float> energy;
        for (int i = 0; i < num_label; ++i) {
            energy.insert(energy.end(), energy_thread[i].begin(), energy_thread[i].end());
        }

        float n = std::sqrt(std::inner_product(energy.begin(), energy.end(), energy.begin(), 0.f));
        if (n > 0.f) for (float& v : energy) v /= n;

        return energy;
    };

    auto g_energy = compute_descriptor(f_signal, num_foreground_label);
    auto b_energy = compute_descriptor(b_signal, num_background_label);

    std::vector<float> desc;
    

    int mode = 0;
    std::vector<const std::vector<float>*> sources;
    if (mode == 0) {
        sources = { &g_energy, &b_energy };
    } else if (mode == 1) {
        sources = { &g_energy };
    } else if (mode == 2) {
        sources = { &b_energy };
    }

    size_t total_size = 0;
    for (auto* s : sources) total_size += s->size();

    desc.reserve(total_size);
    for (auto* s : sources) {
        desc.insert(desc.end(), s->begin(), s->end());
    }

    // std::cout << "mode : " << mode 
    //         << " / size : " << total_size << std::endl;
    return desc;
}

}; // namespace FLASH




