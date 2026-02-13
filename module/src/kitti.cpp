#include <iostream>
#include <iomanip> 
#include <fstream>                     
#include <filesystem>
#include <boost/filesystem.hpp>
#include <chrono> 
#include <vector>                      
#include <tuple>
#include <memory>
#include <algorithm>
#include <thread>
#include <array> 
#include <limits>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <pcl/point_types.h>        
#include <pcl/point_cloud.h>        
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>


// FLASH
#include "flash/manager.h"

namespace fs = std::filesystem;

pcl::PointCloud<pcl::PointXYZL>::Ptr read_cloud(const std::string& cloud_path,
                                                const std::string& label_path)
{
    using CloudL = pcl::PointCloud<pcl::PointXYZL>;
    CloudL::Ptr out(new CloudL());


    std::ifstream cloud_in(cloud_path, std::ios::binary);
    std::ifstream label_in(label_path, std::ios::binary);
    if (!cloud_in.is_open() || !label_in.is_open()) {
        std::cerr << "Could not open bin file: " << std::endl;
        return nullptr;
    }

    cloud_in.seekg(0, std::ios::end);
    std::size_t num_points = cloud_in.tellg() / (4 * sizeof(float));
    cloud_in.seekg(0, std::ios::beg);
    std::vector<float> raw_data(num_points * 4);
    cloud_in.read(reinterpret_cast<char*>(raw_data.data()), num_points * 4 * sizeof(float));
    cloud_in.close();

    std::vector<uint32_t> raw_labels(num_points);
    label_in.read(reinterpret_cast<char*>(raw_labels.data()), num_points * sizeof(uint32_t));
    label_in.close();

    out->points.resize(num_points);
    for (std::size_t i = 0; i < num_points; ++i) {
        auto& q = out->points[i];
        
        q.x = raw_data[i * 4 + 0];
        q.y = raw_data[i * 4 + 1];
        q.z = raw_data[i * 4 + 2];
        q.label = static_cast<uint16_t>(raw_labels[i] & 0xFFFF);
    }

    out->width = static_cast<uint32_t>(num_points);
    out->height = 1;
    out->is_dense = true;
    return out;
}

std::vector<std::string> get_file_list(const std::string& directory_path) 
{
    std::vector<std::string> files;
    boost::filesystem::path dir(directory_path);
    
    if (!boost::filesystem::exists(dir) || !boost::filesystem::is_directory(dir)) 
    {
        std::cerr << "Directory not found: " << directory_path << '\n';
        return files;
    }

    for (const auto& entry : boost::filesystem::directory_iterator(dir)) 
    {
        if (boost::filesystem::is_regular_file(entry.path())) 
            files.push_back(entry.path().string());
    }

    std::sort(files.begin(), files.end());
    return files;
}


void save_result(const fs::path& save_path, const FLASH::ResultSet& result)
{

    std::ofstream ofs(save_path, std::ios::app);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file: " + save_path.string());
    }

    ofs << result.src << " "
        << result.candidates[0].dst << " "
        << std::fixed << std::setprecision(6) << result.candidates[0].sim
        << "\n";
}


int main(int argc, char** argv)
{
    
    
    const fs::path config_path = "/root/FLASH/module/config/KITTI.yaml";
    const fs::path result_path = "/root/FLASH/module/src/result.txt";
    const fs::path cloud_dir   = "/root/FLASH/dataset/SemanticKITTI/00/velodyne";
    const fs::path label_dir   = "/root/FLASH/dataset/SemanticKITTI/00/labels";

    if (fs::exists(result_path)) 
    {
        fs::remove(result_path);
    }


    const int topk = 10;
    const int num_exclude_recent = 100;

    // FLASH instance generation
    std::cout << "\n";
    YAML::Node flash_cfg = YAML::LoadFile(config_path);
    std::unique_ptr<FLASH::Manager> flash = std::make_unique<FLASH::Manager>();
    flash->LoadParams(flash_cfg);
    auto params = flash->GetParams();
    

    auto cloud_list = get_file_list(cloud_dir.string());
    auto label_list = get_file_list(label_dir.string());
    int N = static_cast<int>(cloud_list.size());
    
    for (int i = 0; i < N; i++)
    {
        
        auto scan = read_cloud(cloud_list[i], label_list[i]);
        
        auto st1 = std::chrono::high_resolution_clock::now();
        flash->GenerateAndSave(scan);
        auto et1 = std::chrono::high_resolution_clock::now();

        auto st2 = std::chrono::high_resolution_clock::now();
        auto result = flash->OnlineLCD(topk, num_exclude_recent);
        auto et2 = std::chrono::high_resolution_clock::now();

        auto elapsed1 = ComputeElapsed(st1, et1);
        auto elapsed2 = ComputeElapsed(st2, et2);
        save_result(result_path, result);
        std::cout << "IDX: " << i << "\ttime1 : " << elapsed1  << "\ttime2 : " << elapsed2 << std::endl;

    }
    // flash->SaveWL();
    return 0;
}