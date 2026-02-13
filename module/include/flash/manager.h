#pragma once


#include <iostream>
#include <vector>
#include <map>
#include <memory>

#include <Eigen/Dense>
#include <pcl/point_types.h>        
#include <pcl/point_cloud.h>        
#include <pcl/io/pcd_io.h>    
#include <yaml-cpp/yaml.h>


#include "flash/flash.h"
#include "flash/parameter.hpp"
#include "flash/external/nanoflann.hpp"
#include "flash/external/KDTreeVectorOfVectorsAdaptor.h"

using DescTree = KDTreeVectorOfVectorsAdaptor< std::vector<std::vector<float>>, float>;


namespace FLASH
{


    class Manager
    {
        private:
            Params _params;
            std::unique_ptr<Generator> _gen;

            std::vector<SemanticGraph> _graphs;
            std::vector<Descriptor> _descs;
            std::vector<std::vector<float>> _descs_tree;
            std::unique_ptr<DescTree> _tree;

        public:
            Manager();
            int Size();
            void LoadParams(const YAML::Node& cfg);
            void SaveWL();
            void LoadWL();
            Params GetParams();
            std::vector<Eigen::Vector3f> GetValidAnchor();

            SemanticGraph GetGraph(int idx);
            Descriptor GetDesc(int idx);
            std::pair<SemanticGraph, Descriptor> GetQuery();

            std::pair<SemanticGraph, Descriptor> Generate(const CloudPtr& cloud_in);
            void Save(const SemanticGraph& graph, const Descriptor& desc);
            void GenerateAndSave(const CloudPtr& cloud_in);
            
            std::vector<size_t> Retrieve(const Descriptor& desc, int k = 10);
            MatchingResult VerifyOne(int cand_idx, const SemanticGraph& qgraph, const Descriptor& qdesc) const;
            ResultSet Verify(int idx, const SemanticGraph& graph, const Descriptor& desc, const std::vector<size_t>& topk);
            ResultSet OnlineLCD(int k, int num_exclude_recent);
            
            ResultSet PR(const CloudPtr& cloud_in, int k);
            ResultSet PR(const SemanticGraph& graph, const Descriptor& desc, int k);

            float PairwiseMatching(const std::pair<SemanticGraph, Descriptor>& src, const std::pair<SemanticGraph, Descriptor>& dst);
    };

}; // namepsace FLASH


