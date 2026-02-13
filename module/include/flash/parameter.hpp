#pragma once

#include <vector>
#include <map>

namespace FLASH
{

    struct Params
    {
        
        float fov_left = -180.0; // unit degree  (x axis 0 degree)
        float fov_right = 180.0; // unit degree  (x axis 0 degree)
        float fov_up =  2.0;      // unit degree 
        float fov_down = -24.8;   // unit degree 

        float leaf_size = 0.2;
        float edge_th = 200.0;
        int num_hop = 3;
        int num_anchor = 4096;
        int l = 10;


        int inlier_th = 20;
        float similarity_th = 0.97;

        bool use_precomputed_wl_map = false;
        std::string precomputed_wl_map_path = "./wl.yaml";

        /* semantic kitti  */
        // std::map<int, std::string>
        std::map<int, std::string> foreground_label = {
            {10, "car"},
            {20, "other-vehicle"},
            {50, "building"},
            {51, "fence"},
            {70, "vegetation"},
            {71, "trunk"},
            {72, "terrain"},
            {80, "pole"},
            {81, "traffic-sign"}
        };

        std::map<int, std::string> background_label = {
            {40, "road"},
            {48, "sidewalk"},
            {49, "other-ground"},
            {50, "building"},
            {51, "fence"},
            {72, "terrain"}
        };

        // eps_per_label: std::map<int, double>
        std::map<int, float> eps_per_label = {
            {10, 1.0},
            {20, 1.0},
            {48, 1.0},
            {50, 1.0},
            {51, 1.0},
            {70, 1.0},
            {71, 1.0},
            {72, 1.0},
            {80, 1.0},
            {81, 1.0}
        };

        // min_pts_per_label: std::map<int, int>
        std::map<int, int> min_pts_per_label = {
            {10, 30},
            {20, 30},
            {48, 50},
            {50, 50},
            {51, 20},
            {70, 50},
            {71, 10},
            {72, 80},
            {80, 10},
            {81, 10}
        };

        // color_map: std::map<int, std::array<int, 3>>
        std::map<int, std::array<int, 3>> color_map = {
            {0, {0, 0, 0}}, {1, {255, 0, 0}}, {10, {100, 150, 245}},
            {11, {100, 230, 245}}, {13, {100, 80, 250}}, {15, {30, 60, 150}},
            {16, {0, 0, 255}}, {18, {80, 30, 180}}, {20, {0, 0, 255}},
            {30, {255, 30, 30}}, {31, {255, 40, 200}}, {32, {150, 30, 90}},
            {40, {255, 0, 255}}, {44, {255, 150, 255}}, {48, {75, 0, 75}},
            {49, {175, 0, 75}}, {50, {255, 200, 0}}, {51, {255, 120, 50}},
            {52, {255, 150, 0}}, {60, {150, 255, 170}}, {70, {0, 175, 0}},
            {71, {135, 60, 0}}, {72, {150, 240, 80}}, {80, {255, 240, 150}},
            {81, {255, 0, 0}}, {99, {50, 255, 255}}, {252, {100, 150, 245}},
            {256, {0, 0, 255}}, {253, {255, 40, 200}}, {254, {255, 30, 30}},
            {255, {150, 30, 90}}, {257, {100, 80, 250}}, {258, {80, 30, 180}},
            {259, {0, 0, 255}}
        };
    

        std::map<int, int> foreground_label_to_index = CreateLabelToIndexMap(foreground_label);
        std::map<int, int> background_label_to_index = CreateLabelToIndexMap(background_label);


        std::map<int, int> CreateLabelToIndexMap(const std::map<int, std::string>& label)
        {
            std::map<int, int> label_to_index;
            int idx = 0;
            for (const auto& pair : label)
            {
                label_to_index[pair.first] = idx++;
            }
            return label_to_index;
        }

        void Clear()
        {
            eps_per_label.clear();
            min_pts_per_label.clear();
            foreground_label.clear();
            background_label.clear();
            color_map.clear();
        }

        void Update(const Params& other)
        {
            fov_left =  other.fov_left;
            fov_right =  other.fov_right;
            fov_down = other.fov_down;
            fov_up =  other.fov_up;

            leaf_size = other.leaf_size;
            edge_th = other.edge_th;
            num_hop = other.num_hop;
            num_anchor = other.num_anchor;
            l = other.l;
            
            inlier_th = other.inlier_th;
            similarity_th = other.similarity_th;
            // desc_type = other.desc_type;

            use_precomputed_wl_map = other.use_precomputed_wl_map;
            precomputed_wl_map_path = other.precomputed_wl_map_path;

            foreground_label = other.foreground_label;
            background_label = other.background_label;
            eps_per_label = other.eps_per_label;
            min_pts_per_label = other.min_pts_per_label;
            color_map = other.color_map;

            foreground_label_to_index = CreateLabelToIndexMap(foreground_label);
            background_label_to_index = CreateLabelToIndexMap(background_label);
        }
    
    };
}; // namespace FLASH