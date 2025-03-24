#!/bin/bash

name=eval
stl_dir="${name}/meshes/peg/"

pcd_dir="pcds/${name}_pcd"
feat_dir="features/${name}_feat"

# generate the point cloud data from the stl files
python3 stl2pc.py $stl_dir -d -o $pcd_dir

# generate the 3D features
python3 gen_uni3d_feat.py $pcd_dir -d -o $feat_dir # -viz
