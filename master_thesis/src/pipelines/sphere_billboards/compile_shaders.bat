glslangvalidator -V billboards.vert -o billboards.vert.spv
glslangvalidator -V billboards.frag -o billboards.frag.spv
glslangvalidator -V -DOUTPUT_NORMALS billboards.frag -o billboards_normals.frag.spv

glslangvalidator -V billboards_depth.vert -o billboards_depth.vert.spv
glslangvalidator -V billboards_depth.frag -o billboards_depth.frag.spv
glslangvalidator -V -DWRITE_VISIBILITY billboards_depth.frag -o billboards_depth_write.frag.spv
