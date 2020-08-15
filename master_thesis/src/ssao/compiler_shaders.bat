glslc CSPrepareNativeNormalsFromInputNormals.comp -o prepare_normals.comp.spv
glslc CSPrepareNativeDepthsAndMips.comp -o prepare_depths.comp.spv
glslc CSGenerateQ3.comp -g -O0 -o ssao.comp.spv