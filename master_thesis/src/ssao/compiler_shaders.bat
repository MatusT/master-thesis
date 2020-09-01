glslc CSPrepareNativeNormalsFromInputNormals.comp -o prepare_normals.comp.spv
glslc CSPrepareNativeDepthsAndMips.comp -o prepare_depths.comp.spv
glslc CSGenerateQ3.comp -g -O0 -o ssao.comp.spv
glslc CSEdgeSensitiveBlur.comp -o blur.comp.spv
glslc CSApply.comp -o apply.comp.spv
glslc normals_from_depth.comp -o normals_from_depth.comp.spv