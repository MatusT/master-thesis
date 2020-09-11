glslangvalidator -V spheres.vert -o spheres.vert.spv
glslangvalidator -DLESS -V spheres.frag -o spheres_less.frag.spv
glslangvalidator -DGREATER -V spheres.frag -o spheres_greater.frag.spv
glslangvalidator -DEARLY -DLESS -V spheres.frag -o spheres_early_less.frag.spv
glslangvalidator -DEARLY -DGREATER -V spheres.frag -o spheres_early_greater.frag.spv

