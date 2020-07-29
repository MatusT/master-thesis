use std::borrow::Cow::Borrowed;
use wgpu::*;
pub struct LinesPipeline {
    pub pipeline: RenderPipeline,
}

impl LinesPipeline {
    pub fn new(
        device: &Device,
        camera_bind_group_layout: &BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        // Shaders
        let vs_module = device.create_shader_module(include_spirv!("lines.vert.spv"));
        let fs_module = device.create_shader_module(include_spirv!("lines.frag.spv"));

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: Borrowed(&[&camera_bind_group_layout]),
            push_constant_ranges: Borrowed(&[]),
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: Borrowed("main"),
            },
            fragment_stage: Some(ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: Borrowed("main"),
            }),
            rasterization_state: Some(RasterizationStateDescriptor {
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: PrimitiveTopology::LineList,
            color_states: Borrowed(&[ColorStateDescriptor {
                format: TextureFormat::Bgra8UnormSrgb,
                color_blend: BlendDescriptor::REPLACE,
                alpha_blend: BlendDescriptor::REPLACE,
                write_mask: ColorWrite::ALL,
            }]),
            depth_stencil_state: Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil_front: StencilStateFaceDescriptor::IGNORE,
                stencil_back: StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            vertex_state: VertexStateDescriptor {
                index_format: IndexFormat::Uint16,
                vertex_buffers: Borrowed(&[VertexBufferDescriptor {
                    step_mode: InputStepMode::Vertex,
                    stride: 12,
                    attributes: Borrowed(&wgpu::vertex_attr_array![0 => Float3]),
                }]),
            },
            sample_count,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self { pipeline }
    }
}
