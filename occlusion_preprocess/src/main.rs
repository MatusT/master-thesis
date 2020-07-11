mod application;
mod biological_structure;
mod camera;
mod hilbert;
mod pipelines;

use wgpu::*;
use winit::{
    event::{DeviceEvent, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub enum ApplicationEvent<'a> {
    WindowEvent(WindowEvent<'a>),
    DeviceEvent(DeviceEvent),
}

fn run(event_loop: EventLoop<()>, window: Window, swapchain_format: TextureFormat) {
    let size = window.inner_size();
    let instance = Instance::new(BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = futures::executor::block_on(instance.request_adapter(
        &RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
        },
        UnsafeExtensions::disallow(),
    ))
    .unwrap();

    let (device, queue) = futures::executor::block_on(adapter.request_device(
        &DeviceDescriptor {
            extensions: Extensions::empty(),
            limits: Limits::default(),
            shader_validation: false,
        },
        None,
    ))
    .unwrap();

    // Initialize the graphics scene
    let mut application =
        application::Application::new(size.width, size.height, device, queue, swapchain_format, 1);

    // Initialize swapchain
    let mut sc_desc = SwapChainDescriptor {
        usage: TextureUsage::OUTPUT_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: PresentMode::Mailbox,
    };
    let mut swap_chain = application.device().create_swap_chain(&surface, &sc_desc);

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        let _ = (
            &instance,
            &adapter,
            &surface,
            &swap_chain,
            &application,
        );

        *control_flow = ControlFlow::Poll;
        match event {
            // Process all the events
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            // Handle resize event as a special case
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                sc_desc.width = size.width;
                sc_desc.height = size.height;

                swap_chain = application.device().create_swap_chain(&surface, &sc_desc);

                application.resize(sc_desc.width, sc_desc.height);
            }
            Event::RedrawRequested(_) => {
                let frame = swap_chain.get_next_frame().unwrap();

                application.render(&frame.output.view);
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            // Gather window + device events
            Event::WindowEvent { event, .. } => {
                application.update(&ApplicationEvent::WindowEvent(event));
            }
            Event::DeviceEvent { event, .. } => {
                application.update(&ApplicationEvent::DeviceEvent(event));
            }
            _ => {}
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();
    window.set_inner_size(winit::dpi::PhysicalSize {
        width: 1920,
        height: 1080,
    });

    #[cfg(not(target_arch = "wasm32"))]
    {
        // env_logger::init();
        // Temporarily avoid srgb formats for the swapchain on the web
        run(event_loop, window, TextureFormat::Bgra8UnormSrgb);
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window, TextureFormat::Bgra8Unorm));
    }
}
