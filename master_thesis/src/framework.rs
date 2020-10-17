use futures::task::LocalSpawn;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use winit::{
    event::{self, DeviceEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopProxy},
};

pub trait ApplicationStructure: 'static + Sized {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::default()
    }
    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self;
    fn resize(
        &mut self,
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    );
    fn window_event(&mut self, event: WindowEvent);
    fn device_event(&mut self, event: DeviceEvent);
    fn render(
        &mut self,
        frame: &wgpu::SwapChainTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spawner: &impl LocalSpawn,
    );
}

struct WindowSetup {
    window: winit::window::Window,
    event_loop: EventLoop<WgpuSetup>,
    event_loop_proxy: EventLoopProxy<WgpuSetup>,
    size: winit::dpi::PhysicalSize<u32>,
}

struct WgpuSetup {
    window: winit::window::Window,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn window_setup(title: &str) -> WindowSetup {
    // #[cfg(not(target_arch = "wasm32"))]
    // {
    //     let chrome_tracing_dir = std::env::var("WGPU_CHROME_TRACE");
    //     subscriber::initialize_default_subscriber(
    //         chrome_tracing_dir.as_ref().map(std::path::Path::new).ok(),
    //     );
    // };

    #[cfg(target_arch = "wasm32")]
    console_log::init().expect("could not initialize logger");

    let event_loop = EventLoop::<WgpuSetup>::with_user_event();
    let event_loop_proxy = event_loop.create_proxy();
    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title(title);
    builder = builder.with_inner_size(winit::dpi::PhysicalSize {
        width: 1920,
        height: 1080,
    });

    let window = builder.build(&event_loop).unwrap();
    let size = window.inner_size();

    #[cfg(target_arch = "wasm32")]
    {
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
    }

    WindowSetup {
        event_loop,
        event_loop_proxy,
        window,
        size,
    }
}

async fn wgpu_setup<E: ApplicationStructure>(
    window: winit::window::Window,
    event_loop_proxy: winit::event_loop::EventLoopProxy<WgpuSetup>,
) {
    env_logger::init();
    log::info!("Initializing the surface...");

    let instance = wgpu::Instance::new(wgpu::BackendBit::VULKAN);
    let surface = unsafe {
        let surface = instance.create_surface(&window);
        surface
    };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
        })
        .await
        .unwrap();

    let optional_features = E::optional_features();
    let required_features = E::required_features();
    let adapter_features = adapter.features();
    assert!(
        adapter_features.contains(required_features),
        "Adapter does not support required features for this example: {:?}",
        required_features - adapter_features
    );

    let needed_limits = E::required_limits();

    let trace_dir = std::env::var("WGPU_TRACE");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: (optional_features & adapter_features) | required_features,
                limits: needed_limits,
                shader_validation: false,
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .unwrap();

    let send_event_result = event_loop_proxy.send_event(WgpuSetup {
        window,
        surface,
        device,
        queue,
    });

    match send_event_result {
        Ok(_) => {}
        Err(_) => {
            log::warn!("Could not send event that wgpu has been initialized. Was the event loop destroyed right away?");
        }
    }
}

struct Application<E> {
    wgpu_setup: WgpuSetup,
    example: E,
    swap_chain: wgpu::SwapChain,
}

fn start<E: ApplicationStructure>(
    event_loop: EventLoop<WgpuSetup>,
    size: winit::dpi::PhysicalSize<u32>,
) {
    #[cfg(not(target_arch = "wasm32"))]
    let (mut pool, spawner) = {
        let local_pool = futures::executor::LocalPool::new();
        let spawner = local_pool.spawner();
        (local_pool, spawner)
    };

    #[cfg(target_arch = "wasm32")]
    let spawner = {
        use futures::{future::LocalFutureObj, task::SpawnError};

        struct WebSpawner {}
        impl LocalSpawn for WebSpawner {
            fn spawn_local_obj(
                &self,
                future: LocalFutureObj<'static, ()>,
            ) -> Result<(), SpawnError> {
                Ok(wasm_bindgen_futures::spawn_local(future))
            }
        }

        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        WebSpawner {}
    };

    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        // TODO: Allow srgb unconditionally
        format: if cfg!(target_arch = "wasm32") {
            wgpu::TextureFormat::Bgra8Unorm
        } else {
            wgpu::TextureFormat::Bgra8UnormSrgb
        },
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };
    #[cfg(not(target_arch = "wasm32"))]
    let mut last_update_inst = Instant::now();
    let mut application: Option<Application<E>> = None;

    log::info!("Entering render loop...");
    let mut start = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match &mut application {
            None => match event {
                event::Event::UserEvent(wgpu_setup) => {
                    let WgpuSetup {
                        device,
                        queue,
                        surface,
                        ..
                    } = &wgpu_setup;

                    let swap_chain = device.create_swap_chain(&surface, &sc_desc);

                    log::info!("Initializing the example...");
                    let example = E::init(&sc_desc, &device, &queue);
                    application = Some(Application {
                        example,
                        wgpu_setup,
                        swap_chain,
                    });
                }
                _ => {}
            },
            Some(Application {
                wgpu_setup:
                    WgpuSetup {
                        window,
                        device,
                        surface,
                        queue,
                        ..
                    },
                example,
                swap_chain,
            }) => match event {
                event::Event::MainEventsCleared => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        if last_update_inst.elapsed() > Duration::from_millis(20) {
                            window.request_redraw();
                            last_update_inst = Instant::now();
                        }

                        pool.run_until_stalled();
                    }

                    #[cfg(target_arch = "wasm32")]
                    window.request_redraw();
                }
                event::Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    log::info!("Resizing to {:?}", size);
                    sc_desc.width = size.width;
                    sc_desc.height = size.height;
                    example.resize(&sc_desc, &device, &queue);
                    *swap_chain = device.create_swap_chain(&surface, &sc_desc);
                }
                event::Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput {
                        input:
                            event::KeyboardInput {
                                virtual_keycode: Some(event::VirtualKeyCode::Escape),
                                state: event::ElementState::Pressed,
                                ..
                            },
                        ..
                    }
                    | WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {
                        example.window_event(event);
                    }
                },
                event::Event::DeviceEvent { event, .. } => {
                    example.device_event(event);
                }
                event::Event::RedrawRequested(_) => {
                    let frame = match swap_chain.get_current_frame() {
                        Ok(frame) => frame,
                        Err(_) => {
                            *swap_chain = device.create_swap_chain(&surface, &sc_desc);
                            swap_chain
                                .get_current_frame()
                                .expect("Failed to acquire next swap chain texture!")
                        }
                    };

                    example.render(&frame.output, &device, &queue, &spawner);

                    drop(frame);

                    // println!("Frame time: {}", start.elapsed().as_millis());
                    start = Instant::now();
                }
                _ => {}
            },
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run<E: ApplicationStructure>(title: &str) {
    let WindowSetup {
        window,
        event_loop,
        event_loop_proxy,
        size,
        ..
    } = window_setup(title);
    futures::executor::block_on(wgpu_setup::<E>(window, event_loop_proxy));
    start::<E>(event_loop, size);
}

#[cfg(target_arch = "wasm32")]
pub fn run<E: ApplicationStructure>(title: &str) {
    let title = title.to_owned();
    let WindowSetup {
        window,
        event_loop,
        event_loop_proxy,
        size,
        ..
    } = window_setup(&title);
    wasm_bindgen_futures::spawn_local(async move {
        wgpu_setup::<E>(window, event_loop_proxy).await;
    });
    start::<E>(event_loop, size);
}

// This allows treating the framework as a standalone example,
// thus avoiding listing the example names in `Cargo.toml`.
#[allow(dead_code)]
fn main() {}
