cargo build --target wasm32-unknown-unknown --bin occlusion

wasm-bindgen --out-dir target/generated --web target/wasm32-unknown-unknown/debug/occlusion.wasm