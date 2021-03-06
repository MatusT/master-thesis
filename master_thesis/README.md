# Main example

You can run the main example with the following command:

```
cargo run --release --bin occlusion list_of_object_files
```

Replace **list_of_object_files** with a space delimited list of files (in .ron format)

Example provided:
```
cargo run --release --bin occlusion ..\data\SARS-CoV-2\SARS-CoV-2.ron
cargo run --release --bin occlusion ..\data\RandomSpheres\sphere.ron
```

Controls:
- Arrow Up/Down - move up/down in a list of modifiable SSAO parameters
- 1/2 - switch between modification of SSAO Far(1) and Near(2)
- +/- - increase/decrease Fog/SSAO parameter currently being modified 
- C - reload colors from a file **colors.ron**
- S - switch between final view/SSAO Far/SSAO near
- F - modify fog distance
- W - turn on printing frame time to console

Example output after modification:
```
Ssao settings Near
[ ] Shadow multiplier: 4.7999973
[ ] Shadow power: 3.099999
[ ] Horizon angle threshold: 0
[ ] Sharpness: 0
[ ] Detail shadow strength: 3
[*] Radius: 56
[ ] Fog: 10000.0
```

# Patches

TODO