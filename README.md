# face-prediction-rs
Face prediction CLI using rust ORT library and the UltraFace model. Iterates image folder `[image_folder]` and compares faces found to `[test_case_path]`

# Setup
1. download the [Ultra face](https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface) 640 onnx model and put it in `[ultra_model_path]`
1. download the [Arc face](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface/model) arcfaceresnet100-11-int8.onnx model and put it in `[arc_model_path]`
2. add images to `[image_folder]`
3. run `cargo build --release`
4. run `./target/release/face-prediction [ultra_model_path] [arc_model_path] [image_folder] [output_dir] [test_case_path]`
