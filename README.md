# face-prediction-rs
Face prediction using rust ORT library and the UltraFace model. Iterates input folder and draws bounding boxes on all images within file

# Setup
1. download the [Ultra face](https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface) 640 onnx model and put it in `[model_path]`
2. add images to `[image_dir]`
3. run `cargo run [model_path] [image_dir] [output_dir]`
