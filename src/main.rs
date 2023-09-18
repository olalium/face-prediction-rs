mod post_processor;

use face_prediction::{process_folder, ultra_predictor::UltraPredictor, Config};
use ort::OrtError;
use std::{
    env,
    fs::{self},
    path::Path,
    process,
    time::Instant,
};

fn main() -> Result<(), OrtError> {
    let args: Vec<String> = env::args().collect();

    let config = Config::new(&args).unwrap_or_else(|err| {
        println!("Problem parsing arguments: {}", err);
        process::exit(1);
    });

    let model_path = Path::new(&config.model_path);
    let folder_path = Path::new(&config.folder_path);
    let image_output_folder = Path::new(&config.result_folder);

    let start = Instant::now();
    let predictor = UltraPredictor::new(&model_path, 1).unwrap_or_else(|ort_err| {
        println!("Problem creating onnx session: {}", ort_err.to_string());
        process::exit(1)
    });
    println!("Prediction startup took {:?}", start.elapsed());

    fs::create_dir(&image_output_folder)
        .unwrap_or_else(|err| println!("Unabel to create output dir: {}", err.to_string()));

    process_folder(&folder_path, &predictor, &image_output_folder)
        .unwrap_or_else(|err| println!("Problem processing folder: {:?}", err.to_string()));

    println!("\nTotal time elapsed: {:?}", start.elapsed());
    return Ok(());
}
