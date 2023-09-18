mod post_processor;

use face_prediction::{
    arcface_predictor::ArcFacePredictor,
    get_file_paths_from_folder, process_file_paths,
    ultra_predictor::{self, UltraPredictor},
    Config,
};
use ort::OrtError;
use std::{
    env,
    fs::{self},
    path::Path,
    process,
    time::Instant,
};

fn main() -> Result<(), OrtError> {
    let start = Instant::now();
    let args: Vec<String> = env::args().collect();

    let config = Config::new(&args).unwrap_or_else(|err| {
        println!("Problem parsing arguments: {}", err);
        process::exit(1);
    });

    let ultra_model_path = Path::new(&config.ultra_model_path);
    let arc_face_model_path = Path::new(&config.arc_model_path);
    let folder_path = Path::new(&config.folder_path);
    let image_output_folder = Path::new(&config.result_folder);

    let ultra_predictor = UltraPredictor::new(ultra_model_path, 1).unwrap_or_else(|ort_err| {
        println!(
            "Problem creating ultra onnx session: {}",
            ort_err.to_string()
        );
        process::exit(1)
    });

    let face_arc_predictor =
        ArcFacePredictor::new(arc_face_model_path, 10).unwrap_or_else(|ort_err| {
            println!("Problem creating arc onnx session: {}", ort_err.to_string());
            process::exit(1)
        });

    fs::create_dir(image_output_folder)
        .unwrap_or_else(|err| println!("Unabel to create output dir: {}", err.to_string()));

    let file_paths = get_file_paths_from_folder(folder_path).unwrap_or_else(|err| {
        println!("Problem getting files from folder: {:?}", err.to_string());
        process::exit(1)
    });

    process_file_paths(
        &file_paths,
        &ultra_predictor,
        image_output_folder,
        &face_arc_predictor,
    )
    .unwrap_or_else(|err| println!("Problem processing folder: {:?}", err.to_string()));

    println!("\nTotal time elapsed: {:?}", start.elapsed());
    return Ok(());
}
