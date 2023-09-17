mod post_processor;

use face_prediction::{image_processor::UltraImage, ultra_predictor::UltraPredictor, Config};
use ort::OrtError;
use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
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

    process_directory(&folder_path, &predictor, &image_output_folder)
        .unwrap_or_else(|err| println!("Problem processing folder: {:?}", err.to_string()));

    return Ok(());
}

fn process_directory(
    dir_path: &Path,
    predictor: &UltraPredictor,
    image_output_folder: &Path,
) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(dir_path)? {
        let entry = entry?.path();
        if entry.is_file() {
            process_file(&entry, &predictor, &image_output_folder)?;
        } else {
            process_directory(&entry, &predictor, &image_output_folder)?;
        }
    }
    Ok(())
}

fn process_file(
    image_path: &PathBuf,
    predictor: &UltraPredictor,
    image_output_folder: &Path,
) -> Result<(), Box<dyn Error>> {
    println!(
        "\nProcessing file: {:?}",
        fs::canonicalize(image_path).expect("")
    );
    let mut start = Instant::now();
    let mut image = UltraImage::new(&image_path).unwrap_or_else(|err| {
        println!("Error when trying to open image: {}", err.to_string());
        process::exit(4);
    });
    println!("Image initialization took {:?}", start.elapsed());

    start = Instant::now();
    let output = predictor.run(&image.image)?;
    println!("Preprocessing and inference took {:?}", start.elapsed());

    start = Instant::now();
    image
        .draw_bboxes(output.bbox_with_confidences, &image_output_folder)
        .unwrap_or_else(|err| {
            println!("Error when trying to draw to image: {}", err.to_string());
            process::exit(1);
        });
    println!("Drawing bboxes and to file took {:?}", start.elapsed());
    Ok(())
}
