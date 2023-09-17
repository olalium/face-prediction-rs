use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use ultra_predictor::UltraPredictor;

use crate::image_processor::UltraImage;

pub mod image_processor;
pub mod post_processor;
pub mod ultra_predictor;

pub struct Config {
    pub model_path: String,
    pub folder_path: String,
    pub result_folder: String,
}

impl Config {
    pub fn new(args: &[String]) -> Result<Config, &str> {
        if args.len() < 4 {
            return Err("Not enough arguments");
        }

        let model_path = args[1].clone();
        let folder_path = args[2].clone();
        let result_folder = args[3].clone();

        Ok(Config {
            model_path,
            folder_path,
            result_folder,
        })
    }
}

pub fn process_folder(
    dir_path: &Path,
    predictor: &UltraPredictor,
    image_output_folder: &Path,
) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(dir_path)? {
        let entry = entry?.path();
        if entry.is_file() {
            process_file(&entry, &predictor, &image_output_folder)?;
        } else {
            process_folder(&entry, &predictor, &image_output_folder)?;
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
        process::exit(1);
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
