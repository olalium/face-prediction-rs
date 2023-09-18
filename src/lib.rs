use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process,
};

use ultra_predictor::UltraPredictor;

use crate::image_processor::UltraImage;
use rayon::prelude::*;

pub mod image_processor;
pub mod post_processor;
pub mod ultra_predictor;

pub struct Config {
    pub model_path: String,
    pub folder_path: String,
    pub result_folder: String,
}

const CHUNK_SIZE: usize = 10;

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

pub fn get_file_paths_from_folder(dir_path: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut file_paths: Vec<PathBuf> = vec![];
    for dir_entry in fs::read_dir(dir_path)? {
        let entry = dir_entry?.path();
        if entry.is_file() {
            file_paths.push(entry);
        } else {
            file_paths.extend(get_file_paths_from_folder(&entry)?);
        }
    }
    return Ok(file_paths);
}

pub fn process_file_paths(
    file_paths: &Vec<PathBuf>,
    predictor: &UltraPredictor,
    image_output_folder: &Path,
) -> Result<(), Box<dyn Error>> {
    for file_paths in file_paths.chunks(CHUNK_SIZE) {
        let images = par_get_images(file_paths);
        process_images(images, &predictor, &image_output_folder);
    }
    Ok(())
}

fn par_get_images(file_paths: &[PathBuf]) -> Vec<UltraImage> {
    file_paths
        .into_par_iter()
        .filter_map(|file_path| match UltraImage::new(file_path) {
            Ok(image) => Some(image),
            Err(error) => {
                println!(
                    "Unable to initalize file: {:?}, because of {}",
                    &file_path,
                    error.to_string()
                );
                return None;
            }
        })
        .collect()
}

fn process_images(images: Vec<UltraImage>, predictor: &UltraPredictor, image_output_folder: &Path) {
    images.into_iter().for_each(|mut image| {
        process_image(&mut image, &predictor, &image_output_folder).unwrap();
    });
}

fn process_image(
    image: &mut UltraImage,
    predictor: &UltraPredictor,
    image_output_folder: &Path,
) -> Result<(), Box<dyn Error>> {
    let output = predictor.run(&image.image)?;
    image
        .draw_bboxes(output.bbox_with_confidences, image_output_folder)
        .unwrap_or_else(|err| {
            println!("Error when trying to draw to image: {}", err.to_string());
            process::exit(1);
        });
    Ok(())
}
