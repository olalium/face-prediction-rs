use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use arcface_predictor::ArcFacePredictor;
use ndarray::Array;
use post_processor::ArcFaceOutput;
use ultra_image::UltraImage;
use ultra_predictor::UltraPredictor;

use rayon::prelude::*;

pub mod arcface_image;
pub mod arcface_predictor;
pub mod post_processor;
pub mod ultra_image;
pub mod ultra_predictor;

pub struct Config {
    pub ultra_model_path: String,
    pub arc_model_path: String,
    pub folder_path: String,
    pub result_folder: String,
}

const CHUNK_SIZE: usize = 10;

impl Config {
    pub fn new(args: &[String]) -> Result<Config, &str> {
        if args.len() < 5 {
            return Err("Not enough arguments");
        }

        let ultra_model_path = args[1].clone();
        let arc_model_path = args[2].clone();
        let folder_path = args[3].clone();
        let result_folder = args[4].clone();

        Ok(Config {
            ultra_model_path,
            arc_model_path,
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
    arc_predictor: &ArcFacePredictor,
) -> Result<(), Box<dyn Error>> {
    for file_paths in file_paths.chunks(CHUNK_SIZE) {
        let images = par_get_images(file_paths);
        process_images(images, &predictor, &image_output_folder, &arc_predictor);
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

//TODO my man
fn process_images(
    images: Vec<UltraImage>,
    predictor: &UltraPredictor,
    image_output_folder: &Path,
    arc_predictor: &ArcFacePredictor,
) {
    let mut output_vec: Vec<ArcFaceOutput> = vec![];
    images.into_iter().for_each(|mut image| {
        let arc_output =
            process_image(&mut image, &predictor, &image_output_folder, &arc_predictor).unwrap();
        output_vec.extend(arc_output);
    });

    let mut embeddings: Vec<Vec<f32>> = vec![];
    for output in output_vec {
        embeddings.push(output.embedding);
    }

    for embedding in embeddings.clone() {
        let mut vec0 = Array::from(embedding);
        let norm_vec0 = (vec0.mapv(|x| x * x).sum()).sqrt();
        vec0 /= norm_vec0;
        embeddings.as_slice().iter().for_each(|in_embedding| {
            let mut vec1 = Array::from(in_embedding.clone());
            let norm_vec1 = (vec1.mapv(|x| x * x).sum()).sqrt();
            vec1 /= norm_vec1;

            let vecx = (vec0.clone() - vec1).mapv(|v| v * v).sum();
            println!("ditance: {:#?}", vecx);
        });
    }
}

fn process_image(
    image: &mut UltraImage,
    predictor: &UltraPredictor,
    image_output_folder: &Path,
    arc_predictor: &ArcFacePredictor,
) -> Result<Vec<ArcFaceOutput>, Box<dyn Error>> {
    let output = predictor.run(&image.image)?;
    let embeddings = arc_predictor.run(&image.image, output.bbox_with_confidences)?;
    // image
    //     .draw_bboxes(output.bbox_with_confidences, image_output_folder)
    //     .unwrap_or_else(|err| {
    //         println!("Error when trying to draw to image: {}", err.to_string());
    //         process::exit(1);
    //     });
    Ok(embeddings)
}
