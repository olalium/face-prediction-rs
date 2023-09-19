use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use arcface_predictor::ArcFacePredictor;
use ndarray::{Array, Array1};
use post_processor::{ArcFaceOutput, UltraOutput};
use ultra_image::UltraImage;
use ultra_predictor::UltraPredictor;

use rayon::prelude::*;

pub mod arcface_image;
pub mod arcface_predictor;
pub mod config;
pub mod post_processor;
pub mod ultra_image;
pub mod ultra_predictor;

static CHUNK_SIZE: usize = 10;

pub fn process_file_path<'a>(
    file_path: &'a Path,
    ultra_predictor: &UltraPredictor,
    arc_predictor: &ArcFacePredictor,
) -> Result<(&'a Path, Vec<f32>), Box<dyn Error>> {
    let ultra_image = UltraImage::new(file_path)?;
    let ultra_output = &ultra_predictor.run(&ultra_image.image)?;
    let arc_face_output =
        &arc_predictor.run(&ultra_image.image, &ultra_output.bbox_with_confidences)?;
    let normalized_embedding = normalize_embedding(arc_face_output[0].embedding.clone());
    Ok((&ultra_image.image_path, normalized_embedding))
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

pub fn process_file_paths<'a>(
    file_paths: &'a Vec<PathBuf>,
    ultra_predictor: &'a UltraPredictor,
    // image_output_folder: &Path,
    arc_predictor: &'a ArcFacePredictor,
) -> Vec<(&'a Path, Vec<Vec<f32>>)> {
    let mut images_with_embedding_result: Vec<(&Path, Vec<Vec<f32>>)> = vec![];
    for file_paths in file_paths.chunks(CHUNK_SIZE) {
        let images = par_get_ultra_images(file_paths);
        let ultra_outputs = run_ultra_prediciton(&images, &ultra_predictor);
        let images_with_arc_face_outputs =
            run_arc_face_prediction(ultra_outputs, images, &arc_predictor);
        let images_with_embeddings = calculate_embeddings(images_with_arc_face_outputs);
        images_with_embedding_result.extend(images_with_embeddings)
        // draw_boxes(
        //     images,
        //     &ultra_predictor,
        //     &image_output_folder,
        // );
    }
    return images_with_embedding_result;
}

fn par_get_ultra_images(file_paths: &[PathBuf]) -> Vec<UltraImage> {
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

fn run_ultra_prediciton(
    ultra_images: &Vec<UltraImage>,
    predictor: &UltraPredictor,
) -> Vec<UltraOutput> {
    ultra_images
        .into_iter()
        .filter_map(|ultra_image| {
            let ultra_output = predictor.run(&ultra_image.image);
            match ultra_output {
                Ok(ultra_output) => Some(ultra_output),
                Err(error) => {
                    println!("Unable to get run result because of {}", error.to_string());
                    return None;
                }
            }
        })
        .collect()
}

fn run_arc_face_prediction<'a>(
    ultra_outputs: Vec<UltraOutput>,
    images: Vec<UltraImage<'a>>,
    predictor: &ArcFacePredictor,
) -> Vec<(UltraImage<'a>, Vec<ArcFaceOutput>)> {
    images
        .into_iter()
        .zip(ultra_outputs.into_iter())
        .filter_map(|(image, ultra_output)| {
            let arc_output = predictor.run(&image.image, &ultra_output.bbox_with_confidences);
            match arc_output {
                Ok(arc_output) => Some((image, arc_output)),
                Err(error) => {
                    println!("Unable to get run result because of {}", error.to_string());
                    return None;
                }
            }
        })
        .collect()
}

fn calculate_embeddings<'a>(
    images_with_arc_face_outputs: Vec<(UltraImage<'a>, Vec<ArcFaceOutput>)>,
) -> Vec<(&Path, Vec<Vec<f32>>)> {
    images_with_arc_face_outputs
        .into_iter()
        .map(|(image, arc_face_outputs)| {
            let embeddings: Vec<Vec<f32>> = arc_face_outputs
                .into_iter()
                .map(|output| normalize_embedding(output.embedding))
                .collect();
            (image.image_path, embeddings)
        })
        .collect()
}

fn normalize_embedding(embedding: Vec<f32>) -> Vec<f32> {
    let embedding = Array::from(embedding);
    let l2_norm = f32::sqrt(embedding.mapv(|v| v * v).sum());
    let normalized_embedding = embedding.mapv(|v| v / l2_norm);
    normalized_embedding.to_vec()
}

pub fn calculate_distances(
    compare_embeddings: Vec<f32>,
    images_with_embeddings: Vec<(&Path, Vec<Vec<f32>>)>,
) -> Vec<(String, f32)> {
    let mut path_with_dist: Vec<(String, f32)> = vec![];
    let compare_embeddings_arr = Array::from(compare_embeddings);

    for (image_path, image_embeddings) in images_with_embeddings {
        let readable_path = image_path
            .to_path_buf()
            .into_os_string()
            .into_string()
            .unwrap();
        let mut lowest_dist: f32 = 100.0;

        for image_embedding in image_embeddings {
            let image_embedding_arr = Array::from(image_embedding);
            let dist = calculate_distance(image_embedding_arr, compare_embeddings_arr.clone());
            if dist < lowest_dist {
                lowest_dist = dist;
            }
        }
        path_with_dist.push((readable_path, lowest_dist))
    }
    return path_with_dist;
}

fn calculate_distance(arr_a: Array1<f32>, arr_b: Array1<f32>) -> f32 {
    let sub_res = &arr_a - &arr_b;
    let sqr_res = sub_res.mapv(|v| v * v);
    return sqr_res.sum();
}
// fn draw_boxes(
//     image: &mut UltraImage,
//     predictor: &UltraPredictor,
//     image_output_folder: &Path,
// ) -> Result<(), Box<dyn Error>> {
//     let output = predictor.run(&image.image)?;
//     image
//         .draw_bboxes(output.bbox_with_confidences, image_output_folder)
//         .unwrap_or_else(|err| {
//             println!("Error when trying to draw to image: {}", err.to_string());
//             process::exit(1);
//         });
//     Ok(())
// }
