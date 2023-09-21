use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

use image::{imageops::FilterType, DynamicImage, ImageError, ImageFormat, Rgb, RgbImage};
use imageproc::{drawing::draw_hollow_rect, rect::Rect};

use crate::{
    post_processor::UltraResult,
    ultra_predictor::{ULTRA_INPUT_HEIGHT, ULTRA_INPUT_WIDTH},
};

pub struct UltraImage<'a> {
    pub image: RgbImage,
    pub raw_image: DynamicImage,
    pub image_path: &'a Path,
}

impl UltraImage<'_> {
    pub fn new(path: &Path) -> Result<UltraImage, ImageError> {
        let start = Instant::now();

        let raw_image = image::open(path)?;
        let image = raw_image
            .resize_to_fill(
                ULTRA_INPUT_WIDTH as u32,
                ULTRA_INPUT_HEIGHT as u32,
                FilterType::Triangle,
            )
            .to_rgb8();

        println!(
            "Image initialization of {:?} took {:?}",
            &path,
            start.elapsed()
        );

        return Ok(UltraImage {
            raw_image,
            image,
            image_path: path,
        });
    }

    pub fn draw_bboxes(
        &mut self,
        bbox_with_confidences: UltraResult,
        output_folder: &Path,
    ) -> Result<(), ImageError> {
        let start = Instant::now();
        self.image = draw_bboxes_on_image(
            self.image.clone(),
            bbox_with_confidences,
            ULTRA_INPUT_WIDTH as u32,
            ULTRA_INPUT_HEIGHT as u32,
        );

        let mut output_path = PathBuf::from(output_folder);
        let file_name = PathBuf::from(&self.image_path.file_name().expect("file_name not found"));
        output_path.push(file_name);

        File::create(&output_path)?;
        image::save_buffer_with_format(
            &output_path,
            &self.image,
            ULTRA_INPUT_WIDTH as u32,
            ULTRA_INPUT_HEIGHT as u32,
            image::ColorType::Rgb8,
            ImageFormat::Jpeg,
        )?;
        println!("Drawing bboxes and to file took {:?}", start.elapsed());
        Ok(())
    }
}

/// Draw bounding boxes with confidence scores on the image.
fn draw_bboxes_on_image(
    mut frame: RgbImage,
    bboxes_with_confidences: Vec<([f32; 4], f32)>,
    width: u32,
    height: u32,
) -> RgbImage {
    let (width, height) = (width as f32, height as f32);

    for (bbox, _) in bboxes_with_confidences.iter() {
        // Coordinates of top-left and bottom-right points
        // Coordinate frame basis is on the top left corner
        let (x_tl, y_tl) = (bbox[0] * width, bbox[1] * height);
        let (x_br, y_br) = (bbox[2] * width, bbox[3] * height);
        let rect_width = x_br - x_tl;
        let rect_height = y_br - y_tl;

        let face_rect =
            Rect::at(x_tl as i32, y_tl as i32).of_size(rect_width as u32, rect_height as u32);

        frame = draw_hollow_rect(&frame, face_rect, Rgb::from([0, 255, 0]));
    }

    frame
}
