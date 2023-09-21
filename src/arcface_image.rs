use image::{
    imageops::{crop_imm, FilterType},
    DynamicImage, ImageError, RgbImage,
};

use crate::{
    post_processor::Bbox,
    ultra_predictor::{ULTRA_INPUT_HEIGHT, ULTRA_INPUT_WIDTH},
};

pub struct ArcFaceImage {
    pub image: RgbImage,
}

impl ArcFaceImage {
    pub fn new(raw_image: DynamicImage, bbox: Bbox) -> Result<ArcFaceImage, ImageError> {
        let cropped_raw_image = crop_ultra_image_raw(raw_image, bbox)?;
        let raw_image = DynamicImage::from(cropped_raw_image)
            .resize_to_fill(128, 128, FilterType::Triangle)
            .to_rgb8();

        return Ok(ArcFaceImage { image: raw_image });
    }
}

// crop image returning sub_image, assumes bbox is found of ULTRA_INPUT_WIDTH x ULTRA_INPUT_HEIGHT version of image
fn crop_ultra_image_raw(image: DynamicImage, bbox: Bbox) -> Result<RgbImage, ImageError> {
    let width: f32 = image.width() as f32;
    let height: f32 = image.height() as f32;

    let aspect_ratio_raw_image = width / height;
    let aspect_ratio_ultra = ULTRA_INPUT_WIDTH as f32 / ULTRA_INPUT_HEIGHT as f32;

    let (x_tl, y_tl, x_br, y_br): (f32, f32, f32, f32) =
        if aspect_ratio_raw_image > aspect_ratio_ultra {
            let scaled_width = aspect_ratio_ultra * height;
            let offset = (width - scaled_width) / 2.0;
            (
                bbox[0] * scaled_width + offset,
                bbox[1] * height,
                bbox[2] * scaled_width + offset,
                bbox[3] * height,
            )
        } else if aspect_ratio_raw_image < aspect_ratio_ultra {
            let scaled_height = (1.0 / aspect_ratio_ultra) * width;
            let offset = (height - scaled_height) / 2.0;
            (
                bbox[0] * width,
                bbox[1] * scaled_height + offset,
                bbox[2] * width,
                bbox[3] * scaled_height + offset,
            )
        } else {
            // raw_image has same aspect ratio
            (
                bbox[0] * width,
                bbox[1] * height,
                bbox[2] * width,
                bbox[3] * height,
            )
        };

    let rect_width = x_br - x_tl;
    let rect_height = y_br - y_tl;

    let mut binding = image.to_rgb8();
    let sub_image = crop_imm(
        &mut binding,
        x_tl as u32,
        y_tl as u32,
        rect_width as u32,
        rect_height as u32,
    );
    Ok(sub_image.to_image())
}
