use image::{
    imageops::{crop_imm, FilterType},
    DynamicImage, ImageError, RgbImage,
};

use crate::post_processor::Bbox;

pub struct ArcFaceImage {
    pub image: RgbImage,
}

impl ArcFaceImage {
    pub fn new(image: RgbImage, bbox: Bbox) -> Result<ArcFaceImage, ImageError> {
        let cropped_image = crop_image(image, bbox, 640, 480)?;

        let image = DynamicImage::from(cropped_image)
            .resize_to_fill(128, 128, FilterType::Triangle)
            .to_rgb8();

        return Ok(ArcFaceImage { image });
    }
}

fn crop_image(
    mut image: RgbImage,
    bbox: Bbox,
    width: u32,
    height: u32,
) -> Result<RgbImage, ImageError> {
    let (width, height) = (width as f32, height as f32);

    // Coordinates of top-left and bottom-right points
    // Coordinate frame basis is on the top left corner
    let (x_tl, y_tl) = (bbox[0] * width, bbox[1] * height);
    let (x_br, y_br) = (bbox[2] * width, bbox[3] * height);
    let rect_width = x_br - x_tl;
    let rect_height = y_br - y_tl;

    let sub_image = crop_imm(
        &mut image,
        x_tl as u32,
        y_tl as u32,
        rect_width as u32,
        rect_height as u32,
    );
    Ok(sub_image.to_image())
}
