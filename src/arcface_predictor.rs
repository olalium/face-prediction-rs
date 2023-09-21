use std::{path::Path, process, time::Instant};

use image::RgbImage;
use ndarray::{Array4, CowArray, IxDyn};
use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, OrtError, Session,
    SessionBuilder, Value,
};

use crate::{
    arcface_image::ArcFaceImage,
    post_processor::{ArcFaceOutput, UltraResult},
    ultra_image::UltraImage,
};

pub struct ArcFacePredictor {
    pub name: String,
    pub session: Session,
}

pub static ARC_FACE_NAME: &str = "ArcFacePredictor";

impl ArcFacePredictor {
    pub fn new(model_filepath: &Path, num_threads: i16) -> Result<ArcFacePredictor, OrtError> {
        let start = Instant::now();

        let environment = Environment::builder()
            .with_name(ARC_FACE_NAME.to_string())
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .with_log_level(LoggingLevel::Verbose)
            .build()?
            .into_arc();

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(num_threads)?
            .with_model_from_file(&model_filepath)?;

        println!("{} startup took {:?}", ARC_FACE_NAME, start.elapsed());
        Ok(ArcFacePredictor {
            name: ARC_FACE_NAME.to_string(),
            session,
        })
    }

    pub fn run(
        &self,
        ultra_image: &UltraImage,
        bboxes: &UltraResult,
    ) -> Result<Vec<ArcFaceOutput>, OrtError> {
        let start = Instant::now();
        let mut arc_face_outputs: Vec<ArcFaceOutput> = vec![];

        for (bbox, _) in bboxes {
            let image = ArcFaceImage::new(ultra_image.raw_image.clone(), bbox.clone())
                .expect("something went wrong");
            let image_tensor = self.get_image_tensor(&image.image);
            let image_input = self.get_image_input(&image_tensor)?;
            let raw_outputs = self.session.run(image_input).unwrap_or_else(|err| {
                println!("somehting went wrong running session: {}", err);
                process::exit(1)
            });
            arc_face_outputs.push(ArcFaceOutput::new(raw_outputs)?);
        }

        println!(
            "{} preprocessing and inference took {:?}",
            ARC_FACE_NAME,
            start.elapsed()
        );
        Ok(arc_face_outputs)
    }

    fn get_image_tensor(&self, image: &RgbImage) -> CowArray<f32, IxDyn> {
        let image_tensor = CowArray::from(Array4::from_shape_fn(
            (1, 3, 112 as usize, 112 as usize),
            |(_, c, y, x)| ((image[(x as _, y as _)][c] as f32 / 255.0) - 0.5) / 0.5,
        ))
        .into_dyn();

        return image_tensor;
    }

    fn get_image_input<'a>(
        &self,
        image_tensor: &'a CowArray<'a, f32, IxDyn>,
    ) -> Result<Vec<Value<'a>>, OrtError> {
        let input_value = Value::from_array(self.session.allocator(), &image_tensor)?;
        let input = vec![input_value];

        return Ok(input);
    }
}
