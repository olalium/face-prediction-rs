use std::{path::Path, time::Instant};

use image::RgbImage;
use ndarray::{Array4, CowArray, IxDyn};
use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, OrtError, Session,
    SessionBuilder, Value,
};

use crate::post_processor::UltraOutput;

pub struct UltraPredictor {
    pub name: String,
    pub session: Session,
}

pub static ULTRA_PREDICTOR_NAME: &str = "UltraPredictor";

impl UltraPredictor {
    pub fn new(model_filepath: &Path, num_threads: i16) -> Result<UltraPredictor, OrtError> {
        let start = Instant::now();

        let environment = Environment::builder()
            .with_name(ULTRA_PREDICTOR_NAME.to_string())
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .with_log_level(LoggingLevel::Verbose)
            .build()?
            .into_arc();

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(num_threads)?
            .with_model_from_file(&model_filepath)?;

        println!(
            "{} startup took {:?}",
            ULTRA_PREDICTOR_NAME,
            start.elapsed()
        );
        Ok(UltraPredictor {
            name: ULTRA_PREDICTOR_NAME.to_string(),
            session,
        })
    }

    pub fn run(&self, image: &RgbImage) -> Result<UltraOutput, OrtError> {
        let start = Instant::now();

        let image_tensor = self.get_image_tensor(&image);
        let image_input = self.get_image_input(&image_tensor)?;
        let raw_outputs = self.session.run(image_input)?;
        let ultra_output = UltraOutput::new(raw_outputs)?;

        println!(
            "{} preprocessing and inference took {:?}",
            ULTRA_PREDICTOR_NAME,
            start.elapsed()
        );
        Ok(ultra_output)
    }

    fn get_image_tensor(&self, image: &RgbImage) -> CowArray<f32, IxDyn> {
        let image_tensor = CowArray::from(Array4::from_shape_fn(
            (1, 3, 480 as usize, 640 as usize),
            |(_, c, y, x)| {
                let mean = [0.485, 0.456, 0.406][c];
                let std = [0.229, 0.224, 0.225][c];
                (image[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
            },
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
