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
