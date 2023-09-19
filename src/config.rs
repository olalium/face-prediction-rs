pub struct Config {
    pub ultra_model_path: String,
    pub arc_model_path: String,
    pub folder_path: String,
    pub result_folder: String,
    pub test_case_path: String,
}

impl Config {
    pub fn new(args: &[String]) -> Result<Config, &str> {
        if args.len() < 6 {
            return Err("Not enough arguments");
        }

        let ultra_model_path = args[1].clone();
        let arc_model_path = args[2].clone();
        let folder_path = args[3].clone();
        let result_folder = args[4].clone();
        let test_case_path = args[5].clone();

        Ok(Config {
            ultra_model_path,
            arc_model_path,
            folder_path,
            result_folder,
            test_case_path,
        })
    }
}
