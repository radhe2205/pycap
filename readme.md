This project is part of udacity nanodegree course. This project works on captcha images present 
in kaggle dataset at: https://www.kaggle.com/fournierp/captcha-version-2-images 

Install the libraries mentioned in setup.py file.

Model is already trained and weights of model are present in folder saved_models. Default run of file captcha_reader
will process all 1070 images through best model found. Output of which is mismatched predictions and finally total number
of right predictions.

Each file in the saved_models folder has weight corresponding to a specific configuration.

* All the tunable parameters are present on top of the file captcha_reader.py

Run Configurations
1) To regenerate all the weights and re-learn all the images call get_results_for_all_parameters from main function. This will finally output corresponding results of each type of configurations it has run.
2) Learning parameter
    * reload_weights: Set this to True to use already present weights in the saved_models folder. Setting this false will trigger learning of all the images again.
    * single_char: Setting this true will use single-label classification(character extraction from image), setting it false will enable multi-label classification.
 
 
Note: Do not change avg_character_length or any input data specific parameters.