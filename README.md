# Code for my Bachelor's thesis: "Modeling Creativity using Artificial Neural Networks"
This code is the modification of the work of Nam and colleagues (Nam et al., 2023). Their code can be found at: https://github.com/andrewnam/gfn_attractors <br>
The thesis can be found in a Google document: https://docs.google.com/document/d/1jkktVjxmE--mMV4ckx6sFCs6asPFnBGwToZr_rNxO5M/edit?usp=sharing<br>
The aim of this thesis was to model creative behaviour in an artificial neural network through raising the level of entropy in that network.
## Resulting models and metrics
The models trained during this project can be found on Google Drive: https://drive.google.com/drive/folders/14wSmfE2_qlnk91Q_Y1R-Pm0Plis-wrzI?usp=sharing <br>
The raw metrics from running these models can be seen here: https://drive.google.com/drive/folders/17TRD9y0kJceQB3mQRNY6sKkaV_vJVveJ?usp=sharing

## Code
This project was run using Python 3.10.11 <br>
The rest of the modules and requirements can be found in ``` requirements.txt ```<br>
## Running the code, examples
Here I will provide some examples of the commands used for training the models. The exact training jobs that I used for training my models can be found under **/example_hpc_jobs/**<br>
### Generating results
``` python generate_results.py --config configs/dsprites_gfn_dynamics_em_half_sd.yaml --device -1 --dynamics true --fixed_f ./src/rundata_dsprites/dsprites_500_ref_forwardmlp.pt --fixed_b ./src/rundata_dsprites/dsprites_500_ref_backwardmlp.pt --load ..\hpc\models\dsprites\dspritesmlp_400_half.pt --vectors false --plot_img true --content_name  ```
