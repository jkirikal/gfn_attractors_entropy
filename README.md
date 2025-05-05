# Code for my Bachelor's thesis: "Modeling Creativity using Artificial Neural Networks"
This code is the modification of the work of Nam and colleagues (Nam et al., 2023). Their code can be found at: https://github.com/andrewnam/gfn_attractors <br>
The thesis can be found in a Google document: https://docs.google.com/document/d/1jkktVjxmE--mMV4ckx6sFCs6asPFnBGwToZr_rNxO5M/edit?usp=sharing<br>
The aim of this thesis was to model creative behaviour in an artificial neural network through raising the level of entropy in that network.
## Resulting models and metrics
The models trained during this project can be found on Google Drive: https://drive.google.com/drive/folders/14wSmfE2_qlnk91Q_Y1R-Pm0Plis-wrzI?usp=sharing <br>
The raw metrics from running these models can be seen here: https://drive.google.com/drive/folders/17TRD9y0kJceQB3mQRNY6sKkaV_vJVveJ?usp=sharing

## Code
This project was run using Python 3.10.11. <br>
The rest of the modules and requirements can be found in ``` requirements.txt ```.<br>
## Running the code, examples
Here I will provide some examples of the commands used for training the models. The exact training jobs that I used for training my models can be found under **/example_hpc_jobs/**.<br>
### The main code
The code for training models on the HBV and dSprites tasks are **/src/diffusion_gfn.py** and **/src/dsprites_gfn.py**, respectively.
### Training the models
An example of training a dSprites model with a double fixed_sd variant. This model is trained for 400 epochs. <br>
```python dsprites_gfn.py --config ../configs/dsprites_fixed_double.yaml --device 0 --dynamics true --epochs 400 --run_name dsprites_400_double```. <br> <br>
An example of training a HBV model with a fixed_mlp and double standard deviation. Therefore this requires the loading of a fixed-forward and a fixed-backward module, that determine the standard deviation at each step. <br>
```python diffusion_gfn.py --config ../configs/diffusion_gfn_dynamics_em_double_sd.yaml --device 0 --dynamics true --epochs 400 --run_name gfn_fixedmlp_double_400 --fixed_f ./rundata/train_500_forwardmlp.pt --fixed_b ./rundata/train_500_backwardmlp.pt```.

### Generating results
This is an example of running the fixed_mlp version of the dSprites model. Here I have to load the fixed-forward- and fixed-backward-stepping model every time that I run it. <br>
``` python generate_results.py --config configs/dsprites_gfn_dynamics_em_half_sd.yaml --device -1 --dynamics true --fixed_f ./src/rundata_dsprites/dsprites_500_ref_forwardmlp.pt --fixed_b ./src/rundata_dsprites/dsprites_500_ref_backwardmlp.pt --load ..\hpc\models\dsprites\dspritesmlp_400_half.pt --vectors false --plot_img true --content_name  mlp_half_sd```. <br> <br>
Here is an example of running a fixed_sb model on the HBV dataset task. <br>
```python generate_results.py --dynamics true --device -1 --from_same_point true --plot_gif true --vectors true --load ..\hpc\models\binary_vec\gfn_fixed_double_400.pt --config configs\diffusion_fixed_double.yaml --content_name vector_sd_double```,
