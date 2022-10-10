## Description


Calibration and uncertainty metrics used in the study of the applicability domain in federated multitask learning. 

The two Jupyter Notebooks provide examples on code usage for Conformal Prediction and Platt scaling. The code on Conformal Prediction is based on the implementation from https://github.com/ptocca. 

## Environment

A minimal environment is provided in `environment.yml` <br>
The conda environment can be set up by: 

```
conda env create --file environment.yml
```

The resulting conda environment can then be activated:  
```
conda activate conformal_efficiency
```
In this environment, the tests can be run to check proper functioning by: 
```
python -m unittest
```

