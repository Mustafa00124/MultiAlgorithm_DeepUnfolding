@echo off
setlocal enabledelayedexpansion

:: List of model names
set models=median svd dft Madu2 Serial2 Ensemble2

:: List of seeds
set seeds=4 5

:: Loop through each seed
for %%s in (%seeds%) do (
    echo Running with seed: %%s
    
    :: Loop through each model name
    for %%m in (%models%) do (
        echo Running model: %%m with seed %%s
        call run_script_madu.bat --writecsv --epochs 60 --run all --layers 3 --name %%m --mode %%m --seed %%s

        :: Print a message before the delay
        echo Don't stop cooking, we will resume in 10 minutes...

        :: Add a 20-minute delay
        timeout /t 1200 /nobreak
    )
)

echo All models and seeds have been processed.
pause