@echo off
setlocal enabledelayedexpansion

:: Default values
set MODEL=roman_r
set NAME=roman_r
set EPOCHS=90
set RUN=all
set WRITECSV=
set LAYERS=3
set THRESHOLD_ALL=


:: Define category mapping
set "CATEGORY_0=baseline/pedestrians"
set "CATEGORY_1=baseline/PETS2006"
set "CATEGORY_2=baseline/highway"
set "CATEGORY_3=baseline/office"
set "CATEGORY_4=lowFramerate/port_0_17fps"
set "CATEGORY_5=lowFramerate/tramCrossroad_1fps"
set "CATEGORY_6=lowFramerate/tunnelExit_0_35fps"
set "CATEGORY_7=lowFramerate/turnpike_0_5fps"
set "CATEGORY_8=thermal/corridor"
set "CATEGORY_9=thermal/diningRoom"
set "CATEGORY_10=thermal/lakeSide"
set "CATEGORY_11=thermal/library"
set "CATEGORY_12=thermal/park"
set "CATEGORY_13=badWeather/blizzard"
set "CATEGORY_14=badWeather/skating"
set "CATEGORY_15=badWeather/snowFall"
set "CATEGORY_16=badWeather/wetSnow"
set "CATEGORY_17=dynamicBackground/boats"
set "CATEGORY_18=dynamicBackground/canoe"
set "CATEGORY_19=dynamicBackground/fall"
set "CATEGORY_20=dynamicBackground/fountain01"
set "CATEGORY_21=dynamicBackground/fountain02"
set "CATEGORY_22=dynamicBackground/overpass"
set "CATEGORY_23=shadow/backdoor"
set "CATEGORY_24=shadow/bungalows"
set "CATEGORY_25=shadow/busStation"
set "CATEGORY_26=shadow/copyMachine"
set "CATEGORY_27=shadow/cubicle"
set "CATEGORY_28=shadow/peopleInShade"
set "CATEGORY_29=nightVideos/bridgeEntry"
set "CATEGORY_30=nightVideos/busyBoulvard"
set "CATEGORY_31=nightVideos/fluidHighway"
set "CATEGORY_32=nightVideos/streetCornerAtNight"
set "CATEGORY_33=nightVideos/tramStation"
set "CATEGORY_34=nightVideos/winterStreet"
set "CATEGORY_35=turbulence/turbulence0"
set "CATEGORY_36=turbulence/turbulence1"
set "CATEGORY_37=turbulence/turbulence2"
set "CATEGORY_38=turbulence/turbulence3"
set "CATEGORY_47=cameraJitter/badminton"
set "CATEGORY_48=cameraJitter/boulevard"
set "CATEGORY_49=cameraJitter/sidewalk"
set "CATEGORY_50=cameraJitter/traffic"

:: Set category groups
set "GROUP_all=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 47 48 49 50"
set "GROUP_baseline=0 1 2 3"
set "GROUP_lowFramerate=4 5 6 7"
set "GROUP_thermal=8 9 10 11 12"
set "GROUP_badWeather=13 14 15 16"
set "GROUP_dynamicBackground=17 18 19 20 21 22"
set "GROUP_shadow=23 24 25 26 27 28"
set "GROUP_nightVideos=29 30 31 32 33 34"
set "GROUP_turbulence=35 36 37 38"
set "GROUP_cameraJitter=47 48 49 50"

:: Parse command-line arguments
:arg_loop
if "%~1"=="--threshold_all" set "THRESHOLD_ALL=--threshold_all" & shift & goto arg_loop
if "%~1"=="" goto end_args
if "%~1"=="--model" set "MODEL=%~2" & shift & shift & goto arg_loop
if "%~1"=="--name" set "NAME=%~2" & shift & shift & goto arg_loop
if "%~1"=="--epochs" set "EPOCHS=%~2" & shift & shift & goto arg_loop
if "%~1"=="--run" set "RUN=%~2" & shift & shift & goto arg_loop
if "%~1"=="--writecsv" set "WRITECSV=--writecsv" & shift & goto arg_loop
if "%~1"=="--layers" set "LAYERS=%~2" & shift & shift & goto arg_loop
shift
goto arg_loop

:end_args

:: Determine which category list to run based on the RUN variable
for %%G in (all baseline lowFramerate thermal badWeather dynamicBackground shadow nightVideos turbulence cameraJitter) do (
    if /i "!RUN!"=="%%G" set "CATEGORY_LIST=!GROUP_%%G!"
)

:: If RUN does not match any group, assume it's a specific category or a custom list
if not defined CATEGORY_LIST set "CATEGORY_LIST=!RUN!"

:: Run the Python script for each category in the list
for %%i in (!CATEGORY_LIST!) do (
    set CATEGORY_NUM=%%i
    set CATEGORY_PATH=!CATEGORY_%%i!

    python main_roman.py --log-dir ./logs/3_layers/!CATEGORY_PATH!/ ^
                        --coeff-L .8 --coeff-S .05 ^
                        --coeff-Sside .025 --split 0 ^
                        --hidden-filters 1 --initial-lr 0.003 ^
                        --loss_type L_tversky_bce --reweighted ^
                        --layers !LAYERS! --category !CATEGORY_NUM! ^
                        --name !NAME! --model !MODEL! ^
                        --epochs !EPOCHS! !WRITECSV! !THRESHOLD_ALL!
)

endlocal
