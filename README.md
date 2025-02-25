# DACSynthformer

DACSynthformer is a basic transformer that runs on the Descript Audio Codec representation of audio. It maintains the "stacked" codebook at each transformer time step (as oppose to laying the codebooks down "horizonally" for examaple). It uses a smallish causal mask during training, so that during autoregressive inference we can use a small context window. It uses RoPE positional encoding because absolute positions are irrelevant for the continuous and stable audio textures we wish to generate. Conditioning is provided as a vector combining a one-hot segment for sound class, and real number(s) for parameters.

## Install option 1 (conda only) 
~~~
conda create --name dacformer python==3.10
conda activate dacformer
pip install -r requirements.txt
jupyter lab &
~~~

## Install option 2 (Docker) 

> ### Create the docker container  
> ~~~
> #updated Feb 8, 2025 to use docker buildx:
> docker buildx build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --file Dockerfile.txt --tag yourtag --load .
> ~~~

> ### Run the docker container
> ~~~
> docker run --ipc=host --gpus all -it -v $(pwd):/dacsynthformer  -v /home/lonce/scratchdata:/scratch --name callitwhatyouwill --rm -p 8888:8888  yourtag
> cd /dactransformer
> jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
> ~~~
> I use scratch as the root directory for data, etc. 

## There are 2 primary notebooks:  
1) Train.ipynb - this is the main notebook for training the model. The dataloader loads pre-coded DAC files (4 codebooks for 44.1kHz sampled audio). It creates checkpoints that you can use to generate audio. 
2) Inference.Decode.ipynb - uses a stored trained model to first generate a DAC coded file and then decodes that to audio with the Descript codec ("DAC"). 

Each of the notebooks has a "**Parameters**" section near the top.

* In Train.ipynb, choose your params.yaml file (which is where most of your model parameters are set). The  "**Parameters**" section also is where you choose the platform ('cpu' or 'cuda') and the starting epoch number (0 unless you want to pick up training where a previous run left off). 

* In Inference.Decode.ipynb, the "**Parameters**" is where you set the "experiment name" to whatever you named your experiment in the params.yaml file you trained with. 

  

## Quick Start:

Once you have created and are in your conda (or docker) environment and have started jupyter, just:

* Set paramfile = 'params_mini.yaml' in the "**Parameters**" section of Train.ipynb
* Run All Cells (params_mini.yaml uses testdata with 4 sounds, and trains on 10 epochs.)
* Make sure   experiment_name="mini_test_01" in the "**Parameters**" section of Inference.Decode.ipynb, and cptnum=10.  
* Run All Cells 

This runs a very minimal model, and doesn't train long enough to generate anything but noise. The intention is that you can see that Training and Inference code is functioning!

## Preparing data: 

The conda and docker environments have already installed the Descript DAC codec package, so you can encode .wav files to .dacs and decode .dac file to .wavs. Just encode a folder of wav files like this:

`python3 -m dac encode /my/wavs/folder --model_bitrate 8kbps --n_quantizers 4 --output my/output/folder/`. For more information about the Descript codec, see:

https://github.com/descriptinc/descript-audio-codec

The prepare your excel data file (that pandas will use). It should have columns, with labels in the first row:

Full File Name     |        Class Name         |    Param1   | ....  | ParamN

The file name includes no path (you provide that in a params.yaml config file). Class Names are whatever you choose. Synthformer will create a separate one-hot class element in the conditioning vector used for training and inference for each unique Class Name. (You can see examples of the excel files in testdata). Consider a balance of classes for your training data! 

Note: you typically write a little python program to generate your excel Pandas Frames from your collection of data files. 



## To train:  

1) Edit (or create) a parameter.yaml file with model parameters, folder names for your data and file names for the excel Pandas file.  
2) Open Train.ipynb, set your parameter.yaml file  in the "**Parameters**" section of Training.ipynb and any other parameters you want (such as  DEVICE). 
3) Run all cells.
4) There is a ridiculously tiny data set of dac files in test/data, and a few prepared parameter files:
   1)  params_mini.yaml - for QuickStart and code debugging. Runs all code, but with minimal data and a tiny model.
   2)  params_sm.yaml - good for debugging since it quickly uses all stages of the computation.
   3)  params_med.yaml - uses a slightly bigger model, and the tiny data set for training period long enough to actually see that training and inference work. This runs in a reasonable time on a cpu (maybe 20 minutes training, 2 minute inference) 
   4)  params.yaml - defines a bigger model, and meant for running a larger dataset for many epochs.



## Optional Info: 

* Prepared dataset:

The testdata/ folder has everything you need as a template for creating your own dataset. However, here is a dataset along with a param.yaml file that specifies a medium-size model that you can use for testing, seeing how bi you might want your data and model to be for your own work, etc: 

https://drive.google.com/file/d/1IdMb4v9wD4nHlFLFJe-pl85rFQW0eF-Y/view?usp=sharing

With this data and model spec, I see training at a rate of about 2 minutes per epoch on a powerful desktop machine (CPU only). You can see that it is training after about 10 epochs, and starting to produce something reasonable for some of the sounds after 30 epochs. Reduce the size of the model for speedier training.

The sounds are from the Syntex sound textures data set syntex data set ( https://syntex.sonicthings.org/soundlist) 

* Pistons, with a 'rate' parameter,
*  Wind, with a 'gustiness' parameter
* Applause, with a 'number of clappers' parameter
* Bees (bugs , with a 'busy-body' parameter (how fast and far the bees move)
* Peepers, with a 'frequency range' parameter
* TokWottle, with a 'wood to metal' hit ratio parameter 
* FM, with a 'modulation frequency' parameter




<!--   https://lonce.org/downloads/dacsynthformer/runs.zip -->
