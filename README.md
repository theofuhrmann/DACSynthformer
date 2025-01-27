# DACSynthformer

DACSynthformer is a basic transformer that runs on the Descript Audio Codec representation of audio. It maintains the "stacked" codebook at each transformer time step (as oppose to laying the codebooks down "horizonally" for examaple). It uses a smallish causal mask during training, so that during autoregressive inference we can use a small context window. It uses RoPE positional encoding because absolute positions are irrelevant for the continuous and stable audio textures we wish to generate. Conditioning is currently under experimental deveopment in so check Github for status on that. 

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
> docker image build --build-arg USER_ID=$(id -u) --build-arg > GROUP_ID=$(id -g) --file Dockerfile.txt --tag whatever .
> ~~~

> ### Run the docker container
> ~~~
> docker run --ipc=host --gpus all -it -v $(pwd):/dacsynthformer  -v /home/lonce/scratchdata:/scratch --name dacsynthformer --rm -p 8888:8888  dacsynth
> cd /dactransformer
> jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
> ~~~
> I use scratch as the root directory for data, etc. 

## There are 2 primary notebooks:  
1) Train.ipynb - this is the main notebook for training the model. The dataloader loads pre-coded DAC files (4 codebooks for 44.1kHz sampled audio). It creates checkpoints that you can use to generate audio. 
2) CKPT_DAC_AUDIO.ipynb - uses a stored trained model to first generate a DAC coded file and then decodes that to audio. 

Each of the notebooks has a "parameters" section near the top for choosing the model and some parameter that determine the architectore (for training), or some options (for inference).



## To train:  

1) Edit the parameter file, params.yaml (or make your own)
2) Open Train.ipynb and set the parameter file name and any other parameters in  
3) Run all cells
4) (There is a params_sm.yaml and a test/data folder for running bare minimum config and dataset)


## Stored weights (checkpoints): 
If you just want to just explore the inference phase of the model, the weights for a trained model are available here:
https://lonce.org/downloads/dacsynthformer/runs.zip
Unzip this file in your main dacsynthformer directory and run first the CKPT2DAC.ipynb and then the DAC2Audio.ipynb notebooks.

The stored model was trained on 4 texture sounds from the syntex data set (Pistons, Wind, Applause, Bees) with one-hot conditioning info for the class, and one continuous parameter for manipulating each sound. You can listen to examples of the sounds without downloading data sets here: https://syntex.sonicthings.org/soundlist

