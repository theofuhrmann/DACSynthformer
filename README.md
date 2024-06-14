# DACSynthformer

DACSynthformer is a basic transformer that runs on the Descript Audio Codec representation of audio. It maintains the "stacked" codebook at each transformer time step (as oppose to laying the codebooks down "horizonally" for examaple). It uses a smallish causal mask during training, so that during autoregressive inference we can use a small context window. It uses RoPE positional encoding because absolute positions are irrelevant for the continuous and stable audio textures we wish to generate. Conditioning is currently under experimental deveopment in so check Github for status on that. 

## Create the docker container  
~~~
docker image build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --file Dockerfile.txt --tag whatever .
~~~

## Run the docker container
~~~
docker run --ipc=host --gpus "device=0" -it -v $(pwd):/whatever  -v /scratch:/scratch --name dacshynthformer --rm whatever
~~~
I use scratch as the root directory for data, etc. 

## There are 3 noteooks:  
1) Train.ipynb - this is the main notebook for training the model. The dataloader loads pre-coded DAC files (4 codebooks for 44.1kHz sampled audio). It creates checkpoints that you can use to generate audio. 
2) CKPT2DAC.ipynb - this notebook takes a checkpoint and generates DAC files of arbitrary length.
3) DAC2Audio.ipynb - this notebook takes a DAC file and generates a WAV file.

Each of the notebooks has a "parameters" section at the top. 

