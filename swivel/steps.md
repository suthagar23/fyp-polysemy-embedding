# Swivel Steps

##Pre Processing for Swivel
1. Need normal CPU based instance for pre-matrix creation (Need Tensorflow - Python2) . 
Use AWS Deeplearning-Ubuntu AMI for Instance
2. Configure your aws, and then unlimt the file count


    $ aws configure
    $ ulimit -n 50000 instead of 1200
    
  3 .  Change conda enviroment


    $ condo info --envs
    $ source activate tensorflow_p27
    
 4 .  Run this pre-process code

    $ python prep.py --output_dir /tmp/swivel_data --input /tmp/wiki.txt



## Training the embeddings

1. Need p2.8large instance (8GPU access)
2. Download all files from S3 to the instance, and change the conda enviromemnt


    $ conda info --envs
    $ source activaye tensorflow_p27
    $ python ./swivel.py --input_base_path /tmp/swivel_data \
       --output_base_path /tmp/swivel_data --num_epochs 40
    
3 .  Open another tap, and Goto /egpu_output folder. And see the logs by these command (there will be 8 files for 8 workers)

      $ tail -f ps.0 

4 .  Too see the GPU usage 

      $ nvidia-smi

      



## Exploring and evaluating the embeddings

1. Normal Instance is enough for this


    $ source activate tensorfow_p27
    $ python text2bin.py -o vecs.bin -v vocab.txt /tmp/swivel_data/*_embedding.tsv

You can do some simple exploration using `nearest.py`:

    ./nearest.py -v vocab.txt -e vecs.bin
    query> dog
    dog
    dogs
    cat
    ...
    query> man woman king
    king
    queen
    princess
    ...

To evaluate the embeddings using common word similarity and analogy datasets,
use `eval.mk` to retrieve the data sets and build the tools:

    $  make -f eval.mk
    $ python wordsim.py -v vocab.txt -e vecs.bin *.ws.tab
    $ analogy --vocab vocab.txt --embeddings vecs.bin *.an.tab

