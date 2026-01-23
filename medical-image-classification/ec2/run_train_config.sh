data_dir=vindr-spinexr-subset
python train.py --batch-size 32 \
                 --learning-rate 0.001 \
                 --early-stopping-rounds 10 \
                 --data /home/ubuntu/data/$data_dir \
                 --model_dir /home/ubuntu/data/$data_dir/spine-model \
                 --output_dir /home/ubuntu/data/$data_dir/spine-output \
                 --model_name DenseNet121 \
                 --num_epochs 2 \
                 --val_interval 1 \

