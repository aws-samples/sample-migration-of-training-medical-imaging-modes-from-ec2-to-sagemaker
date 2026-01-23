data_dir=vindr-spinexr
python train.py --batch-size 32 \
                 --learning-rate 0.001 \
                 --early-stopping-rounds 10 \
                 --data /home/ubuntu/data/$data_dir \
                 --model_dir /home/ubuntu/data/$data_dir/spine-model1 \
                 --output_dir /home/ubuntu/data/$data_dir/spine-output1 \
                 --model_name ViT \
                 --num_epochs 1 \
                 --val_interval 1

