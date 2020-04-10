### Cube 3

###### Generate training and validation data
python scripts/generate_dataset.py --env cube3 --back_max 30 --data_dir data/cube3/train/ --num_per_file 1000000 --num_files 1000 --num_procs 1
python scripts/generate_dataset.py --env cube3 --back_max 30 --data_dir data/cube3/val/ --num_per_file 10000 --num_files 1 --num_procs 1

###### Train cost-to-go function
python ctg_approx/avi.py --env cube3 --states_per_update 50000000 --batch_size 10000 --nnet_name cube3 --train_dir data/cube3/train/ --val_dir data/cube3/val/ --max_itrs 1000000 --loss_thresh 0.05 --num_update_procs 1
python ctg_approx/avi.py --env cube3 --states_per_update 50000000 --batch_size 10000 --nnet_name cube3 --train_dir data/cube3/train/ --val_dir data/cube3/val/ --max_itrs 1200000 --update_num 15 --loss_thresh 0.05 --num_update_procs 1

###### Solve with A* search, use --verbose for more information
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3/19/ --env cube3 --weight 0.1 --batch_size 100 --results_file results/cube3/results.pkl

###### Solve with greedy best-first search (GBFS)
python search_methods/gbfs.py --model_dir saved_models/cube3/19/ --env cube3 --data_dir data/cube3/val/ --max_steps 30


### 15-puzzle

###### Generate training and validation data
python scripts/generate_dataset.py --env puzzle15 --back_max 500 --data_dir data/puzzle15/train/ --num_per_file 1000000 --num_files 1000 --num_procs 1
python scripts/generate_dataset.py --env puzzle15 --back_max 500 --data_dir data/puzzle15/val/ --num_per_file 10000 --num_files 1 --num_procs 1

###### Train cost-to-go function
python ctg_approx/avi.py --env puzzle15 --states_per_update 50000000 --batch_size 10000 --nnet_name puzzle15 --train_dir data/puzzle15/train/ --val_dir data/puzzle15/val/ --max_itrs 1000000 --loss_thresh 0.1 --num_update_procs 1

###### Solve with A* search, use --verbose for more information
python search_methods/astar.py --states data/puzzle15/test/data_0.pkl --model saved_models/puzzle15/59/ --env puzzle15 --weight 0.8 --batch_size 20000 --results_file results/puzzle15/results.pkl --language cpp --nnet_batch_size 10000

