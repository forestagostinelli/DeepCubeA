### 15-puzzle

###### Train cost-to-go function
python ctg_approx/avi.py --env puzzle15 --states_per_update 50000000 --batch_size 10000 --nnet_name puzzle15 --max_itrs 1000000 --loss_thresh 0.1 --back_max 500 --num_update_procs 30

###### Solve with A* search, use --verbose for more information
python search_methods/astar.py --states data/puzzle15/test/data_0.pkl --model saved_models/puzzle15/current/ --env puzzle15 --weight 0.8 --batch_size 20000 --results_dir results/puzzle15/ --language cpp --nnet_batch_size 10000

###### Compare solutions to shortest path
python scripts/compare_solutions.py --soln1 data/puzzle15/test/data_0.pkl --soln2 results/puzzle15/results.pkl

### 24-puzzle

###### Train cost-to-go function
python ctg_approx/avi.py --env puzzle24 --states_per_update 50000000 --batch_size 10000 --nnet_name puzzle24 --max_itrs 1000000 --loss_thresh 0.2 --back_max 500 --num_update_procs 30

###### Solve with A* search, use --verbose for more information
python search_methods/astar.py --states data/puzzle24/test/data_0.pkl --model saved_models/puzzle24/current/ --env puzzle24 --weight 0.8 --batch_size 20000 --results_dir results/puzzle24/ --language cpp --nnet_batch_size 10000

###### Compare solutions to shortest path
python scripts/compare_solutions.py --soln1 data/puzzle24/test/data_0.pkl --soln2 results/puzzle24/results.pkl
