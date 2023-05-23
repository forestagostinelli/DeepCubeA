### Cube3

###### Train cost-to-go function
python ctg_approx/avi.py --env cube3 --states_per_update 50000000 --batch_size 10000 --nnet_name cube3 --max_itrs 1000000 --loss_thresh 0.06 --back_max 30 --num_update_procs 30
cp -r saved_models/cube3/current/* saved_models/cube3/target/  # manually update target network
python ctg_approx/avi.py --env cube3 --states_per_update 50000000 --batch_size 10000 --nnet_name cube3 --max_itrs 1200000 --loss_thresh 0.06 --back_max 30 --num_update_procs 30

###### Solve with A* search, use --verbose for more information
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3/current/ --env cube3 --weight 0.6 --batch_size 10000 --results_dir results/cube3/ --language cpp --nnet_batch_size 10000

###### Compare solutions to shortest path
python scripts/compare_solutions.py --soln1 data/cube3/test/data_0.pkl --soln2 results/cube3/results.pkl


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


### 35-puzzle

###### Train cost-to-go function
python ctg_approx/avi.py --env puzzle35 --states_per_update 50000000 --batch_size 10000 --nnet_name puzzle35 --max_itrs 1000000 --loss_thresh 1.0 --back_max 1000 --max_update_steps 200 --num_update_procs 30

###### Solve with A* search, use --verbose for more information
python search_methods/astar.py --states data/puzzle35/test/data_0.pkl --model saved_models/puzzle35/current/ --env puzzle35 --weight 0.8 --batch_size 20000 --results_dir results/puzzle35/ --language cpp --nnet_batch_size 10000

###### See solution results
python scripts/compare_solutions.py --soln1 results/puzzle35/results.pkl --soln2 results/puzzle35/results.pkl


### 48-puzzle

###### Train cost-to-go function
python ctg_approx/avi.py --env puzzle48 --states_per_update 50000000 --batch_size 10000 --nnet_name puzzle48 --max_itrs 2000000 --loss_thresh 1.0 --back_max 1000 --max_update_steps 200 --num_update_procs 30 --num_test 1000

###### Solve with A* search, use --verbose for more information
python search_methods/astar.py --states data/puzzle48/test/data_0.pkl --model saved_models/puzzle48/current/ --env puzzle48 --weight 0.6 --batch_size 20000 --results_dir results/puzzle48/ --language cpp --nnet_batch_size 10000

###### See solution results
python scripts/compare_solutions.py --soln1 results/puzzle48/results.pkl --soln2 results/puzzle48/results.pkl


### Lights Out 7x7
###### Train cost-to-go function
python ctg_approx/avi.py --env lightsout7 --states_per_update 500000 --batch_size 1000 --nnet_name lightsout7 --max_itrs 1000000 --loss_thresh 1.0 --back_max 50 --max_update_steps 200 --update_method astar --num_update_procs 30 --num_test 1000

###### Solve with A* search, use --verbose for more information
python search_methods/astar.py --states data/lightsout7/test/data_0.pkl --model saved_models/lightsout7/current/ --env lightsout7 --weight 0.2 --batch_size 1000 --results_dir results/lightsout7/ --language cpp --nnet_batch_size 10000

###### See solution results
python scripts/compare_solutions.py --soln1 results/lightsout7/results.pkl --soln2 results/lightsout7/results.pkl


### Sokoban
python ctg_approx/avi.py --env sokoban --states_per_update 5000000 --batch_size 1000 --nnet_name sokoban --max_itrs 1000000 --loss_thresh 1.0 --back_max 1000 --max_update_steps 50 --update_method gbfs --num_update_procs 30 --num_test 1000

###### Solve with A* search, use --verbose for more information
# NOTE: is faster when just using one GPU because the batch size is small
python search_methods/astar.py --states data/sokoban/test/data_0.pkl --model saved_models/sokoban/current/ --env sokoban --weight 0.8 --batch_size 1 --results_dir results/sokoban/ --language python --nnet_batch_size 10000

###### See solution results
python scripts/compare_solutions.py --soln1 results/sokoban/results.pkl --soln2 results/sokoban/results.pkl
