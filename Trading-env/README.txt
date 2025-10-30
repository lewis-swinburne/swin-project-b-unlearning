this folder contains both training scripts and unlearning scripts for financial data.
using gym-trading-env as the environment.


folders:
'checkpoints' - container for training checkpoints, outputs and other data 
'data' - container for financial data (after BTC and ETH are downloaded)
'exports' - container outputs of any benchmarking, evaluation or plotting script
'images' - testing images from different stages of development
'unlearning_results' - container for unlearning outputs (benchmarking and monitoring) and other data


scripts:
- compiling scripts:
'compile_unlearning_benchmarks' - collects data from unlearning JSON files (monitoring) and puts them in .csv format for comparison. Compares component (CPU & RAM) performance statistics between seeds and epochs.

epoch: number of unlearning sessions
seed: ensures reproducibility, same seed and epoch should guarantee same outcome.

forget_degradation_percent: how much the model's performance degraded on data it was supposed to unlearn - higher values mean the agent forgot more.
retain_preservation_percent: how well the model performed on data it should remember.

original_forget_mean: average performance on forget set before unlearning
original_retain_mean: average performance on retain set before unlearning

unlearned_forget_mean: average performance on forget set after unlearning
unlearned_retain_mean: average performance on retain set after unlearning


'compile_unlearing_results' - collects data from unlearning JSON files (benchmarking) and puts them in .csv format for comparison. Compares unlearning peformance between seeds and epochs.

unlearn_epochs: number of unlearning session
seed: seed used
start_timestamp / end_timestamp: when the unlearning run started and finished
total_duration_seconds: how long the entire unlearning took

peak_cpu_percent: maximum CPU usage during unlearning
peak_memory_mb: maximum memory usage during unlearning
average_cpu_percent: average CPU usage during unlearning
average_memory_mb: average memory usage during unlearning

- plotting scripts:
'plot_learning_average_rewards' - collects average training rewards for each 50 episodes, then plots on graph that shows the average reward over time.

'plot_learning_market_vs_agent' - completes one run of agent training session with BTC-USD data, then plots outcome comparison source financial data to agent performance.

'plot_learning_market_vs_agent_ETH' - completes one run of agent training session but with ETH-USD data, in order to compare performance of agent with different data set. then plots outcome.

'plot_unlearning_all_epochs' - reads the benchmarking JSON files and creates graphs comparing the outcomes of the unlearning sessions

'plot_unlearning_market_vs_agent' - completes one run of the agent unlearning session with the specified unlearned model. plots outcome. can be changed to use BTC or ETH financial data.


- learning & unlearning scripts:
'trading_download' - downloads both BTC/USD and ETH/USD from 2020 to 2025 for usage as training data.

'trading_learning' - completes series of training episodes, based off of BTC-USD financial data. currently max total episodes is set at 4000 and new run episodes is set at 1000. These values can be changed but 1000 episodes takes around 24 hours to complete.

'trading_poisoning' - completes series of unlearning episodes, as determined by user (epochs). corruption level is also a variable set by user (0.1 to 0.15 seems acceptable, any higher has too variable results). seed values are to prevent random runs from occuring and establish consistent and replicable results.
example run: python3 trading_poisoning.py --epochs 10 25 50 75 100 150 200 --seeds 0 1 42 123 768 --corruption 0.1


Ideal run order:
0.5: trading_download // get financial data if not already downloaded
1: trading_learning // conduct learning
2: poisoning_current // conduct unlearning
