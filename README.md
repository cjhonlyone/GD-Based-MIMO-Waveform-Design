## Neural Network-inspired Phase-coded Waveform Design for MIMO Radar Based on Gradient Descent
### Python Program
#### Requirments

```
Python 3.8.0
torch == 2.0+cu118
cuda == 11.8
cudnn == 8.8.1
```

#### Usage

```
python mptest.py example.csv
```

When executing the script, a CSV file containing various test vectors must be provided. 

Each row in the file represents a set of design parameters, primarily including the number of channels, code length, and target PSL.

Upon execution, a folder named with `WD` followed by the timestamp will be created, and the currently running code will be copied into this folder for reproducibility. 

Additionally, a CSV file named with `DL` followed by the timestamp will be generated, documenting the execution time and optimization results for each set of parameters.

The results for each parameter set will be saved in a folder prefixed with `Job.` 

The `pyWaveform.mat` file will contain the optimized waveform, while `trainloss.mat` will record the metrics during the optimization process.

For Weighted PSL optimization, both `G` and `E` need to be specified. This means that the lags between `G` and `E` will be assigned the weight `W` during the optimization process.

For instance, if `G` is set to 10, `E` to 20, and `W` to 0.999, this indicates that sidelobes between lags 10 and 20 will be suppressed.

If you set `G` to 0 and `E` to `N-1`, the optimization will apply to all lags for PSL.

### Matlab Program

Take a look at the file `lse_time_perf.m`.

### Citation

Please cite this paper in your publications if it helps your research:

```
@ARTICLE{10739961,
  author={Cao, Jiahui and Sun, Jinping and Wang, Guohua and Zhang, Yuxi and Wang, Wenguang and Wang, Jun},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Neural Network-Inspired Phase-Coded Waveform Design for MIMO Radar Based on Gradient Descent}, 
  year={2024},
  doi={10.1109/TAES.2024.3488687}}
```
