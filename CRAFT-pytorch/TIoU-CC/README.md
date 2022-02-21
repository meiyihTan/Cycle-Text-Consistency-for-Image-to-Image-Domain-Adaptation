# TIoU-metric on Python3

The project is forked from [this repo](https://github.com/PkuDavidGuan/TIoU-metric-python3 "this repo").

# Get started

Install prerequisite packages:
```shell
pip install shapely Polygon3
```

Please refer to main.py on how to run the evaluation function.
There are a few main parameters:
1.  root_path: root path of this folder.
2. dataset_name: 'icdar15', 'total_text', or 'ctw'.
3. run_name: given name of the output result folder 'epoch1,... etc'.
4. res_path: folder with all inference output text files. You can refer to the 'results' folder for some samples.
5. output_path: consists the evaluation output result files with per image results.

Do note that the inference output file for ICDAR15 must be "res\_img\_([0-9]+).txt" and file name Total Text should be "img([0-9]+).txt" and ctw is "([0-9]+).txt".
