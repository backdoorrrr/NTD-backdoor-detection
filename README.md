# NTD-backdoor-detection 
(NTD for Face Recognition)

---
If you find it is useful and used for publication. Please kindly cite our work as:
> ```
> @inproceedings{li2021ntd,
>   title={NTD: Non-Transferability enabled Deep Learning Backdoor Detection},
>   author={Li, Yinshan and Ma, Hua and Zhang, Zhi and Gao, Yansong and Abuadbba, Alsharif and Xue, Minhui and Fu, Anmin and Zheng, Yifeng and Al-Sarawi, Said F and Abbott, Derek},
>   journal={IEEE Transactions on Information Forensics and Security},
>   year={2023},
>   publisher={IEEE}
> }

## Required environment

- ```tensorflow-gpu==2.2  ```
- ```keras==2.1.5  ```
- ```python==3.7.9  ```

## Description of the file
msceleb1m_lfw_mapping_probability.txt 

https://github.com/inlmouse/MS-Celeb-1M_WashList

For the comparison of lfw to msceleb1m on github, similarity of less than 0.5 is considered to be non-overlapping.

Our experiment refers to this list to filter out 79 people with similarity of less than 0.5 and consistent with the name of the facescrub dataset, which is not considered to overlap with msceleb1m.


## Run
1、Divide the task dataset into Offline_dataset and Online_dataset, at a scale of approximately 3:7, and save it under the dataset folder.

2、Put the pretrained model into the model_data folder.

3、The Offline phase(determine threshold phase):

    python offline_determine_threshold.py -p Gerard_Butler -s tanimoto -n 3 -r 0.005  
    
The parameters (preset frr('-r'), similarity metric('-s'), comparison set n('-n')) are optional, '-p' can be a person's name or global.

Image results are saved at the plot/offline_plot, and results of related thresholds are saved in the result/offline_csv_result.csv for online stage reading.  

4、The Online phase (testing phase):

    python online_testing.py -p Gerard_Butler
    
The image results are saved at plot/online_plot, and each experiment result is kept in the result/result.txt
    

## Reference
Thanks to github author bubbliiiing for providing some of the open source code, https://github.com/bubbliiiing/keras-face-recognition
