# GPU Cluster Advisor

An AI-powered GPU cluster advisor that predicts workload runtime, GPU utilization, resource efficiency, and scheduling behavior using real-world hyperscale production workload traces.

## Overview

Large AI workloads often require significant GPU resources, but choosing an efficient resource configuration can be difficult. Over-allocation may waste expensive computing resources, while under-allocation may increase execution time or create performance bottlenecks.

GPU Cluster Advisor uses machine-learning models to analyze historical GPU-cluster workloads and estimate performance outcomes such as:

- Workload execution duration
- GPU utilization
- GPU memory usage
- Scheduling delay
- Resource efficiency
- GPU underutilization risk

The long-term goal is to provide data-driven resource recommendations for AI workloads running on heterogeneous GPU clusters.

## Project Status

🚧 This project is currently under development.

Planned development stages include:

1. Explore and preprocess the workload-trace data
2. Build baseline prediction models
3. Train and evaluate CatBoost models
4. Predict workload execution duration
5. Predict GPU utilization and memory usage
6. Identify potentially underutilized GPU allocations
7. Develop an interactive performance-advisor dashboard

## Dataset

This project uses the **Alibaba Cluster Trace GPU v2026** dataset.

The dataset was released by researchers associated with Alibaba Serverless Infrastructure and accompanies the research paper:

> **Heterogeneity at Hyperscale: Characterization and Scheduling of Large Production AI Clusters at Alibaba**

The trace contains anonymized workload information from a large-scale production AI cluster. According to the dataset documentation, the trace spans approximately six months and covers up to 155,410 GPUs across 37,707 GPU servers at hourly peak.

The released data includes information related to:

- AI workload and model categories
- GPU models
- GPU, CPU, and memory resource requests
- GPU and system-resource utilization
- Workload execution duration
- Scheduling and readiness delays
- Workload priority classes
- Server inventory and anonymized cluster topology

### Dataset Attribution

All dataset credit belongs to the original dataset authors and contributors.

This repository is an independent student machine-learning project and is not affiliated with, endorsed by, or sponsored by Alibaba or the dataset authors.

The dataset was not created by the author of this repository. This project uses the released trace only for educational and research purposes.

Users should obtain the dataset from its official source and review the applicable documentation, license, citation requirements, and terms of use before using it.

The dataset is not redistributed in this repository.

### Dataset Citation

If you use the Alibaba Cluster Trace GPU v2026 dataset, please cite the accompanying paper:

```bibtex
@inproceedings{asi_trace_2026,
  title = {Heterogeneity at Hyperscale: Characterization and Scheduling of Large Production AI Clusters at Alibaba},
  author = {Suyi Li and Lingyun Yang and Haoxuan Yu and Sheng Yao and Tianyuan Wu and Xiaoxiao Jiang and Hanfeng Lu and Kangjin Wang and Chenhao Wang and Shenglin Xu and Lun Wang and Qingyang Duan and Shenghao Liang and Xiu Lin and Wenchao Wu and Yinghao Yu and Guodong Yang and Liping Zhang and Wei Wang},
  booktitle = {20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26)},
  year = {2026}
}
