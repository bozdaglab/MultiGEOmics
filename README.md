# MultiGEOmics
MultiGEOmics: Graph-Based Integration of Multi-Omics via Biological
Information Flows

### This repository is our PyTorch implementation of MultiGEOmics.
![Model Architecture](Mhttps://raw.githubusercontent.com/bali-eng/MultiGEOmics/main/Model_architecture_image/Architecture.jpg)
We compare MultiGEOmics with two sets of SOTA models. main_1.py from IGCN paper to run on ADNI, ROSMAP, TCGA_BRCA, TCGA_GBM datasets and main_2.py from DeepKEGG paper to run on AML, BLCA, BRCA, LIHC, PRAD, WT datasets. All the datasets have been collected from these two papers. 


### Create and activate environment
```shell script
conda create -p ./mulgeoenv python=3.8.10 -y
conda activate ./mulgeoenv
```



### Install dependencies 
```shell script
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install dgl-cu117 -f https://data.dgl.ai/wheels/repo.html
pip install -r req.txt
```


### To run the code
```shell script
 python main_run.py --dataset=TCGA_GBM --epochs=450 --early_stopping=150
```


