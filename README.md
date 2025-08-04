# MultiGEOmics




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


# To run the code
```shell script
python main_run.py --dataset=TCGA_BRCA --epochs=450 --early_stopping=150
```