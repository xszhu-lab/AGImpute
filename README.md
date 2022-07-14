
# AGImpute
A composite structure model for single-cell RNA-seq imputation
## Tabel of Contents
- [Install AGImpute](#installAGImpute)
- [Install dependences](#installdependences)
- [Input file format](#inputfileformat)
- [Output files](#outputfiles)
- [Run with Testdata](#runwithtestdata)

## <a name="installAGImpute"></a>Install AGImpute
- **Download** 
```
git clone https://github.com/xszhu-lab/AGImpute.git
cd AGImpute
```
## <a name="installdependences"></a>Install dependences
- **Install** 
AGImpute is implemented in `python`(>3.8) and `pytorch`(>10.1) or `cuda`(11.4),Please install `python`(>3.8) and `pytorch`(>10.1) or cuda dependencies before run AGImpute.Users can either use pre-configured conda environment(recommended)or build your own environmen manually.
 ```
 pip install -r requirements.txt 
 ```

## Commands and options
```
python3 AGImpute.py --help
```
You should see the following output:
```
Usage:AGImpute.py [OPRIONS]

    Options:
    --GPU                       Use GPU for AGImpute.
    --batch_size                Size of the batches.
    --lr                        Adam: learning rate.
    --K                         K parameters.
    --channels                  Number of image channels.
    --img_size                  Training set size.
    --epochs_a                  Number of epochs of training autoencoder.
    --epochs_g                  Number of epochs of training gan.
    --b1                        Adam: decay of first order momentum of gradient.
    --b2                        Adam: decay of first order momentum of gradient.
    --gamma                     Gamma parameters.
    --D_throd                   Dropout events threshold positioning algorithm confidence value.
    --feature_gene_num          Number selection of Feature Genes
    --dim_thord                 Latent-dim throd.
    --name                      Name of h5ad.
    --file_c                    Path of h5ad file.
    --file_model                Path of model file.
    --outdir                    The directory for output.
```

### <a name="inputfileformat"></a>Input file format
Our input requires a `CSV` file, the `index` should be `cell`, and the `column` should be `gene`.

![image](https://github.com/xszhu-lab/AGImpute/blob/main/images/expression%20matrix.png)

### <a name="outputfiles"></a>Output files
### <a name="runwithtestdata"></a>Run with Testdata
```
python AGImpute.py --name sim_counts --file_c ./Testdata 
```
