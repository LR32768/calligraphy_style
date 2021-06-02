# calligraphy_style

## Environment
xxx  
pip -r requirements.txt

## Code
generate_data/
extract_strokes/
train/

## Get dataset

To run this code, the first thing to do is to download the skeleton of KAITI from [this link](https://cloud.tsinghua.edu.cn/f/5f503ff1457b4d0e8760/), unzip this file, and put its contain under './data/KAITI_skeleton/'. So, we can read this data with './data/KAITI_skeleton/FontXXXXX.mat' in the code.

The training dataset of our Transformer and VQ-VAE model is the strokes of characters of both source style and target style. How to get the strokes of these characters is one of our contribution in this project. 

Because generating these data consumes huge amount of time and computation (3 * AMD Ryzen Threadripper 2990WX CPU for ~10 hours), we suggest to run the demo code which only generate small amount of data. We also give the TsinghuaCloud link for the whole dataset for you guys to download.

### Get annotated KAITI stroke skeleton

### Get training dataset

(suggest) generate small amount of data
```
```

(not suggest) generate all data
```
```

### Get testing dataset

(suggest) generate small amount of data
```
```

(not suggest) generate all data
```
```

## Training:  
### with Transformer based model
### with VQ-VAE based model

## Testing: transfer font style
### with Transformer based model
### with VQ-VAE based model

