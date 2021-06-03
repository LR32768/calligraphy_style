# calligraphy_style

## Environment
Ubuntu 18.04.5 LTS with cuda support  
> pytorch==1.8.1  
> torchvision=0.9.1  
> tqdm==4.60.0  
> opencv-python==4.2.0.34  
> numpy==1.19.2  
> pillow==8.2.0  
> matplotlib==3.4.1  
> scipy==1.6.3  
> scikit-image==0.17.2  

## File and Directory Explaination
* ./checkpoints/  
> > the path to save checkpoints  
* ./data/  
> > font file, font skeleton dataset, and training testing dataset directory to be generated  
* ./example_results/  
> > directory to save style transfer results  
* ./generate_data/  
> > code to extract and plot strokes, plot characters  
* .gitignore  
> > specifies intentionally untracked files to ignore  
* dataset.py  
> > load the dataset  
* test_vae.py  
> > testing code  
* train_vae.py  
> > training code  
* vqvae.py  
> > model file  


## Get dataset

### Get annotated KAITI stroke skeleton

To run this code, the first thing to do is to download KAITI_skeleton.tar.gz from [this link](https://cloud.tsinghua.edu.cn/d/15fef53062234b95b984/), unzip this file, and put its contain under './data/KAITI_skeleton/'. So, we can read this data with './data/KAITI_skeleton/FontXXXXX.mat' in the code.

### Get training & testing dataset

The training dataset of our Transformer and VQ-VAE model is the strokes of characters of both source style and target style. How to get the strokes of these characters is one of our contribution in this project. 

Because generating these data consumes huge amount of time and computation (3 * AMD Ryzen Threadripper 2990WX CPU for ~10 hours), we suggest to run the demo code which only generate small amount of data for visualize the effect of code. To get the complete datasat, download training_data.tar.gz and testing_data.tar.gz from [TsinghuaCloud link](https://cloud.tsinghua.edu.cn/d/15fef53062234b95b984/), unzip them, and put them under ./data/.


(suggest) generate small amount of source style data for visualize the effect of code
```
python ./generate_data/extract_strokes/get_strokes_KAITI.py --unicode 13312
python ./generate_data/plot_character/plot_font.py --unicode 13312 --font_style KAITI
```
> the output picture will be saved in   
> >     ./example_results/strokes/process_of_extract_KAITI  
> >     ./example_results/strokes/strokes_KAITI  
> >     ./example_results/whole_fonts/KAITI  

(suggest) generate small amount of target style data for visualize the effect of code
```
python ./generate_data/extract_strokes/get_strokes_SONG.py --unicode 13312
python ./generate_data/plot_character/plot_font.py --unicode 13312 --font_style SONG
```
> the output picture will be saved in   
> >     ./example_results/strokes/process_of_extract_SONG  
> >     ./example_results/strokes/strokes_SONG  
> >     ./example_results/whole_fonts/SONG  

## Training:  
```
python train_vae.py --epoch 200
```
* the output of training stroke data will be saved in 
> > ./example_results/styletrans_out_training/  
* the checkpoints will be saved in 
> > ./checkpoints  

## Testing: transfer font style
```
python test_vae.py --shift --weight_path ./checkpoints/KaiSong_size256_vqvae_200.pt
```
* the output of testing font data will be saved in 
> > ./example_results/styletrans_out_testing/  