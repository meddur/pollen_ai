# pollen_ai

This repo contains the training and prediction python scripts as used in https://doi.org/1866/28706  

The prediction script should be plug and play once the dependencies are installed. Notably:

> Keras 1.0

> Tensorflow 1.14

Yes, I agree this shouldn't run on legacy releases. The saved models should also work using PyTorch (which I would recommend using)

The images located in the fossil_dataset folder are can be used in order to play around with the scripts - it is unlabeled data gathered from fossil pollens, captured using the Classyfinder slide scanner.

Classification results will be stored in the images_post_class folder as a .csv file. In order to troubleshoot, the images can be locally copied in their corresponding subfolder 
(i.e. CNN_BELA/images_post_class/level_1/1002/alnus_c)


Model saves can be accessed using [dvc](https://dvc.org/). They are hosted on a public drive @ https://drive.google.com/drive/folders/1aljipIKgMW7unbS_lwOIASlJEcH0s5Ci?usp=share_link


Start by cloning this repository


Once dvc is installed, navigate to your cloned repository (/CNN_BELA)

```
dvc init
# add a remote repo and pull the data
dvc remote add --default younameit gdrive://1aljipIKgMW7unbS_lwOIASlJEcH0s5Ci
dvc pull
# At this point your browser should open in order to authorize dvc
# If authorized, the models will be downloaded in /CNN_BELA/checkpoints_saves
```

Additional data (training images, classification results) can be found here: https://osf.io/t2xns/
