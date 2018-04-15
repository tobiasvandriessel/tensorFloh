# tensorFloh

Important: We will need Windows to do some steps!

We used C++ to preprocess our data, you have the following data structure:
  - You need to have the data folder structure as said in part A. 
  - So in UFC-101 you need to have the different class videos + a folds folder with 5 folders in it with names (1, 2, 3, 4, 5)
  - Have a Own data with our data provided in part A
  - WINDOWS needed for this: Read the README in the data/ folder included in the zip file and follow the instructions

Now the videos are put into seperate folds and we can process them for the offline part.
 
Now we will need to run the C++ code, which was based on the Tensorflow Base program distributed on slack: (link)[https://drive.google.com/file/d/1h4PfHsoPoOISOSMzPQMb6DR1RJh2vVPL/view?usp=sharing] . Copy all our files in there and overwrite them. Possibly, you will need to adjust the project settings to load the opencv libraries from your path. We compiled opencv from source to be able to use the extra modules, but I don't think that's necessary. The only thing that might not work is the writeflofile. Now you can run the c++ code which should create the mirror data and the normal data.
  - In the python train_model.py you can find the training of the models. 
  - As you can see there is multiple models that you can run with different dropouts and epoches, adjust them to your liking.
  - simply run the script
