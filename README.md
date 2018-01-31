# Face-feature-extraction-SqueezeNet-CenterLoss

This is an implementation of feature extraction method with [center loss function](https://ydwen.github.io/papers/WenECCV16.pdf) and the small network [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf). The network is trained on [CASIA_WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) with preprocessing of face cropping and affine transform. 

We tested the verification accuracy on [LFW dataset](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) and got an accuracy of 97.6%.

After training, if you truncate the fully connected layers in SqueezeNet, the caffemodel will be only 2.6 MB.

How to train the model

1, download and make the caffe with center loss (see the [link](https://github.com/ydwen/caffe-face) in the paper)

2, download a proper face dataset(better with cropped face), resize the images to the same size. (I used 226X264)

3, create the anno.txt with each face labelled, see the format in data/anno.txt

4, run the python script (This script is written in python 2.7)




The loss change during training is:

center loss increased at the beginning, but decreased a little approaching the end
<div align="center">
  <img src="https://github.com/HoiM/Face-feature-extraction-SqueezeNet-CenterLoss/blob/master/output/loss/center_loss.jpg"><br><br>
</div>

softmax loss decreased all the way
<div align="center">
  <img src="https://github.com/HoiM/Face-feature-extraction-SqueezeNet-CenterLoss/blob/master/output/loss/softmax_loss.jpg"><br><br>
</div>

the total loss decreased all the way (ratio between center loss and softmax loss is 0.008:1)
<div align="center">
  <img src="https://github.com/HoiM/Face-feature-extraction-SqueezeNet-CenterLoss/blob/master/output/loss/loss.jpg"><br><br>
</div>
