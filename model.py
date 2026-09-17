import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.model=nn.Sequential(
            #image is of from (channel,height,width)=(3,128,128)
            # so input has three channels
            #`1st Block-----------------------------------------------------------
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            #so here basically We applying 32 different learnable filters to the input image,
            #producing 32 feature maps.
            nn.ReLU(),
            # at this point we have 32*128*128
        
            nn.Conv2d(32,32,kernel_size=3),
            #padding is not used here so the height and width will decrease by 2
            #1st layer can learn relatively simpler pattern/feature by doing convolution on 
            #those features model can learn relatively complex pattern
            nn.ReLU(),

            #at this point we have 32*128*128
            nn.MaxPool2d(kernel_size=2,stride=2),

            #this is kind of down sampling and at this stage we will have 32*63*63

            #The first block works with relatively detailed spatial information 
            #and begins extracting basic visual features.
                                        #---------------------------#
            #2nd block------------------
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            #now we have 64 feature maps so 64*63*63

            nn.Conv2d(64, 64, kernel_size=3),
            #padding is not used here so the height and width will decrease by 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #now we have 64*30*30
                                #---------------------------------#
            #3rd block
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            #now we have 128 feature maps so 128*30*30

            nn.Conv2d(128, 128, kernel_size=3),
            #padding is not used here so the height and width will decrease by 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
                                            #---------------------------------#

            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            #padding is not used here so the height and width will decrease by 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #now we have 256*6*6


            #adding one more layer to solve underfitting
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            #padding is not used here so the height and width will decrease by 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Dropout(0.25),#to avoid overfitting droping out 25% feature matrix
            
            #fully connected layer
            nn.Flatten(),#after flatening we will have 512*2*2 input so we will map them to 1024(increased to 1500) neuron
            nn.Linear(512 * 2 * 2, 1500),#increases the no of neuron
            nn.ReLU(),
            #o/p layer
            nn.Dropout(p=0.4),#to avoid overfitting dropping 40% neuron
            nn.Linear(1500,38),
            
            #we dont use max_soft as we will use cross-entropy for validation which uses softmax
            #by it self
        )

    def forward(self,x):
        return self.model(x)