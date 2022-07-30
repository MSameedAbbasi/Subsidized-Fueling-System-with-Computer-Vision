# Subsidized Fueling System with Computer Vision

Government provides fuel subsidies when the fuel costs get too expensive for the average civilian. This subsidy reduces the cost of fuel for the end buyer by a fixed amount, and the government bears any payment difference thus everyone gets equal profit from this subsidy, however a person driving a motorbike or a budget vehicle is more in need of this subsidy as compared to a large luxury vehicle owner. 

This system provides a solution to achieve this goal. Subsidized Fueling System (SFS) is a software prototype that assists the already existing fuel refilling and distribution systems to independently charge a different amount to every person that buys fuel, this charged price is dependent on the type of car that is being refueled.

The prototype uses Computer Vision to capture the vehicles arriving at the station for refueling through a CCTV camera (Webcam is use here in the prototype). The vehicles are captured and the image is fed to a Machine learning model. The model then predicts the category of vehicle i.e. Small, Medium, Large and the application then displays the current rate of fuel after subsidy as per the government's decision. 

A Pakistani Dataset of different vehicles was prepared which included bikes, rickshaws, different categories of cars from cheap to luxurious. The dataset included 4,034 images out of which 3205 were used as a training set, 419 were used as a validation set and the remaining 410 were used as a test set. The target categories of the vehicles were set to three. Transfer learning was carried out on two models: Resnet50 and InceptionResnet v2.

## Resnet50

- Total Parameters: 23,888,771
- Trainable Parameters: 301,059
- Epochs: 50
- Test Accuracy: 56.8%

## InceptionResnet v2

- Total Parameters: 54,631,651
- Trainable Parameters: 294,915
- Epochs: 50
- Test Accuracy: 92.38%
