# XPert: Peripheral Circuit & Neural Architecture Co-search for Area and Energy-efficient Xbar-based Computing (Design & Automation Conference, 2023)
The hardware-efficiency and accuracy of Deep Neural Networks (DNNs) implemented on In-memory Computing (IMC) architectures primarily depend on the DNN architecture and the peripheral circuit parameters. It is therefore essential to holistically co-search the network and peripheral parameters to achieve optimal performance. To this end, we propose XPert, which co-searches network architecture in tandem with peripheral parameters such as the type and precision of analog-to-digital converters, crossbar column sharing and the layer-specific input precision using an optimization-based design space exploration. Compared to VGG16 baselines, XPert achieves 10.24x (4.7x) lower EDAP, 1.72x (1.62x) higher TOPS/W,1.93x (3x) higher TOPS/mm2 at 92.46% (56.7%) accuracy for CIFAR10 (TinyImagenet) datasets. The paper is available at https://arxiv.org/abs/2303.17646

# Instructions for running Phase1-Cosearch

## Running the Co-search
Run the Code using 
```
python Phase1_VGG16_backbone.py with the variables described below.
```
## Variable Description 

```
--lr: Learning Rate
--hw_params: Tuple of #PEs per Tile, #Crossbars per PE, #Tiles, Crossbar Size
--target_latency: Target Latency
--target_area: Target Area Constraint (mm^2)
--epochs: Total Number of Search Epochs
--wt_prec: Weight Precision
--cellbit: Number of bits per NVM device
--area_tolerance: The uncerainity in the on-chip area tolerated in the searched model. 
```
# Instructions for XPertSim C++ Evaluation

## Step 1: Create the Network_custom.csv file

The Network_custom.csv file contains the DNN information required for evaluation. For each layer, we append an additional layer to ensure 64 Crossbar arrays per tile. Each row has the following format: 

feature size, feature size, input channels, kernel size, kernel size, output channels, max-pool flag, padding, column sharing, ADC type (1: Flash ADC, 0: SAR ADC), Input Precision, ADC Precision.

## Step 2: Update the VGG.py with the desired model

Add the desired model configurations to the cfg_list and make sure to add the cfg variable in the "def VGG8()" function.

The model can be defined inside the cfg_list by using the following format: 

(Layer Type: ('C': Convolutional layer, 'L': Linear Layer), input channels, output channels, kernel size, output size ('same': if output feature size is same as input feature size), scaling factor. 

Like Network_custom.csv we append each layer with ('C', 64, 512, 3, 'same', 8). 

## Step 3 Run the Evaluation

Run the inference.py using "python inference.py --model custom"


# Acknowledgement
This repository is adapted from the Neurosim platform https://github.com/neurosim/DNN_NeuroSim_V1.3.

