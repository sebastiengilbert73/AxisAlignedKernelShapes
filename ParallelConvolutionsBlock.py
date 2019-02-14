import torch
import torch.nn
import torch.nn.functional
import torchvision
import PIL.Image
import ast
import argparse


class ParallelConvolutionsBlock(torch.nn.Module):
    def __init__(self,
                 inputNumberOfChannels,
                 kernelDimensionsList, # Ex.: [(9, 3), (5, 5), (3, 9)] (H, W)
                 kernelsOutputNumbers, # Ex: [14, 4, 14]. The block output number of channels will be the sum of all the kernel outputs (concatenation)
                 dropoutRatio
                 ):
        super(ParallelConvolutionsBlock, self).__init__()
        #print ("ParallelConvolutionsBlock.__init__(): kernelDimensionsList = {}; kernelsOutputNumbers = {}".format(kernelDimensionsList, kernelsOutputNumbers))
        if len(kernelDimensionsList) != len(kernelsOutputNumbers):
            raise ValueError("ParallelConvolutionsBlock.__init__(): len(kernelDimensionsList) ({}) != len(kernelsOutputNumbers) ({})".format(len(kernelDimensionsList), len(kernelsOutputNumbers)))
        # Check that each dimension is odd
        for kernelNdx in range(len(kernelDimensionsList)):
            kernelSize = kernelDimensionsList[kernelNdx]
            if len (kernelSize) != 2:
                raise ValueError("ParallelConvolutionsBlock.__init__(): The kernel size {} is not of length 2".format(kernelSize))
            if kernelSize[0] % 2 != 1:
                raise ValueError("ParallelConvolutionsBlock.__init__(): The kernel size {} first element is not odd".format(kernelSize))
            if kernelSize[1] % 2 != 1:
                raise ValueError("ParallelConvolutionsBlock.__init__(): The kernel size {} second element is not odd".format(kernelSize))

        self.kernelsList = torch.nn.ModuleList()

        for kernelNdx in range(len(kernelDimensionsList)):
            kernel = torch.nn.Conv2d(in_channels=inputNumberOfChannels,
                                     out_channels=kernelsOutputNumbers[kernelNdx],
                                     kernel_size=kernelDimensionsList[kernelNdx],
                                     stride=1,
                                     padding=(int(kernelDimensionsList[kernelNdx][0]/2), int(kernelDimensionsList[kernelNdx][1]/2))
                                     )
            self.kernelsList.append(kernel)

        self.dropout = torch.nn.Dropout(p=dropoutRatio)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        activationsList = []

        # Compute the convolution with each kernel
        for kernel in self.kernelsList:
            activation = kernel(inputs)
            activation = self.relu(activation)
            activationsList.append(activation)

        # Concatenate the activations
        concatenatedActivations = torch.cat(activationsList, dim=1) # We concatenate along C, the channels dimension. Note that all other dimensions must be the same.

        # Dropout
        return self.dropout(concatenatedActivations)


def main():
    print ("ParallelConvolutionsBlock.py main()")
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernelDimensionsListList', help="The list of lists of kernel dimensions. Default: '[[(9,3),(5,5),(3,9)],[(9,3),(5,5),(3,9)]]'", default='[[(9,3),(5,5),(3,9)],[(9,3),(5,5),(3,9)]]')
    args = parser.parse_args()

    args.kernelDimensionsListList = ast.literal_eval(args.kernelDimensionsListList)

    for blockNdx in range(len(args.kernelDimensionsListList)):
        block = ParallelConvolutionsBlock(inputNumberOfChannels=3,
                                      kernelDimensionsList=args.kernelDimensionsListList[blockNdx],
                                      kernelsOutputNumbers=[3,4,5],
                                      dropoutRatio=0.5)

if __name__ == '__main__':
    main()