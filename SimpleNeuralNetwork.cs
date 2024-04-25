using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;

public class SimpleNeuralNetwork
{
    public float[][] InputToHiddenWeights;
    public float[][] HiddenToOutputWeights;
    public float[] HiddenBias;
    public float[] OutputBias;
    private float[] hiddenLayerOutputs;

    public SimpleNeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    {
        InputToHiddenWeights = InitializeWeights(inputSize, hiddenSize);
        HiddenToOutputWeights = InitializeWeights(hiddenSize, outputSize);
        HiddenBias = new float[hiddenSize];
        OutputBias = new float[outputSize];
    }

    private float[][] InitializeWeights(int fromSize, int toSize)
    {
        System.Random rand = new System.Random();
        float[][] weights = new float[fromSize][];
        // Calculate the standard deviation for He initialization
        double stddev = Math.Sqrt(2.0 / fromSize);

        for (int i = 0; i < fromSize; i++)
        {
            weights[i] = new float[toSize];
            for (int j = 0; j < toSize; j++)
            {

                double u1 = 1.0 - rand.NextDouble();
                double u2 = 1.0 - rand.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                       Math.Sin(2.0 * Math.PI * u2);
                weights[i][j] = (float)(stddev * randStdNormal);
            }
        }
        return weights;
    }


    public float[] Forward(int[] inputs)
    {
        float[] hiddenLayerInputs = new float[InputToHiddenWeights[0].Length];
        for (int i = 0; i < InputToHiddenWeights.Length; i++)
        {
            for (int j = 0; j < InputToHiddenWeights[i].Length; j++)
            {
                hiddenLayerInputs[j] += inputs[i] * InputToHiddenWeights[i][j];
            }
        }


        hiddenLayerOutputs = hiddenLayerInputs.Select(x => Math.Max(0, x + HiddenBias[0])).ToArray();

        float[] outputLayerInputs = new float[HiddenToOutputWeights[0].Length];
        for (int i = 0; i < HiddenToOutputWeights.Length; i++)
        {
            for (int j = 0; j < HiddenToOutputWeights[i].Length; j++)
            {
                outputLayerInputs[j] += hiddenLayerOutputs[i] * HiddenToOutputWeights[i][j];
            }
        }

        // Apply softmax activation
        float[] outputLayerOutputs = Softmax(outputLayerInputs.Select(x => x + OutputBias[0]).ToArray());
        return outputLayerOutputs;
    }

    private float[] Softmax(float[] z)
    {
        float max = z.Max();
        float scale = z.Sum(t => (float)Math.Exp(t - max));
        return z.Select(t => (float)Math.Exp(t - max) / scale).ToArray();
    }


    public void Train(List<int[]> trainingInputs, List<int[]> trainingOutputs, int epochs, float learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalError = 0;
            for (int i = 0; i < trainingInputs.Count; i++)
            {
                // Forward pass
                int[] input = trainingInputs[i];
                int[] trueOutput = trainingOutputs[i];
                float[] predictedOutput = Forward(input);

                // Calculate error (Cross-Entropy Loss) for logging purposes
                for (int j = 0; j < trueOutput.Length; j++)
                {
                    totalError += (float)(-trueOutput[j] * Math.Log(predictedOutput[j]));
                }

                // Calculate the gradient of the loss function w.r.t. the output of the softmax layer
                float[] outputLayerGradient = new float[trueOutput.Length];
                for (int j = 0; j < trueOutput.Length; j++)
                {
                    outputLayerGradient[j] = predictedOutput[j] - trueOutput[j];
                }

                // Update weights for Hidden to Output layer
                for (int j = 0; j < HiddenToOutputWeights.Length; j++)
                {
                    for (int k = 0; k < HiddenToOutputWeights[j].Length; k++)
                    {
                        float weightUpdate = learningRate * outputLayerGradient[k] * hiddenLayerOutputs[j];
                        HiddenToOutputWeights[j][k] -= weightUpdate;
                    }
                }

                // Calculate gradients for the hidden layer
                float[] hiddenLayerGradient = new float[HiddenBias.Length];
                for (int j = 0; j < hiddenLayerOutputs.Length; j++)
                {
                    float gradient = 0;
                    for (int k = 0; k < outputLayerGradient.Length; k++)
                    {
                        gradient += outputLayerGradient[k] * HiddenToOutputWeights[j][k];
                    }
                    // Apply the derivative of the ReLU function
                    hiddenLayerGradient[j] = gradient * (hiddenLayerOutputs[j] > 0 ? 1 : 0);
                }

                // Update weights for Input to Hidden layer
                for (int j = 0; j < InputToHiddenWeights.Length; j++)
                {
                    for (int k = 0; k < InputToHiddenWeights[j].Length; k++)
                    {
                        float weightUpdate = learningRate * hiddenLayerGradient[k] * input[j];
                        InputToHiddenWeights[j][k] -= weightUpdate;
                    }
                }
            }

            // Log the total error at the end of each epoch
            Debug.Log($"Epoch {epoch + 1}, Total Error: {totalError}");
        }
    }


}

