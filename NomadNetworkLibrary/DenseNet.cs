using System;
using System.Collections.Generic;
using Nomad.Matrix;

namespace NomadNetworkLibrary
{
    /// <summary>
    /// Simple MLP Neural Network
    /// </summary>
    public class DenseNet
    {
        public int[] LayerInfo { get; }       //layer information
        private readonly Layer[] _layers;    // dense layers in the network

        /// <summary>
        /// Constructor setting up layers
        /// </summary>
        /// <param name="layer">Layers of this network</param>
        public DenseNet(IReadOnlyList<int> layer)
        {
            //deep copy layers
            LayerInfo = new int[layer.Count];
            for (var i = 0; i < layer.Count; i++)
                LayerInfo[i] = layer[i];

            //creates neural layers
            _layers = new Layer[layer.Count - 1];

            for (var i = 0; i < _layers.Length; i++) _layers[i] = new Layer(layer[i], layer[i + 1]);
        }

        /// <summary>
        /// High level feedforward for this network
        /// </summary>
        /// <param name="inputs">Inputs to be feed forwared</param>
        /// <returns></returns>
        public Matrix FeedForward(Matrix inputs)
        {
            //feed forward
            _layers[0].FeedForward(inputs);
            for (var i = 1; i < _layers.Length; i++) _layers[i].FeedForward(_layers[i - 1].Outputs);

            return _layers[^1].Outputs; //return output of last layer
        }

        /// <summary>
        /// High level back porpagation
        /// Note: It is expexted the one feed forward was done before this back prop.
        /// </summary>
        /// <param name="expected">The expected output form the last feedforward</param>
        public void BackProp(Matrix expected)
        {
            // run over all layers backwards
            for (var i = _layers.Length - 1; i >= 0; i--)
                if (i == _layers.Length - 1)
                    _layers[i].BackPropOutput(expected); //back prop output
                else
                    _layers[i].BackPropHidden(_layers[i + 1].Dz, _layers[i + 1].Weights); //back prop hidden

            //Update weights
            foreach (var t in _layers)
                t.UpdateWeights();
        }

        /// <summary>
        /// Each individual layer in the ML{
        /// </summary>
        public class Layer
        {
            public int NumberOfInputs { get; }
            public int NumberOfOuputs { get; }


            public Matrix Outputs; // Z
            public Matrix Inputs; // A (Or X for the first layer)
            public Matrix Weights; // W (TODO: add b)
            public Matrix WeightsDelta; // dJdW
            public Matrix Dz; // dZ
            public Matrix Error; // J(W) TODO: (J, B)

            public static Random Random = new Random();

            /// <summary>
            /// Constructor initilizes vaiour data structures
            /// </summary>
            /// <param name="numberOfInputs">Number of neurons in the previous layer</param>
            /// <param name="numberOfOuputs">Number of neurons in the current layer</param>
            public Layer(int numberOfInputs, int numberOfOuputs)
            {
                NumberOfInputs = numberOfInputs;
                NumberOfOuputs = numberOfOuputs;

                //initilize datastructures
                Outputs = new Matrix(numberOfOuputs, 1);
                Inputs = new Matrix(numberOfInputs, 1);
                Weights = new Matrix(numberOfOuputs, numberOfInputs);
                WeightsDelta = new Matrix(numberOfOuputs, numberOfInputs);
                Dz = new Matrix(numberOfOuputs, 1);
                Error = new Matrix(numberOfOuputs, 1);

                InitilizeWeights(); //initilize weights
            }

            /// <summary>
            /// Initilize weights between -0.5 and 0.5
            /// </summary>
            public void InitilizeWeights()
            {
                Weights.InRandomize(-0.5, 0.5);
            }

            /// <summary>
            /// Feedforward this layer with a given input
            /// </summary>
            /// <param name="inputs">The output values of the previous layer</param>
            /// <returns></returns>
            public Matrix FeedForward(Matrix inputs)
            {
                Inputs = inputs;
                Outputs = (Weights * inputs).Map(Math.Tanh);
                return Outputs;
            }

            /// <summary>
            /// TanH derivate 
            /// </summary>
            /// <param name="value">An already computed TanH value</param>
            /// <returns></returns>
            public double TanHDer(double value)
            {
                return 1 - value * value;
            }

            /// <summary>
            /// Back propagation for the output layer
            /// </summary>
            /// <param name="expected">The expected output</param>
            public void BackPropOutput(Matrix expected)
            {
                Error = Outputs - expected;
                Dz = Error.Hadamard(Outputs.Map(TanHDer));
                WeightsDelta = (Dz * Inputs).T();
            }

            /// <summary>
            /// Back propagation for the hidden layers
            /// </summary>
            /// <param name="da">the da value of the forward layer</param>
            /// <param name="weightsFoward">the weights of the forward layer</param>
            public void BackPropHidden(Matrix da, Matrix weightsFoward)
            {
                Dz = weightsFoward.T() * da;
                Dz.InHadamard(Outputs.Map(TanHDer));
                WeightsDelta = Dz * Inputs.T();
            }

            /// <summary>
            /// Updating weights
            /// </summary>
            public void UpdateWeights()
            {
                Weights -= WeightsDelta * 0.033f;
            }
        }
    }
}
