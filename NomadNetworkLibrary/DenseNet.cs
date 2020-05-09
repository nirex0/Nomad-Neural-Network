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

            for (var i = 0; i < _layers.Length; i++)
                _layers[i] = new Layer(layer[i], layer[i + 1], i);
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
            for (var i = 1; i < _layers.Length; i++) _layers[i].FeedForward(_layers[i - 1].A);

            return _layers[^1].A; //return output of last layer
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
                    _layers[i].BackPropHidden(_layers[i + 1].Dz, _layers[i + 1].W); //back prop hidden

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
            public int Index { get; }

            public Matrix A; // Outputs
            public Matrix Z; // Transfer
            public Matrix X; // Inputs
            public Matrix W; // Weights
            public Matrix dw; // Delta Weights
            public Matrix Dz; // Delta Transfer
            public Matrix Error; // J(W)

            public static Random Random = new Random();

            /// <summary>
            /// Constructor initilizes vaiour data structures
            /// </summary>
            /// <param name="numberOfInputs">Number of neurons in the previous layer</param>
            /// <param name="numberOfOuputs">Number of neurons in the current layer</param>
            /// <param name="index">Index of layer in network</param>
            public Layer(int numberOfInputs, int numberOfOuputs, int index)
            {
                Index = index;
                NumberOfInputs = numberOfInputs;
                NumberOfOuputs = numberOfOuputs;

                //initilize datastructures
                A = new Matrix(numberOfOuputs, 1);
                X = new Matrix(numberOfInputs, 1);
                W = new Matrix(numberOfOuputs, numberOfInputs);
                dw = new Matrix(numberOfOuputs, numberOfInputs);
                Dz = new Matrix(numberOfOuputs, 1);
                Error = new Matrix(numberOfOuputs, 1);

                InitilizeWeights(); //initilize weights
            }

            /// <summary>
            /// Initilize weights between -0.5 and 0.5
            /// </summary>
            public void InitilizeWeights()
            {
                W.InRandomize(-0.5, 0.5);
                W *= 0.01;
            }

            /// <summary>
            /// Feedforward this layer with a given input
            /// </summary>
            /// <param name="inputs">The output values of the previous layer</param>
            /// <returns></returns>
            public Matrix FeedForward(Matrix inputs)
            {
                X = inputs;
                Z = W * inputs;
                A = Z.Map(Math.Tanh);
                return A;
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
                var da = A - expected;
                var gp = A.Map(TanHDer);
                Dz = da.Hadamard(gp);
                dw = (Dz * X).T();
            }

            /// <summary>
            /// Back propagation for the hidden layers
            /// </summary>
            /// <param name="da">the da value of the forward layer</param>
            /// <param name="weightsFoward">the weights of the forward layer</param>
            public void BackPropHidden(Matrix da, Matrix weightsFoward)
            {
                Dz = weightsFoward.T() * da;
                Dz.InHadamard(A.Map(TanHDer));
                dw = Dz * X.T();
            }

            /// <summary>
            /// Updating weights
            /// </summary>
            public void UpdateWeights()
            {
                W -= dw * 0.03f;
            }
        }
    }
}
