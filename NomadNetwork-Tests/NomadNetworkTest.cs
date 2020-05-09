using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using NomadNetworkLibrary;
using System.Diagnostics;
using static System.Math;

namespace NomadNetwork_Tests
{
    [TestClass]
    public class NomadNetworkTest
    {
        [TestMethod]
        public void DenseNetXor()
        {
            var net = new DenseNet(new int[] { 3, 25, 25, 1 });

            var inputs = new List<Matrix>();
            var outputs = new List<Matrix>();

            // 0 0 0    => 0
            inputs.Add(new Matrix(new double[,] { { 0.0 }, { 0.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new double[,] { { 0.0 } }));

            // 0 0 1    => 1
            inputs.Add(new Matrix(new double[,] { { 0.0 }, { 0.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new double[,] { { 1.0 } }));

            // 0 1 0    => 1
            inputs.Add(new Matrix(new double[,] { { 0.0 }, { 1.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new double[,] { { 1.0 } }));

            // 0 1 1    => 0
            inputs.Add(new Matrix(new double[,] { { 0.0 }, { 1.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new double[,] { { 1.0 } }));

            // 1 0 0    => 1
            inputs.Add(new Matrix(new double[,] { { 1.0 }, { 0.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new double[,] { { 1.0 } }));

            // 1 0 1    => 0
            inputs.Add(new Matrix(new double[,] { { 1.0 }, { 0.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new double[,] { { 0.0 } }));

            // 1 1 0    => 0
            inputs.Add(new Matrix(new double[,] { { 1.0 }, { 1.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new double[,] { { 0.0 } }));

            // 1 1 1    => 1
            inputs.Add(new Matrix(new double[,] { { 1.0 }, { 1.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new double[,] { { 1.0 } }));

            for (var i = 0; i < 50; i++)
            {
                net.FeedForward(inputs[i % 8]);
                net.BackProp(outputs[i % 8]);
            }

            var correct = 0;
            for (var i = 0; i < 125; i++)
            {
                correct += Abs(net.FeedForward(inputs[0])[0, 0]) < 0.1 ? 1 : 0;
                correct += Abs(net.FeedForward(inputs[1])[0, 0]) - 1 < 0.1 ? 1 : 0;
                correct += Abs(net.FeedForward(inputs[2])[0, 0]) - 1 < 0.1 ? 1 : 0;
                correct += Abs(net.FeedForward(inputs[3])[0, 0]) < 0.1 ? 1 : 0;
                correct += Abs(net.FeedForward(inputs[4])[0, 0]) - 1 < 0.1 ? 1 : 0;
                correct += Abs(net.FeedForward(inputs[5])[0, 0]) < 0.1 ? 1 : 0;
                correct += Abs(net.FeedForward(inputs[6])[0, 0]) < 0.1 ? 1 : 0;
                correct += Abs(net.FeedForward(inputs[7])[0, 0]) - 1 < 0.1 ? 1 : 0;
            }

            var acc = (double)correct / 1000.0 * 100;

            var correctness = true;
            correctness &= Abs(net.FeedForward(inputs[0])[0, 0]) < 0.1;
            correctness &= Abs(net.FeedForward(inputs[1])[0, 0]) - 1 < 0.1;
            correctness &= Abs(net.FeedForward(inputs[2])[0, 0]) - 1 < 0.1;
            correctness &= Abs(net.FeedForward(inputs[3])[0, 0]) < 0.1;
            correctness &= Abs(net.FeedForward(inputs[4])[0, 0]) - 1 < 0.1;
            correctness &= Abs(net.FeedForward(inputs[5])[0, 0]) < 0.1;
            correctness &= Abs(net.FeedForward(inputs[6])[0, 0]) < 0.1;
            correctness &= Abs(net.FeedForward(inputs[7])[0, 0]) - 1 < 0.1;

            Trace.WriteLine(" Acc: " + acc);
            Assert.IsTrue(correctness, "Network did not learn XOR");
        }
    }
}
