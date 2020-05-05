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
            var net = new DenseNet(new int[] { 3, 25, 25, 1 }); //intiilize network

            var inputs = new List<Matrix>();
            var outputs = new List<Matrix>();

            var m = new Matrix(3, 1);
            var n = new Matrix(1, 1);

            // 0 0 0    => 0
            m[0, 0] = 0; m[1, 0] = 0; m[2, 0] = 0;
            n[0, 0] = 0;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 0 0 1    => 1
            m[0, 0] = 0; m[1, 0] = 0; m[2, 0] = 1;
            n[0, 0] = 1;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 0 1 0    => 1
            m[0, 0] = 0; m[1, 0] = 1; m[2, 0] = 0;
            n[0, 0] = 1;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 0 1 1    => 0
            m[0, 0] = 0; m[1, 0] = 1; m[2, 0] = 1;
            n[0, 0] = 0;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 1 0 0    => 1
            m[0, 0] = 1; m[1, 0] = 0; m[2, 0] = 0;
            n[0, 0] = 1;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 1 0 1    => 0
            m[0, 0] = 1; m[1, 0] = 0; m[2, 0] = 1;
            n[0, 0] = 0;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 1 1 0    => 0
            m[0, 0] = 1; m[1, 0] = 1; m[2, 0] = 0;
            n[0, 0] = 0;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 1 1 1    => 1
            m[0, 0] = 1; m[1, 0] = 1; m[2, 0] = 1;
            n[0, 0] = 1;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            //Itterate 5000 times and train each possible output
            //5000*8 = 40000 traning operations
            for (var i = 0; i < 5000; i++)
            {
                net.FeedForward(inputs[0]);
                net.BackProp(outputs[0]);

                net.FeedForward(inputs[1]);
                net.BackProp(outputs[1]);

                net.FeedForward(inputs[2]);
                net.BackProp(outputs[2]);

                net.FeedForward(inputs[3]);
                net.BackProp(outputs[3]);

                net.FeedForward(inputs[4]);
                net.BackProp(outputs[4]);

                net.FeedForward(inputs[5]);
                net.BackProp(outputs[5]);

                net.FeedForward(inputs[6]);
                net.BackProp(outputs[6]);

                net.FeedForward(inputs[7]);
                net.BackProp(outputs[7]);
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
            Assert.IsTrue(correctness,"Network did not learn XOR");
        }
    }
}
