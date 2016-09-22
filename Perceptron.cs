using System;

namespace NeuralNetworks
{
    class Perceptron
    {
        double[][] W;   //weight - trained weights
        double a;       //alpha - training rate
        double t;       //theta - activation level

        public Perceptron(int inputSize, int outputSize, double trainingRate, double activation)
        {
            this.W = new double[outputSize][];
            for (int i = 0; i < this.W.Length; i++)
            {
                this.W[i] = new double[inputSize+1];
                for (int j = 0; j < this.W[i].Length; j++)
                {
                    this.W[i][j] = 0.0;
                }
            }
            this.a = trainingRate;
            this.t = activation;
        }

        public int[]  Process(int[][] X)
        {
            int[] y = new int[X.Length];
            for (int i = 0; i < X.Length; i++)
            {
                for (int j = 0; j < W.Length; j++)
                {
                    double yIn = W[j][0];
                    for (int k = 1; k < W[j].Length; k++)
                    {
                        yIn += W[j][k] * X[i][k - 1];
                    }

                    if (yIn > t) y[i] = 1;
                    else if (-1 * t <= yIn && yIn <= t) y[i] = 0;
                    else y[i] = -1;
                }
            }
            return y;
        }

        public void Train(int[][] X, int[][] target)
        {
            bool noChange = false;
            while (!noChange)
            {
                noChange = true;
                for (int i = 0; i < X.Length; i++)
                {
                    if (!Train(X[i], target[i])) noChange = false;
                }
            }
        }

        public bool Train(int[] X, int[] target)
        {
            bool noChange = true;
            int y = 0;
            for (int j = 0; j < target.Length; j++)
            {
                double yIn = W[j][0];
                for (int k = 1; k < W[j].Length; k++)
                {
                    yIn += W[j][k] * X[k - 1];
                }

                if (yIn > t) y = 1;
                else if (-1 * t <= yIn && yIn <= t) y = 0;
                else y = -1;

                if (y != target[j])
                {
                    noChange = false;
                    W[j][0] += a * target[j];
                    for (int k = 1; k < W[j].Length; k++)
                    {
                        W[j][k] += a * X[k - 1] * target[j];
                    }
                }
            }
            return noChange;
        }

        public override string ToString(){
            string str = "";
            for (int i = 0; i < this.W.Length; i++)
            {
                for (int j = 0; j < this.W[i].Length; j++)
                {
                    str += String.Format("[{0}]", this.W[i][j]);
                }
                str += "\n";
            }
            return str;
        }
    }
}
