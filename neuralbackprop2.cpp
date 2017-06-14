#include <iostream>
#include <cstdlib>
#include <ctime>
#include <math.h>
using namespace std;

const float alpha = 0.05;
const float noisemax = 0.4;

//embody a single multilevel neural network with 1 binary output
class Perceptron
{
public:
	//get a binary prediction, given array of binary input
	int getPrediction(int*);
	//make it learn...pass it an array of binary inputs, and desired binary output
	void train(int*, int);
	//constructor - pass it however many input nodes
	Perceptron(int);
	//a neural network needs to store:
	//an array of output weights
	float* outputweight;
	//two dimensional array of hidden weights
	float** hiddenweight;
	//how many inputs and hiddens
	int size;
};

//a threshold function like >=0, but differentiable
float sigmoid(float x)
{
	return 1/(1+pow(2.71828,-x));
}

//constructor, takes a number of input nodes
//making all the arrays
//initializing the weights +/- random values within a range
Perceptron::Perceptron(int nodes)
{
	//save number of nodes
	size=nodes;
	//array of floats for the output weights, +1 for the bias
	outputweight=new float[nodes+1];
	//go through each output weight and make a random #
	for(int i=0; i<nodes+1; i++)
		//random weight between -noisemax/2, noisemax/2
		outputweight[i]=(float)(((rand()%1000) / 1000.0)*noisemax - noisemax/2);
	//array of array for the hiddens, +1 for the bias	
	hiddenweight = new float*[nodes+1];
	for(int i=0; i<nodes+1; i++)
	{
		//make the 2nd dimension and give everyone a random value
		hiddenweight[i]=new float[nodes];
		for(int j=0; j<nodes; j++)
			hiddenweight[i][j]=(float)(((rand()%1000) / 1000.0)*noisemax - noisemax/2);
	}

}

//getting a binary prediction of either 0,1 from a array of binary inputs 0,1
int Perceptron::getPrediction(int* inputs)
{
	//making an array of hidden values, same size as the input
	float hidden[size];

	//for each hidden node
	for (int hn=0; hn<size; hn++)
	{
		//sum up all the input times weight
		float sum = 0;
		for (int i=0; i<size; i++)
			//making it -1/1 instead of 0/1 and then do the dot product
			sum += (inputs[i]==0?-1:1) * hiddenweight[hn][i];
		//throw in the bias of 1
		sum+=1 * hiddenweight[hn][size];
		//sigmoiding it!
		//this is our threshold, effectively >=0?		
		hidden[hn] = sigmoid(sum);
	}
	
	//do it all over again for the output
	float sum=0;
	for(int i=0; i<size; i++)
	{
		sum+= hidden[i] * outputweight[i];
	}
	sum+=outputweight[size];
	//if it's closer to 1, return 1, otherwise return 0
	return sigmoid(sum)>=0.5? 1:0;
}

//trains the neuron to get correct prediction
void Perceptron::train(int* inputs, int want)
{
	//recalculating getPrediction, because we want the hidden intermediate values
	float hidden[size];

	//same as getPrediction to get the hidden values for determining error
	for (int hn=0; hn<size; hn++)
	{
		float sum = 0;
		for (int i=0; i<size; i++)
			sum += (inputs[i]==0?-1:1) * hiddenweight[hn][i];
		sum+=hiddenweight[hn][size];
		hidden[hn] = sigmoid(sum);
	}

	float sum=0;
	for(int i=0; i<size; i++)
	{
		sum+= hidden[i] * outputweight[i];
	}
	sum+=outputweight[size];
	float prediction =  sigmoid(sum);

	//now to train!

	//error is badness * doutput/dinput (how much off multiplied by derivative of sigmoid)
	float error = (want - prediction) * prediction * (1-prediction);
	//every little perceptron gets its own error
	float hiddenerror[size+1];

	//for each hidden node
	for(int i=0; i<size; i++)
	{
		//hidden error is outputerror * dhidden/dinput
		hiddenerror[i]=hidden[i]*(1-hidden[i])*outputweight[i]*error;
		//weight += error * input-to-the-output-layer
		//throw in alpha - learning rate
		outputweight[i]+= error * hidden[i] * alpha;
	}
	//do it for the bias weight
	outputweight[size]+=alpha * error;
	//for each hidden node
	for(int hn=0; hn<size; hn++)
	{
		//for each input weight to that hidden node
		for(int i=0; i<size; i++)
		{
			//same formula: error * input * alpha
			hiddenweight[hn][i]+= alpha * hiddenerror[hn] * (inputs[i]==0?-1:1);
		}
		hiddenweight[hn][size]+= alpha * hiddenerror[hn];
	}
}

main()
{
	//random number seed based on time
	//we need random numbers for the initial weights
 	std::srand(std::time(0)); 
	
	//instantiate new neural net with 2 inputs
	Perceptron* neuron=new Perceptron(2);
	
	//instantiate data array for training
	int data[2];

	//train 1000000 times
	for (int i=0; i<1000000; i++)
	{
		data[0]=0; data[1]=0;
		neuron->train(data,0);
		data[0]=0; data[1]=1;
		neuron->train(data,1);
		data[0]=1; data[1]=0;
		neuron->train(data,1);
		data[0]=1; data[1]=1;
		neuron->train(data,1);
	}
	
	//printing the result of the training
	data[0]=0; data[1]=0;
	cout<<"0,0 is " <<neuron->getPrediction(data)<<endl;
	data[0]=0; data[1]=1;
	cout<<"0,1 is "<<neuron->getPrediction(data)<<endl;
	data[0]=1; data[1]=0;
	cout<<"1,0 is "<<neuron->getPrediction(data)<<endl;
	data[0]=1; data[1]=1;
	cout<<"1,1 is "<<neuron->getPrediction(data)<<endl;
	
}
