#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <tgmath.h>
#include <regex>
#include <chrono>

// I'm not using "using namespace std" because I want to see how ugly the code
// turns out to be.

class LogisticRegressionModel
{
public:
	LogisticRegressionModel(std::string filename, std::vector<int> predictors, int target, double learning_rate, int iterations = 600, int train_num = 800)
	{
		this->filename = filename;
		this->df = this->read_csv();
		this->predictors = predictors;
		this->target = target;
		this->learning_rate = learning_rate;
		this->iterations = iterations;
		this->train_num = train_num;
		this->weights = this->init_weights();
	}
	std::string filename;
	std::vector<std::vector<double>> df;
	std::vector<int> predictors;
	int target;
	double learning_rate;
	int iterations;
	int train_num;
	double training_time;
	std::vector<std::vector<double>> train;
	std::vector<std::vector<double>> test;
	// I originally had weights as a matrix as well, but once I gave up on
	// actually implementing matrix multiplication cause I was too sleepy, I
	// I realized weights works as just a vector.
	std::vector<double> weights;
	std::vector<std::vector<double>> data_matrix;

	void run_lr();
	void predict();
	std::vector<std::vector<double>> read_csv();
	std::vector<double> init_weights();
	std::vector<double> sigmoid(std::vector<std::vector<double>> &pmat);
};

std::vector<double> LogisticRegressionModel::init_weights()
{
	std::vector<double> weights;
	for (int i = 0; i < this->predictors.size() + 1; i++)
	{
		weights.push_back(1.0);
	}

	return weights;
}

void LogisticRegressionModel::run_lr()
{
	// We already loaded the data during construction, so start by dividing the
	// data into training and test sets.
	for (int i = 0; i < this->train_num; i++)
	{
		this->train.push_back(this->df[i]);
	}

	for (int i = this->train_num; i < this->df.size(); i++)
	{
		this->test.push_back(this->df[i]);
	}

	// Initialize the data_matrix with a column of ones for the intercept, and
	// our predictor columns.
	for (int i = 0; i < this->train.size(); i++)
	{
		std::vector<double> row = {1.0};
		for (int j = 0; j < this->predictors.size(); j++)
		{
			row.push_back(this->train[i][this->predictors[j]]);
		}
		data_matrix.push_back(row);
	}

	// I'm tracking training time starting here, because this is when the model
	// starts to iteratively fit the data. That's where the significant time is
	// spent.
	auto start = std::chrono::high_resolution_clock::now();

	// Iterate 'iterations' times, updating weights each time.
	for (int i = 0; i < this->iterations; i++)
	{
		// Now, instead of doing matrix multiplication, which would require transposing
		// the data_matrix, we can just iterate over the rows and columns of the
		// both weights and the data_matrix.
		std::vector<std::vector<double>> weighted_data;
		for (int j = 0; j < this->data_matrix.size(); j++)
		{
			std::vector<double> row{0.0};
			for (int k = 0; k < this->weights.size(); k++)
			{
				row[0] += (this->data_matrix[j][k] * this->weights[k]);
			}
			weighted_data.push_back(row);
		}

		std::vector<double> probs = this->sigmoid(weighted_data);

		// Update current error vector
		std::vector<double> errors = std::vector<double>(this->train.size(), 0);
		for (int j = 0; j < this->train.size(); j++)
		{
			errors[j] = this->train[j][this->target] - probs[j];
		}

		// Apply error to weights
		for (int j = 0; j < this->weights.size(); j++)
		{
			double error_sum = 0;
			for (int k = 0; k < errors.size(); k++)
			{
				error_sum += errors[k] * this->data_matrix[k][j];
			}
			this->weights[j] += this->learning_rate * error_sum;
		}
	}

	auto finish = std::chrono::high_resolution_clock::now();

	training_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();

	std::cout << "Coefficients: " << std::endl;
	std::cout << "Intercept: " << this->weights[0] << std::endl;
	for (int i = 1; i < this->weights.size(); i++)
	{
		std::cout << "Coefficient " << i << ": " << this->weights[i] << std::endl;
	}
}

// Function to return a vector of sigmoid values from an input matrix-
// but in C++! We take the matrix as a matrix of doubles, and loop through each
// value in the matrix, adding a sigmoid value to the output vector.
// Note: This is C++, I'm passing by reference because I'm scared of pointers
//       and what to cover my bases here. I think it's this by default anyhow.
std::vector<double> LogisticRegressionModel::sigmoid(std::vector<std::vector<double>> &pmat)
{
	std::vector<double> sigmoids;
	for (int i = 0; i < pmat.size(); i++)
	{
		for (int j = 0; j < pmat[i].size(); j++)
		{
			// I actually found a stack overflow talking about how this might
			// not be the fastest sigmoid function, just noting it here!
			// (https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm)
			sigmoids.push_back(1 / (1 + std::exp(-pmat[i][j])));
		}
	}
	return sigmoids;
}

// Read the csv file into a matrix that will serve as a replacement for our R
// data frame.
std::vector<std::vector<double>> LogisticRegressionModel::read_csv()
{
	std::ifstream inFS;
	std::string line;
	std::vector<std::vector<double>> matrix;

	inFS.open(filename);
	if (!inFS.is_open())
	{
		std::cout << "Could not open file " << filename << std::endl;
	}

	// The first line is the header, discard.
	std::getline(inFS, line);

	// Read each observation/line, and read each attribute of each observation
	// into a vector, then add that to the matrix.
	while (std::getline(inFS, line))
	{
		// Best way to do this found on stack overflow yet again
		// (https://stackoverflow.com/questions/5607589/right-way-to-split-an-stdstring-into-a-vectorstring)
		std::vector<double> row;
		std::stringstream ss(line);
		std::string data;
		while (std::getline(ss, data, ','))
		{
			// Clean up the data so it is just numeric. Remove quotes from the index
			data = std::regex_replace(data, std::regex("\""), "");
			row.push_back(std::stod(data));
		}
		matrix.push_back(row);
	}

	return matrix;

	inFS.close();
}

void LogisticRegressionModel::predict()
{
	// We can use data_matrix again but this time apply our found weights to
	// the test data.
	data_matrix.clear();
	for (int i = 0; i < this->test.size(); i++)
	{
		std::vector<double> row = {1.0};
		for (int j = 0; j < this->predictors.size(); j++)
		{
			row.push_back(this->test[i][this->predictors[j]]);
		}
		data_matrix.push_back(row);
	}

	// We get to reuse the weight matrix multiplication code from building the model.
	std::vector<std::vector<double>> prediction;
	for (int j = 0; j < this->data_matrix.size(); j++)
	{
		std::vector<double> row{0.0};
		for (int k = 0; k < this->weights.size(); k++)
		{
			row[0] += (this->data_matrix[j][k] * this->weights[k]);
		}
		prediction.push_back(row);
	}

	std::vector<double> probs = this->sigmoid(prediction);
	// Counting the confusion matrix data so I can calculate metrics later!
	int tp = 0; // True Positive
	int tn = 0; // True Negative
	int fp = 0; // False Positive
	int fn = 0; // False Negative
	for (int i = 0; i < probs.size(); i++)
	{
		// I frankly don't know what this does.
		// probs[i] = probs[i] * std::exp(probs[i]);
		if (probs[i] > .5)
		{
			probs[i] = 1;
		}
		else
		{
			probs[i] = 0;
		}
		if (probs[i] == 1 && this->test[i][this->target] == 1)
		{
			tp++;
		}
		else if (probs[i] == 0 && this->test[i][this->target] == 0)
		{
			tn++;
		}
		else if (probs[i] == 1 && this->test[i][this->target] == 0)
		{
			fp++;
		}
		else if (probs[i] == 0 && this->test[i][this->target] == 1)
		{
			fn++;
		}
	}

	// The amount of true predictions divided by the total number of predictions
	std::cout << "Accuracy: " << (double)(tp + tn) / (double)probs.size() << std::endl;
	// The amount of true positives divided by the total number of positives
	// This is an indicator how well a model can predict the positive class
	std::cout << "Sensitivity: " << (double)tp / (double)(tp + fn) << std::endl;
	// The amount of true negatives divided by the total number of negatives
	// This is an indicator how well a model can predict the negative class
	std::cout << "Specificity: " << (double)tn / (double)(tn + fp) << std::endl;
	// Training Time: From the
	std::cout << "Training Time: " << this->training_time << "ms" << std::endl;

	// Print the confusion matrix because Gray told me to.
	std::cout << "Confusion Matrix:" << std::endl;
	std::cout << "P. -> 0 1" << std::endl;
	std::cout << "0 " << tn << " " << fn << std::endl;
	std::cout << "1 " << fp << " " << tp << std::endl;
}

// main() just calls the function that initiates the program, but could be used
// to set up command line interfacing. For now just runs the assigned scenario.
int main()
{
	// This program is going to build a model given the input of:
	// 1. The file name of a csv file of the data where each observation is a
	//    row where the first column is just the row number
	// 2. A vector of the column numbers of predictors
	// 3. The column number of the target variable
	// 4. The learning rate
	//
	// We will eventually be printing:
	// 1. The coefficients of our predictors
	// 2. The intercept
	// 3. Metrics:
	//    1. Accuracy
	//    2. Sensitivity
	//    3. Specificity
	// 4. Training Time
	std::vector<int> predictors;
	int target;
	double learning_rate;
	std::string filename;

	std::cout << "Run the 'Sex' Predictor Scenario at 600 iterations" << std::endl;

	// Set data for 1 predictor
	// Column of 'sex' attribute
	target = 2; // Column of 'survived' attribute
	learning_rate = .001;
	filename = "titanic_project.csv";

	predictors.push_back(3);
	// Run logistic regression (Builds the model, predicts, evaluates)
	LogisticRegressionModel lr1(filename, predictors, target, learning_rate, 600);
	lr1.run_lr();

	// !!! It was after hours of work I was informed that I didn't need to program
	//    the multiple predictor scenarios. The  foundation for expanding the
	//    model is still here, I just don't quite know how to do it myself. !!!

	// Just to note, these results match what R would produce!
	lr1.predict();
}
