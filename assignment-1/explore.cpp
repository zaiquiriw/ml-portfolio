#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <cmath>

using namespace std;

// TODO: Convert double vectors to taking in (explicitly) any numeric value
// Reference: - Iterators: https://www.geeksforgeeks.org/iterators-c-stl/
//            - What get's passed into sort: https://cplusplus.com/reference/iterator/RandomAccessIterator/

class Explore
{
public:
	// Calculate the sum of the vector
	double sum_vector(vector<double> vect)
	{
		double sum = 0;
		for (int i = 0; i < vect.size(); i++)
		{
			sum += vect[i];
		}
		return sum;
	}

	// Calculate the mean of a vector
	double mean_vector(vector<double> vect)
	{
		double mean = sum_vector(vect) / vect.size();
		return mean;
	}

	// Calculate the median of a vector
	double median_vector(vector<double> vect)
	{
		double median;
		// Use an iterator because it is probably better -internet
		vector<double>::iterator it;
		// Find the center if it is even or odd
		sort(vect.begin(), vect.end());
		if (vect.size() % 2 == 0) // If there is an even number of elements
		{
			it = vect.begin() + vect.size() / 2 - 1;
			median = (*it + *(it + 1)) / 2;
		}
		else // if there is an odd number of elements
		{
			it = vect.begin() + vect.size() / 2;
			median = *it;
		}
		return median;
	}

	// Calculate the range of a vector
	double range_vector(vector<double> vect)
	{
		// I could return the values of min and max to mimic the range function
		// of R, but I think since this lesson is about writing my own code,
		// I will just return the difference between the max and min instead of
		// the min and max.
		return max_vector(vect) - min_vector(vect);
	}

	// Calculate the max of a vector (Just for range)
	double max_vector(vector<double> vect)
	{
		double max;
		vector<double>::iterator it;
		sort(vect.begin(), vect.end());
		it = vect.end() - 1;
		max = *it;
		return max;
	}

	// Calculate the min of a vector (Just for range) just with a loop
	double min_vector(vector<double> vect)
	{
		double min;
		vector<double>::iterator it;
		sort(vect.begin(), vect.end());
		it = vect.begin();
		min = *it;
		return min;
	}

	// Calculate the covariance of two vectors
	// Cov(x,y) = E((x-x_mean)(y-y_mean)))/n-1
	double covar_vector(vector<double> x, vector<double> y)
	{
		double sum = 0;
		double mean_x = mean_vector(x);
		double mean_y = mean_vector(y);
		for (int i = 0; i < x.size(); i++)
		{
			float x_i_diff = x[i] - mean_x;
			float y_i_diff = y[i] - mean_y;
			float y_times_x_diff = x_i_diff * y_i_diff;
			// cout << x_i_diff << " * " << y_i_diff << " = " << y_times_x_diff << endl;
			sum = sum + y_times_x_diff;
		}
		return sum / (x.size() - 1);
	}

	// Calculate the correlation of two vectors
	// Cor(x,y) = Cov(x,y)/(standard_devation(x)*standard_devation(y))
	// Using the hint from the assignment:
	// "sigma of a vector can be calculated as the square root of variance(v,v)"
	double cor_vector(vector<double> x, vector<double> y)
	{
		double covar = covar_vector(x, y);
		double sigma_x = sqrt(covar_vector(x, x));
		double sigma_y = sqrt(covar_vector(y, y));
		return covar / (sigma_x * sigma_y);
	}

	// Run suite of statistcal functions on a vector
	void print_stats(vector<double> vect)
	{
		cout << "Sum:    " << sum_vector(vect) << endl;
		cout << "Mean:   " << mean_vector(vect) << endl;
		cout << "Median: " << median_vector(vect) << endl;
		cout << "Range:  " << range_vector(vect) << endl;
	}
};

int main(int argc, char **argv)
{
	ifstream inFS;
	string line;
	string rm_in, medv_in;
	const int MAX_LEN = 1000;
	vector<double> rm(MAX_LEN), medv(MAX_LEN);

	cout << "Opening file Boston.csv." << endl;

	inFS.open("Boston.csv");
	if (!inFS.is_open())
	{
		cout << "Error opening file Boston.csv." << endl;
		return 1;
	}

	cout << "Reading line 1 of Boston.csv." << endl;
	getline(inFS, line);

	// echo heading
	cout << "Headings: " << line << endl;

	// read data
	int numObservations = 0;
	while (inFS.good())
	{
		getline(inFS, rm_in, ',');
		getline(inFS, medv_in, '\n');
		rm.at(numObservations) = stof(rm_in);
		medv.at(numObservations) = stof(medv_in);

		numObservations++;
	}

	rm.resize(numObservations);
	medv.resize(numObservations);

	cout << "New Length: " << rm.size() << endl;

	cout << "Closing file Boston.csv." << endl;
	inFS.close(); // Done

	cout << "Number of records: " << numObservations << endl;

	// Create an Explore object to use stats functions
	Explore explore;

	cout << "\nStats for rm" << endl;
	explore.print_stats(rm);

	cout << "\nStats for medv" << endl;
	explore.print_stats(medv);

	cout << "\n Covariance = " << explore.covar_vector(rm, medv) << endl;

	cout << "\n Correlation = " << explore.cor_vector(rm, medv) << endl;

	cout << "\nProgram terminated." << endl;
}
