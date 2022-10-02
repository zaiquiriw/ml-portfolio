#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <math.h>
#include <chrono>

//Naive Bayes from Scratch
//By Zachary Canoot and Gray Simpson
//Some code reused from the Data Exploration project

//I'm using the namespace since much of the code will look messy with double-vector matrices.
using namespace std;

//MAXLEN makes it easier to read in data as we go, so as long as we know how big our data set it, everything is easy to adjust.
const int MAXLEN = 1500;
//Therefore, these will be our markers for test size.
const int TRAINMAX = 800;
const int TESTMAX = 1046;

//Prototypes
//These first three are from/adapted from the Data Exploration project written by Gray Simpson.
int ReadCSV(vector<int> &pclass, vector<int> &survived, vector<int> &sex, vector<int> &age);
double vector_sum(vector<double> vec);
double vector_mean(vector<double> vec);

void output_matrix(vector<vector<double>> matrix);
double covariance(vector<double> vecA, vector<double> vecB);
double calcAgeLH(double v, double meanv, double varv);
vector<double> calcRawProb(vector<vector<double>>LHpclass,vector<vector<double>>LHsex,vector<vector<double>>ageMean,vector<vector<double>>ageVar,double aprioriDead,double aprioriSurvived, double age, double pclass, double sex);


int main()
{
    cout << "Naive Bayes Machine Learning from Scratch in C++" << endl;

    //------------------------------------------------------Read in data.

    //Prepare to load the file.
    vector<int> pclass(MAXLEN);
    vector<int> survived(MAXLEN);
    vector<int> sex(MAXLEN);
    vector<int> age(MAXLEN);
    //Load file, return number of observations. Due to the nature of the assignment, the name is hardcoded in.
    //The vectors are passed by address so that they are automatically filled.
    int numOfObservations = ReadCSV(pclass,survived,sex,age);

    //If reading the file is successful,
    if (numOfObservations > 0){
        //We're good to go. Data is passed by address, so we have the correct variables now.

        //Let's see some (10) example observations.
        cout << "Head of data: " << endl;
        for(int i = 0; i < 10; i++){
             cout << "Survived: " << survived.at(i) << " Class: " << pclass.at(i) << " Age: " << age.at(i) << " Sex: " << sex.at(i) << endl;
        }
        //spacing
        cout << endl;

        //------------------------------------------------------Separating Data
        //The values for where we start and stop training are declared as constant global variables for ease.
        //We are semi-hardcoding the number of training observations for this, as we are hardcoding the elements it takes as is.


        //------------------------------------------------------Start The Clock
        //Now that we have read in and peeked at the data, let's start the clock for Naive Bayes Machine Learning.
        auto start = std::chrono::high_resolution_clock::now();


        //------------------------------------------------------Calculate Priors
        int numSurvived = 0, numDead = 0;
        //Count the numbers of survival in the training data.
        for(int i = 0; i < TRAINMAX; i++){
            if(survived.at(i) == 1){
                numSurvived++;
            }else{
                numDead++;
            }
        }
        double aprioriSurvived = (double) numSurvived/TRAINMAX;
        double aprioriDead = (double) numDead/TRAINMAX;

        //------------------------------------------------------Calculate Likelihoods (classes/qualitative)

        //A vector of survival
        //We keep this in a double vector due to the nature of our 'output_matrix()' function.
        vector<double> countSurvived = {(double)numDead,(double)numSurvived};
        vector<vector <double>> survivedMatrix = {countSurvived};


        //-----------------------------------SURVIVED|PCLASS
        //Gather a matrix for each pclass value and whether they survived or not.
        //Set up the format
        vector<vector<double>> LHpclass = { {0,0,0}, {0,0,0} };

        //Go through all observations, and sort the counts of survived or not and what class into a matrix.
        for(int i = 0; i < TRAINMAX; i++){
            LHpclass.at((int)survived.at(i)).at(((int) pclass.at(i))-1)++;
        }

        //divide them by their total counts
        for(int i=0;i<3;i++){
            (LHpclass.at(0)).at(i) = (LHpclass.at(0)).at(i) / numDead;
            (LHpclass.at(1)).at(i) = (LHpclass.at(1)).at(i) / numSurvived;
        }


        //-----------------------------------SURVIVED|SEX
        //Gather a matrix for each sex value and whether they survived or not.
        //Set up the format
        vector<vector<double>> LHsex = { {0,0}, {0,0} };

        //Go through all observations, and sort the counts of survived or not and what sex into a matrix.
        for(int i = 0; i < TRAINMAX; i++){
            LHsex.at((int)survived.at(i)).at(((int) sex.at(i)))++;
        }

        //divide them by their total counts
        for(int i=0;i<=1;i++){
            LHsex.at(0).at(i) = LHsex.at(0).at(i) / numDead;
            LHsex.at(1).at(i) = LHsex.at(1).at(i) / numSurvived;
        }


        //------------------------------------------------------Calculate Likelihoods (numbers/quantitative)
        vector<vector<double>> ageMean = {{0,0}};
        vector<vector<double>> ageVar = {{0,0}};
        //make two vectors: ages of people who survived, and ages of those who died
        vector<double> ageSurvived;
        vector<double> ageDied;
        for(int i=0; i<TRAINMAX; i++){
            if(survived.at(i) == 1){
                ageSurvived.push_back(age.at(i));
            }else{
                ageDied.push_back(age.at(i));
            }
        }

        (ageMean.at(0)).at(0) = vector_mean(ageDied);
        (ageMean.at(0)).at(1) = vector_mean(ageSurvived);
        //variance is between itself, so it can be calculated as covariance of itself
        (ageVar.at(0)).at(0) = covariance(ageDied,ageDied);
        (ageVar.at(0)).at(1) = covariance(ageSurvived,ageSurvived);


        //------------------------------------------------------Results
        vector<vector<double>> rawprobs;
        for(int i=TRAINMAX; i<TESTMAX; i++){
            vector<double> rawprob = {calcRawProb(LHpclass,LHsex,ageMean,ageVar,aprioriDead,aprioriSurvived,age.at(i),pclass.at(i),sex.at(i))};
            rawprobs.push_back(rawprob);
        }


        //------------------------------------------------------Wrapping Up
        auto finish = std::chrono::high_resolution_clock::now();
        auto nbayesTime = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        cout << "The Naive Bayes calculation took " << nbayesTime << " milliseconds.\n\nHere are the data values and metrics.\n" << endl;

        cout << "Survived: " << numSurvived << " prob " << aprioriSurvived << "  Dead: " << numDead << " " << aprioriDead << endl;

        cout << "Age Died Mean: " << ageMean.at(0).at(0) << "       Age Survived Mean: " << ageMean.at(0).at(1) << endl;
        cout << "Age Died Variance: " << ageVar.at(0).at(0) << "   Age Survived Variance " << ageVar.at(0).at(1) << endl;


        cout << "\nPclass Likelihoods" << endl;
        output_matrix(LHpclass);

        cout << "\nSex Likelihoods" << endl;
        output_matrix(LHsex);


        //------------------------------------------------------Accuracy, Sensitivity, Specificity
        //these are doubles so that we can put them into a matrix and easily output it
        double truepos = 0, trueneg = 0, falsepos = 0, falseneg = 0;
        //See how the data for each test lines up with the actual values.
        for(int i=0; i< rawprobs.size(); i++){
            if((rawprobs.at(i)).at(1) >= .5){
                if(survived.at(i+TRAINMAX)==1){
                    truepos++;
                }else{
                    falsepos++;
                }
            }else{
                if(survived.at(i+TRAINMAX)==0){
                    trueneg++;
                }else{
                    falseneg++;
                }
            }
        }
        //Make a confusion matrix!
        cout << "\nConfusion Matrix" << endl;
        //built in the order of survival, if they died and were predicted to die, that's in index [0,0]
        vector<vector<double>> confusion = {{trueneg,falseneg}, {falsepos,truepos}};
        output_matrix(confusion);

        cout << "\nAccuracy: " << (double)(truepos + trueneg) / (double)rawprobs.size() << endl;
        cout << "Sensitivity: " << (double)truepos / (double)(truepos + falseneg) << endl;
        cout << "Specificity: " << (double)trueneg / (double)(trueneg + falsepos) << endl;


    }else{
        //Reading file failed; terminate program.
        return 0;
    }

}




//Read in the titanic file.
int ReadCSV(vector<int> &pclass, vector<int> &survived, vector<int> &sex, vector<int> &age){

//Prepare to read the file.
    ifstream in;

    cout << "Attempting to open file 'titanic_project.csv'," << endl;
    in.open("titanic_project.csv");
    if(in.is_open()){
        cout << "File opened successfully." << endl;
    }else{
        cout << "File could not be opened. Terminating program." << endl;
        return 0;
    }

    //Now that we have the file, we can actually read it.
    //We will need more variables.
    string line;
    //We will name the vectors what they're named in the file.
    string num_in, pclass_in, survived_in, sex_in, age_in;
    int numOfObservations = 0;

    cout << "Reading heading: ";
    getline(in,line);
    cout << line << endl;
    //It shows us the column titles, separated by a comma
    //From here, we can move on to actually reading in the data.

    //While there is data
    while(in.peek()!= EOF){
        getline(in,num_in,',');
        getline(in,pclass_in,',');
        getline(in,survived_in,',');
        getline(in,sex_in,',');
        getline(in,age_in);

        //We ignore the num column since it is not relevant to what we need, and is just the number passenger it is. We still read it in (num_in) because it takes up space.

        //put the values into our vectors
        pclass.at(numOfObservations) = stoi(pclass_in);
        survived.at(numOfObservations) = stoi(survived_in);
        sex.at(numOfObservations) = stoi(sex_in);
        age.at(numOfObservations) = stoi(age_in);

        numOfObservations++;
    }

    //make sure the vectors aren't taking up too much space
    pclass.resize(numOfObservations);
    survived.resize(numOfObservations);
    sex.resize(numOfObservations);
    age.resize(numOfObservations);

    cout << "Length is now " << pclass.size() << endl;
    cout << "Now closing file 'titanic_project.csv'" << endl;

    //Always important to clean up.
    in.close();

    return numOfObservations;

}


//Sum up all elements of a vector.
double vector_sum(vector<double> vec){

    double sum = 0;

    vector<double>::iterator it;
    //Go through the vector and add each value to a summation variable.
    for(it = vec.begin(); it < vec.end(); it++ ){
        sum+= *it;
    }

    return sum;
}

//Average all elements of a vector.
double vector_mean(vector<double> vec){

    //Divide the sum by the size of the vector.
    return vector_sum(vec) / vec.size();
}


//Output a matrix (made in vectors)
void output_matrix(vector<vector<double>> matrix){
    //Spacing
    cout << "     ";
    for(int b = 0; b < matrix.at(0).size(); b++){
            cout << b << "     ";
    }
    cout << endl;

    //Data, columns
    for(int a = 0; a < matrix.size(); a++){
        cout << a << "  ";
        //rows
        for(int b = 0; b < matrix.at(0).size(); b++){
            cout << matrix.at(a).at(b) << "  ";

        }
        cout << endl;
    }

}


//Calculate the covariance between two vectors.
double covariance(vector<double> vecA, vector<double> vecB){

    //we'll use this to slowly sum up the values we need to
    double covar = 0;

    //make sure these are compatible vectors
    if(vecA.size() == vecB.size()){

        double meanA = vector_mean(vecA);
        double meanB = vector_mean(vecB);

        vector<double>::iterator itA;
        vector<double>::iterator itB = vecB.begin();
        int i = 0;
        //Iterate through the vector and add each value to a summation variable.
        for(itA = vecA.begin(); itA < vecA.end(); i++){
            covar+= (*itA - meanA) * (*itB - meanB);

            itA++;
            itB++;
        }
        //Now we divide by the size minus one. Both vectors have the same size, so we are safe to use either.
        covar = covar/(vecA.size()-1);
    }else{
        cout << "Vector sizes are not compatible. Covariance cannot be calculated.";
    }

    return covar;
}

//Individual age likelihood
double calcAgeLH(double v, double meanv, double varv){
    double result = 1 /  (sqrt(2 * 3.14159 * varv)* exp(pow(-(v-meanv),2)/(2 * varv)));
    return result;
}

//Naive Bayes itself
vector<double> calcRawProb(vector<vector<double>>LHpclass,vector<vector<double>>LHsex,vector<vector<double>>ageMean,vector<vector<double>>ageVar,double aprioriDead,double aprioriSurvived, double age, double pclass, double sex){
        //Sur for Survived, Per for Perished
        double numSur = (LHpclass.at(1)).at((int)pclass-1) * (LHsex.at(1)).at((int)sex) * aprioriSurvived * calcAgeLH(age,(ageMean.at(0)).at(1), (ageVar.at(0)).at(1));
        double numPer = (LHpclass.at(0)).at((int)pclass-1) * (LHsex.at(0)).at((int)sex) * aprioriDead * calcAgeLH(age,(ageMean.at(0)).at(0), (ageVar.at(0)).at(0));

        double denominator = numSur + numPer;

        return vector<double>{numPer/denominator,numSur/denominator};

}

