#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"


#include <unordered_set>

using namespace std;
using namespace hnswlib;


void check_groundtruth()
{
    int subset_size_milllions = 1;
    size_t qsize = 1;
    char path_gt[1024];
    sprintf(path_gt, "bigann/gnd/idx_%dM.ivecs", subset_size_milllions);
    cout << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * 1000];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + 1000 * i), t * 4);
        if (t != 1000) {
            cout << "err";
            return;
        }
    }

/* check if ground truth is making sense
    ifstream myfile;
    myfile.open("getgt.txt");
    if (!myfile) {
    cerr << "Unable to open file getgt.txt";
    exit(1);   // call system to stop
    }

    unsigned int *Bf = new unsigned int[qsize*1000];

    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < 1000; j++) {
          myfile>>Bf[i*1000+j];
        }
    }



    for (int i = 0; i < qsize; i++) {

      unordered_set<labeltype> gt,bf;
        for (int j = 0; j < 1000; j++) {
             //cout<<"ground truth is "<<massQA[1000 * i + j]<<" "<<"BruteForce search result is "<<BF[i*1000+1000-1-j]<<endl;
             gt.insert(massQA[1000 * i + j]);
             bf.insert(Bf[i*1000+1000-1-j]);
        }
        if(gt != bf)
        {    cout<<"Query "<<i<<" error!"<<endl;
        //if(i==9748)
        //{
          for (int j = 0; j < 1000; j++) {
                if(massQA[1000 * i + j] != Bf[i*1000+1000-1-j])
                  cout<<"ground truth is "<<massQA[1000 * i + j]<<" "<<"BruteForce search result is "<<Bf[i*1000+1000-1-j]<<endl;
             }
        }
    }
*/
    ifstream myfile2;
    myfile2.open("readresult.txt");
    if (!myfile2) {
    cerr << "Unable to open file getgt.txt";
    exit(1);   // call system to stop
    }

    unsigned int *result = new unsigned int[qsize*100];

    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < 100; j++) {
          myfile2>>result[i*100+j];
        }
    }

    for (int i = 0; i < qsize; i++) {
        cout<<"Query "<<i<<endl;
        unordered_set<labeltype> gt;
        for (int j = 0; j < 100; j++) {
            gt.insert(massQA[1000 * i + j]);
            //cout<<"ground truth is "<<massQA[1000 * i + j]<<", result is "<<result[i*100+100-1-j]<<endl;
        }
        int correct=0;
        for (int j = 0; j < 100; j++) {
            if(gt.find(result[i*100+j])!=gt.end())
            correct++;
        }
        cout<<correct<<endl;
      }

}



int main() {
    //sift_test1B();
    check_groundtruth();
    return 0;
};
