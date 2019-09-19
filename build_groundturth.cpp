#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"


#include <unordered_set>

using namespace std;
using namespace hnswlib;

//check the result with brute force search
void get_gt(float *mass, float *massQ, size_t vecsize, size_t qsize, L2Space &l2space, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    BruteforceSearch<float> bs(&l2space, vecsize);
    for (int i = 0; i < vecsize; i++) {
        bs.addPoint((void *) (mass + vecdim * i), (size_t) i);
    }
    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    //answers.swap(vector<std::priority_queue< std::pair< float, labeltype >>>(qsize));
    for (int i = 0; i < qsize; i++) {
        std::priority_queue<std::pair<float, labeltype >> gt = bs.searchKnn(massQ + vecdim * i, 10);
        answers[i] = gt;
    }
}

void sift_groundtruth()
{
  size_t vecsize = 1000000;
  size_t qsize = 10000;
  //size_t qsize = 1000;
  //size_t vecdim = 4;
  size_t vecdim = 128;

  float *mass = new float[vecsize * vecdim];
  ifstream input("sift/sift_base.fvecs", ios::binary);
  //ifstream input("../../sift100k.bin", ios::binary);
  //ifstream input("../../1M_d=4.bin", ios::binary);
  input.read((char *) mass, vecsize * vecdim * sizeof(float));
  input.close();

  float *massQ = new float[qsize * vecdim];
  //ifstream inputQ("../siftQ100k.bin", ios::binary);
  ifstream inputQ("sift/sift_query.fvecs", ios::binary);
  //ifstream inputQ("../../siftQ100k.bin", ios::binary);
  //ifstream inputQ("../../1M_d=4q.bin", ios::binary);
  inputQ.read((char *) massQ, qsize * vecdim * sizeof(float));
  inputQ.close();

  L2Space l2space(vecdim);

  vector<std::priority_queue<std::pair<float, labeltype >>> answers;
  size_t k = 100;
  cout << "Loading gt\n";
  get_gt(mass, massQ, vecsize, qsize, l2space, vecdim, answers,k);

  std::ofstream ofile("sift/groundtruth.bin", std::ios::binary);
  for(auto& it:answers)
  {
    ofile.write((char*) &it, sizeof(it));
  }

  cout<<"finished"<<endl;
}



int main() {
    //sift_test1B();
    sift_groundtruth();
    return 0;
};
