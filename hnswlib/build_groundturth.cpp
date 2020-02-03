#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"


#include <unordered_set>

using namespace std;
using namespace hnswlib;

//check the result with brute force search
void get_gt(unsigned char *mass, unsigned char *massQ, size_t vecsize, size_t qsize, L2SpaceI &l2space, size_t vecdim,
            vector<std::priority_queue<std::pair<int, labeltype >>> &answers, size_t k) {
    BruteforceSearch<int> bs(&l2space, vecsize);
    for (int i = 0; i < vecsize; i++) {
        bs.addPoint((void *) (mass + vecdim * i), (size_t) i);
    }
    (vector<std::priority_queue<std::pair<int, labeltype >>>(qsize)).swap(answers);
    //answers.swap(vector<std::priority_queue< std::pair< float, labeltype >>>(qsize));
    size_t total=0;
    //std::ofstream ofile("groundtruth.bin", std::ios::binary);
    ofstream myfile;
    myfile.open ("getgt.txt");

    //unordered_set<labeltype> g;
    for (int i = 0; i < qsize; i++) {
        cout<<"Searching "<<i<<endl;
        std::priority_queue<std::pair<int, labeltype >> gt = bs.searchKnn(massQ + vecdim * i, 1000);
        answers[i] = gt;
        total += gt.size();

        while (gt.size()) {
            //g.insert(gt.top().second);
            myfile<<gt.top().second<<" ";
            cout<<gt.top().second<<" ";
            gt.pop();
        }
        cout<<endl;
        myfile<<endl;
        //ofile.write((char*) &g, sizeof(g));
    }
    myfile.close();
    //ofile.close();
}

void sift_groundtruth()
{
  int subset_size_milllions = 1;
  size_t vecsize = subset_size_milllions * 1000000;
  size_t qsize = 10000;
  //size_t qsize = 1000;
  //size_t vecdim = 4;
  size_t vecdim = 128;

  //float *mass = new float[vecsize * vecdim];
  unsigned char *mass = new unsigned char[vecsize * vecdim];
  unsigned char *massb = new unsigned char[vecdim];
  //ifstream input("sift/sift_base.fvecs", ios::binary);
  ifstream input("bigann/bigann_base.bvecs", ios::binary);
  //ifstream input("../../sift100k.bin", ios::binary);
  //ifstream input("../../1M_d=4.bin", ios::binary);
  //input.read((char *) mass, vecsize * vecdim * sizeof(float));
  //input.close();
  int in=0;
  for (int i = 0; i < vecsize; i++) {
  input.read((char *) &in, 4);  //in for vector size
  if (in != 128) {
      cout << "file error";
      exit(1);
  }
  input.read((char *) massb, in);

  for (int j = 0; j < vecdim; j++) {
      mass[i*vecdim + j] = massb[j] * (1.0f);
  }
}


  unsigned char *massQ = new unsigned char[qsize * vecdim];
  //ifstream inputQ("../siftQ100k.bin", ios::binary);
  ifstream inputQ("bigann/bigann_query.bvecs", ios::binary);
  //ifstream inputQ("../../siftQ100k.bin", ios::binary);
  //ifstream inputQ("../../1M_d=4q.bin", ios::binary);
  //inputQ.read((char *) massQ, qsize * vecdim * sizeof(float));
  //inputQ.close();
  cout << "Loading queries:\n";
  //unsigned char *massQ = new unsigned char[qsize * vecdim];
  //ifstream inputQ(path_q, ios::binary);

  for (int i = 0; i < qsize; i++) {
      int in = 0;
      inputQ.read((char *) &in, 4);
      if (in != 128) {
          cout << "file error";
          exit(1);
      }
      inputQ.read((char *) massb, in);
      for (int j = 0; j < vecdim; j++) {
          massQ[i * vecdim + j] = massb[j];
      }

  }
  inputQ.close();

  L2SpaceI l2space(vecdim);

  vector<std::priority_queue<std::pair<int, labeltype >>> answers;
  size_t k = 1000;
  cout << "Loading gt 1m\n";
  get_gt(mass, massQ, vecsize, qsize, l2space, vecdim, answers,k);

  //std::ofstream ofile("groundtruth.bin", std::ios::binary);
  /*for(auto& it:answers)
  {
    ofile.write((char*) &it, sizeof(it));
  }*/

  cout<<"finished"<<endl;
}



int main() {
    //sift_test1B();
    sift_groundtruth();
    return 0;
};
