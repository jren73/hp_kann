#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include "gen_zipf.h"
#include <random>
#include <stdlib.h>
#include <unordered_set>
#include <list>

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:

        HierarchicalNSW(SpaceInterface<dist_t> *s) {

        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 48, size_t ef_construction = 200, size_t random_seed = 100) :
                link_list_locks_(max_elements), element_levels_(max_elements) {
            max_elements_ = max_elements;  //the # opf total elements in dataset
            has_deletions_=false;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_ * 2;
            maxM0_ = M_ * 2;
            //maxM0_ = M_;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;

            level_generator_.seed(random_seed);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            //jie: all data in level 0 stores as vector itself
            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);



            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            //jie: allocate a pointer pointing to link
            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            size_links_per_element_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_per_element_level1 = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW() {

            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        size_t max_elements_;
        size_t cur_element_count;   //global variable
        size_t size_data_per_element_;
        size_t size_links_per_element_;
        //jie;
        size_t size_links_per_element_level1;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        //jie:
        size_t ef_l1_;
        size_t ef_reuse_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;
        tableint enterpoint_node_;  //enterpoint is global variable


        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;


        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;

        size_t data_size_;

        bool has_deletions_;


        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;


        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const { //set the value
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const { //return the value
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const { //return the address
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double temp=distribution(level_generator_);
            double r = -log(temp) * reverse_size;


            //jie: generate graph with level 1 promotion rate = PR
            double pr = (double)1/(double)4;
            if(pr>0 && (int)r==0)
            {
                double ttemp=distribution(level_generator_);
                if(ttemp<=pr)
                  r=1;
            }
            //int n = 50*1000000;
            //std::cout<<"before zipf"<<std::endl;
            //double result = (double)zipf(1.0, n)/(double)n;
            //std::cout<<result<<std::endl;
            //double r = -log(result) * reverse_size;
            //std::cout<<r<<std::endl;
            return (int) r;
        }

        //jie: ep_id is the enter point id, data_point is the target, layer specify which layer do the search
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;  //jie: lowerbound is for the farest distance between top_candidates and target
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);   //jie:why that? priority queue sorting
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();  //nearest neighbor from candidates set
                if ((-curr_el_pair.first) > lowerBound) {   //jie: stop searching condition: nearest distance from target to candidates is larger than target to top_candidates
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;  //current vector id

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);  //return data or id???
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
//                  data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        template <bool has_deletions>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const {
            int hop=0;
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
                hop++;
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty()) {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound) {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,///////////
                                         _MM_HINT_T0);////////////////////////
#endif

                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                            {
                                  top_candidates.emplace(dist, candidate_id);
                                  hop++;
                            }

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            //std::cout<<" "<<hop<<" ";
            return top_candidates;
        }

        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }
            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;
                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);;
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }


            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {

                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }


        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_level1(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_level1);
        };

        void mutuallyConnectNewElement(void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates,
                                       int level) {

            //size_t Mcurmax = level ? maxM_ : maxM0_;
            //jie: level 1 connection=2M
            size_t Mcurmax = level>1 ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1);


                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx])
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];

                }
            }
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);


                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *) (ll_other + 1);
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                             dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }
                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }

            }
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }

        void setEf_l1(size_t ef) {
            ef_l1_ = ef;
        }

        void setEf_reuse(size_t ef) {
            ef_reuse_ = ef;
        }


        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (size_t level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    int *data;
                    data = (int *) get_linklist(currObj,level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }


            std::priority_queue<std::pair<dist_t, tableint  >> top_candidates;
            if (has_deletions_) {
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<true>(currObj, query_data,
                                                                                                             ef_);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<false>(currObj, query_data,
                                                                                                              ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements){
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);



            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);


            // Reallocate base layer
            char * data_level0_memory_new = (char *) malloc(new_max_elements * size_data_per_element_);
            memcpy(data_level0_memory_new, data_level0_memory_,cur_element_count * size_data_per_element_);
            free(data_level0_memory_);
            data_level0_memory_=data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) malloc(sizeof(void *) * new_max_elements);
            memcpy(linkLists_new, linkLists_,cur_element_count * sizeof(void *));
            free(linkLists_);
            linkLists_=linkLists_new;

            max_elements_=new_max_elements;

        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);
            //jie:count element #
            std::unordered_map<size_t, size_t> m;
            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                {
                      output.write(linkLists_[i], linkListSize);
                      m[element_levels_[i]]++;
                }
            }
            for(auto it:m)
            {
              std::cout<<it.first<<" "<<it.second<<std::endl;
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {


            std::ifstream input(location, std::ios::binary);

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements=max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            /// Legacy, check that everything is ok

            bool old_index=false;

            auto pos=input.tellg();
            input.seekg(cur_element_count * size_data_per_element_,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    old_index = true;
                    break;
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // check if file is ok, if not this is either corrupted or old index
            if(input.tellg()!=total_filesize)
                old_index = true;

            if (old_index) {
                std::cerr << "Warning: loading of old indexes will be deprecated before 2019.\n"
                          << "Please resave the index in the new format.\n";
            }
            input.clear();
            input.seekg(pos,input.beg);


            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            if(old_index)
                input.seekg(((max_elements_-cur_element_count) * size_data_per_element_), input.cur);


            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);


            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);


            visited_list_pool_ = new VisitedListPool(1, max_elements);


            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;

            std::unordered_map<size_t, size_t> m;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    //std::cout<<i<<" "<<element_levels_[i]<<std::endl;
                    m[element_levels_[i]]++;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    input.read(linkLists_[i], linkListSize);
                }
            }
            //print # of layers
            for(auto it:m)
            {
              std::cout<<it.first<<" "<<it.second<<std::endl;
            }
            input.close();
            Count_in_degree();
            return;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label)
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char* data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
//        static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph. //jie:  does NOT really change the current graph!!!
         * @param label
         */
        void markDelete(labeltype label)
        {
            has_deletions_=true;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            markDeletedInternal(search->second);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur &= ~DELETE_MARK;
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {
            unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId))+2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        void addPoint(void *data_point, labeltype label) {
            addPoint(data_point, label,-1);
        }

        tableint addPoint(void *data_point, labeltype label, int level) {
            //data_point is a pointer!!!
            tableint cur_c = 0;
            { //cpp scope for local variable
                //synchronize point adding process
                std::unique_lock <std::mutex> lock(cur_element_count_guard_);
                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                //cur_c is unique!
                cur_c = cur_element_count;
                cur_element_count++;

                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    std::unique_lock <std::mutex> lock_el(link_list_locks_[search->second]);
                    markDeletedInternal(search->second); //Jie: what's this for??? This avoids some elements might be added twice
                }
                label_lookup_[label] = cur_c;
            }

            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)   //if the point is not the first time add to hnsw graph, no need to generate level
                curlevel = level;

            element_levels_[cur_c] = curlevel;  //record element level


            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_); //void* memset( void* dest, int ch, std::size_t count );

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));  //cur_c is internalID
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);


            if (curlevel) { //upper level allocate memory
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);  //jie: why +1
                //every element all links are allocated together
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

      /*      //jie: allocate more space fpr level 1 to ensure 2M connections
            if (curlevel>1) { //upper level allocate memory
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);  //jie: why +1
                //every element all links are allocated together
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            else if (curlevel==1) { //upper level allocate memory
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_* 2 * curlevel + 1);  //jie: why +1
                //every element all links are allocated together
                memset(linkLists_[cur_c], 0, size_links_per_element_* 2 * curlevel + 1);
            }
      */

            if ((signed)currObj != -1) { //currObj is enter point

                if (curlevel < maxlevelcopy) {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    //the distance between target(inserted point) and enter point
                    for (int level = maxlevelcopy; level > curlevel; level--) {


                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);

                            data = get_linklist(currObj,level);

                            //data = get_linklist_level1(currObj, level);

                            int size = getListCount(data); //jie:???

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    mutuallyConnectNewElement(data_point, cur_c, top_candidates, level);

                    currObj = top_candidates.top().second;
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        //jie: count in-degree
        void Count_in_degree()
        {
            //size_t mycounter[max_elements_];
            size_t *mycounter = (size_t*)malloc(sizeof(size_t) * max_elements_);
            memset(mycounter, 0, sizeof(mycounter));
            unsigned int *data;
            for(size_t j=0; j<max_elements_; j++)
            {
              data = get_linklist0(j);
              int size=getListCount(data);
              tableint *datal = (tableint *) (data + 1);
              //if(size!=0)
              //    cout<<j<<endl;
              for (int i = 0; i < size; i++)
              {
                mycounter[datal[i]]++;
              }
            }
            size_t largerthan=0;
            for(size_t i=0; i<max_elements_; i++)
            {
               //cout<<i<<" "<<mycounter[i]<<endl;

               if(mycounter[i]>=60)
               {
                  largerthan++;
               }
            }
            cout<<"large than 100 nodes "<<largerthan<<endl;
        }

/*
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchFirstLayer(tableint ep_id, const void *data_point, size_t ef) const {
            int layer=1;
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;  //jie: lowerbound is for the farest distance between top_candidates and target
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);   //jie:why that? priority queue sorting
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();  //nearest neighbor from candidates set
                if ((-curr_el_pair.first) > lowerBound) {   //jie: stop searching condition: nearest distance from target to candidates is larger than target to top_candidates
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;  //current vector id

                //std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data = (int*)get_linklist(curNodeNum, 1);
//                  data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }
*/

        //jie: for reuse
        //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_g;
        //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set_g;
///*
        //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        void searchFirstLayer(tableint ep_id, const void *data_point, size_t ef, std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates_g,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &candidate_set_g) const{
            int layer=1;
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            //cout<<"level 1 visited list is "<<*visited_array<<" , tag is "<<visited_array_tag<<endl;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;  //jie: lowerbound is for the farest distance between top_candidates and target
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                top_candidates_g.emplace(-dist, ep_id);
                lowerBound = dist;
                candidate_set_g.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set_g.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;
            //cout<<"top candidates set is "<<candidate_set_g.size()<<endl;

            while (!candidate_set_g.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidate_set_g.top();  //nearest neighbor from candidates set
                if ((-curr_el_pair.first) > lowerBound) {   //jie: stop searching condition: nearest distance from target to candidates is larger than target to top_candidates
                    break;
                }
                candidate_set_g.pop();

                tableint curNodeNum = curr_el_pair.second;  //current vector id

                //std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data = (int*)get_linklist(curNodeNum, 1);
//                  data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    if (top_candidates.size() < ef|| lowerBound > dist1) {
                        candidate_set_g.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidate_set_g.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                        {
                              top_candidates.emplace(dist1, candidate_id);
                              top_candidates_g.emplace(-dist1, candidate_id);
                        }

                        if (top_candidates.size() > ef)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
            //top_candidates_g(top_candidates);
            //candidate_set_g(candidateSet);
            //return top_candidates;
        }

//*/


        template <bool has_deletions>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST_reuse(tableint ep_id, const void *data_point, size_t ef, std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates, std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &candidate_set) const{
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV-1;
            //cout<<"level 0 visited list is "<<*visited_array<<" , tag is "<<visited_array_tag<<endl;

            //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
                //dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = top_candidates.top().first;
                //lowerBound=dist;
                //top_candidates.emplace(dist, ep_id);
                //candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            int count=0;

            while (!candidate_set.empty()) {
                if(count>=250)
                  break;

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                /*if ((-current_node_pair.first) > lowerBound) {
                    break;
                }*/
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
        //                bool cur_node_deleted = isMarkedDeleted(current_node_id);

        #ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
        #endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                          if (candidate_id == 0) continue;
       #ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);////////////
        #endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        count++;
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {

                            //cout<<"update candidate set"<<endl;
                            candidate_set.emplace(-dist, candidate_id);
        #ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,///////////
                                         _MM_HINT_T0);////////////////////////
        #endif

                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                            {
                                  top_candidates.emplace(dist, candidate_id);
                            }

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            //std::cout<<" "<<hop<<" ";
            return top_candidates;
        }

        /* SerachKNN*/

        std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *query_data, size_t k) const {
            //for distance computation check

            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
            int hop=0;
            //std::cout<<maxlevel_<<std::endl;
            //for (int level = maxlevel_; level > 0; level--) {
            //jie: change searching ef on level 1
            for (int level = maxlevel_; level > 1; level--) {
                bool changed = true;
                //if(level==1)
                //    temp_currObj = currObj;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
            //jie: add for level 1
            //cout<<*query_data<<endl;
            //unsigned char *tmp = new unsigned char[128];
            //memcpy(tmp, (unsigned char *)query_data, 128);
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_g;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set_g;
            //clean_dist_comp_time();
            searchFirstLayer(currObj, query_data, ef_l1_, top_candidates_g, candidate_set_g);
            //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_l1 = searchFirstLayer(
            //        currObj, query_data, ef_l1_, top_candidates_g, candidate_set_g);
            //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_l1 = searchFirstLayer(
            //        temp_currObj, query_data, ef_l1_);

            //cout<<top_candidates_l1.size()<<" "<<top_candidates_g.size()<<" "<<candidate_set_g.size()<<endl;


            /*while(!top_candidates_l1.empty())
            {
                tt_currObj = top_candidates_l1.top().second;
                top_candidates_l1.pop();
            }*/


            //top_candidates_temp.swap(top_candidates_l1);
            //cout<<"top candidates in L1"<<endl;

            /*while(!top_candidates_l1.empty())
            {
                current_node_pair = top_candidates_l1.top();
                //cout<<top_candidates_l1.top().first<<" "<<top_candidates_l1.top().second<<endl;
                //candidate_set_g.push(top_candidates_l1.top());
                top_candidates_l1.pop();
            }
            top_candidates_l1.push(current_node_pair);
            */
/*
            while (top_candidates_g.size() > ef_reuse_) {
                top_candidates_g.pop();
            }*/

            tableint temp_currObj = top_candidates_g.top().second;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_try;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_ttry;
            top_try.emplace(std::pair<dist_t, labeltype>(-top_candidates_g.top().first,top_candidates_g.top().second));
            /*int i=0;
            while(!top_candidates_g.empty() && i<ef_reuse_)
            {
              top_ttry.emplace(top_candidates_g.top());
              top_candidates_g.pop();
              i++;
            }*/
            /*if(temp_currObj != current_node_pair.second)
            {    cout<<"error!"<<endl;
                  cout<<current_node_pair.first<<" "<<top_candidates_g.top().first<<endl;
                  dist_t curdist1 = fstdistfunc_(query_data, getDataByInternalId(top_candidates_g.top().second), dist_func_param_);
                  cout<<curdist1<<endl;

            }*/

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

            if (has_deletions_) { //seems never goes here?!?
                //std::cout<<"has deletion"<<std::endl;
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates1=searchBaseLayerST_reuse<true>(
                        temp_currObj, query_data, std::max(ef_, k), top_try, top_candidates_g);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates1=searchBaseLayerST_reuse<false>(
                        temp_currObj, query_data, std::max(ef_, k), top_try, top_candidates_g);
                top_candidates.swap(top_candidates1);
            }
            //top_candidates.swap(top_candidates_g);
            std::priority_queue<std::pair<dist_t, labeltype >> results;
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                results.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            //jie: for distance compute check
            //print_dist_comp_time();
            return results;
        };


    };

}
