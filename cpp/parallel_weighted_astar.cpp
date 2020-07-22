#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <set>
#include <vector>
#include <map>
#include <fstream>
#include <ctime>
#include <assert.h>
#include <csignal>
#include <cstdlib>
#include <stdint.h>
#include <sys/time.h>
#include <queue>          // std::priority_queue
#include <unordered_map>
#include <unordered_set>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <thread>         // std::thread
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/un.h>
#include <boost/functional/hash.hpp>


#include "environments.h"

void error(const char *msg) {
	perror(msg);
	exit(0);
}

void copyArr(int fromArr[], int toArr[], int numElems) {
	for (int i=0; i<numElems; i++) {
		toArr[i] = fromArr[i];
	}
}

void printArr(uint8_t arr[], int numElems) {
	for (int i=0; i<numElems; i++) {
		printf("%i ",arr[i]);
	}
	printf("\n");
}

void printArr(std::vector<uint8_t> arr) {
	for (unsigned int i=0; i<arr.size(); i++) {
		printf("%i ",arr[i]);
	}
	printf("\n");
}

bool fileExists(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

std::string stateToStr(const std::vector<uint8_t> state) {
	std::ostringstream vts;
	std::copy(state.begin(), state.end()-1,std::ostream_iterator<int>(vts, ", "));
	vts << state.back(); 
	return(vts.str());
}

double getTimeElapsed(std::chrono::high_resolution_clock::time_point t1, std::chrono::high_resolution_clock::time_point t2) {
	double timeElapsed = ((double) std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count())/1000000000.0;
	return(timeElapsed);
}

/*** Search Algorithm ***/
//Track environment information
struct Node {
		const Environment *env;
		int depth;
		int parentMove;
		float cost;
		float heuristic;
		Node *parent;

		bool operator==(const Node &other) const {
			return(env->getState() == other.env->getState());
		}

		~Node() {
			delete env;
		}
};

struct NodePointerEq {
	bool operator () ( Node const *lhs, Node const *rhs ) const {
		return lhs->env->getState() == rhs->env->getState();
	}
};


struct Hash {
	size_t operator() (const Node *node) const {
		std::vector<uint8_t> state = node->env->getState();
		size_t hashVal = boost::hash_range(state.begin(), state.end());

		return(hashVal);
	}
};

// Node comparator
class compareNodeCost {
	public:
	bool operator() (const Node *node1, const Node *node2) {
		return(node1->cost > node2->cost);
	}
};

void writeFile(int sockfd, std::vector<Node*> &children) {
	// Get states
	std::vector<uint8_t> states;
	for (unsigned int i=0; i<children.size(); i++) {
		std::vector<uint8_t> state = children[i]->env->getState();
		for (unsigned int j=0; j<state.size(); j++) {
			states.push_back(state[j]);
		}
	}

	// Write states
	unsigned long long dataSendSize = sizeof(uint8_t)*states.size();
	write(sockfd,&dataSendSize,8);

	write(sockfd,&states[0],dataSendSize);
}

void parallelWeightedAStar(const Environment *env, float depthPenalty, int numParallel, std::string socketName) {
	/* Initialize Heuristics */
	printf("INITIALIZING QUEUES\n");
	std::priority_queue<Node*,std::vector<Node*>,compareNodeCost> open;
	std::unordered_set<Node*,Hash,NodePointerEq> closed;

	int sockfd, servlen;
	struct sockaddr_un serv_addr;

    // Initalize socket
    bzero((char *)&serv_addr,sizeof(serv_addr));
    serv_addr.sun_family = AF_UNIX;
    strcpy(serv_addr.sun_path, socketName.c_str());
    servlen = (int) strlen(serv_addr.sun_path) + (int) sizeof(serv_addr.sun_family);

    if ((sockfd = socket(AF_UNIX, SOCK_STREAM,0)) < 0)
        error("Creating socket");
    if (connect(sockfd, (struct sockaddr *) &serv_addr, servlen) < 0)
        error("Connecting");

	std::chrono::high_resolution_clock::time_point searchStartTime = std::chrono::high_resolution_clock::now();

	open.push(new Node{env,0,-1,0,0,NULL}); //Push root node to open
	printf("GET START\n");
	closed.insert(new Node{env,0,-1,0,0,NULL}); //Add root node to seen
	

	int searchItr = 1;
	long numNodesGenerated = 1;
	bool isSolved = false;
	Node *solvedNode = NULL;
	while (isSolved == false) {
		std::chrono::high_resolution_clock::time_point startTime, t1;
		double itrTime, remOpenTime, expandingTime, dataWriteTime, checkClosedTime, heuristicTime, costTime, addToQueueTime;
		int maxDepth = 0, minDepth = 0;
		float maxValue = 0, minValue = 0, minCost = 0, maxCost = 0;

		startTime = std::chrono::high_resolution_clock::now();

		// Remove from open
		int openSize = (int) open.size();
		int numPop = std::min(openSize,numParallel);
		std::vector<Node*> popped;

		//printf("REMOVING FROM OPEN\n");
		t1 = std::chrono::high_resolution_clock::now();
		bool goal_node_found_prev = solvedNode != NULL;
		for (int i=0; i<numPop; i++) {
			Node *node = open.top();
			popped.push_back(node);
			open.pop();

			bool isSolved_itr = node->env->isSolved();
			if (isSolved_itr) {
			    if (numParallel == 1) {
				    solvedNode = node;
			        isSolved = true;
			    } else {
                    if (solvedNode == NULL) {
                        solvedNode = node;
                    } else if (solvedNode->cost > node->cost) {
                        solvedNode = node;
                    }
			    }
			    break;
			}
		}
		if (goal_node_found_prev && (popped[0]->cost >= solvedNode->cost)) {
		    // printf("%f, %f, %f\n", popped[0]->cost, solvedNode->cost, popped[popped.size()-1]->cost);
		    isSolved = true;
		}
		remOpenTime = getTimeElapsed(t1,std::chrono::high_resolution_clock::now());

		// Expand
		//printf("EXPANDING\n");
		t1 = std::chrono::high_resolution_clock::now();
		std::vector<int> depths(popped.size());
		std::vector<Node*> children(popped.size()*env->getNumActions());

		#pragma omp parallel for
		for (unsigned int i=0; i<popped.size(); i++) {
			std::vector<Environment*> children_env = popped[i]->env->getNextStates();
			int depth = popped[i]->depth + 1;
			depths[i] = depth;

			for (unsigned int j=0; j<children_env.size(); j++) {
			    float heuristic_lb = std::max(popped[i]->heuristic - 1, (float) 0.0);  //TODO replace with transition cost
			    float cost = heuristic_lb*(!children_env[j]->isSolved()) + depthPenalty*((float) depth);
				Node *node = new Node{children_env[j],depth,(int) j,cost,heuristic_lb,popped[i]};

				children[i*env->getNumActions() + j] = node;
			}
		}
		minDepth = *std::min_element(depths.begin(),depths.end());
		maxDepth = *std::max_element(depths.begin(),depths.end());
		expandingTime = getTimeElapsed(t1,std::chrono::high_resolution_clock::now());

		// Write children to file
		t1 = std::chrono::high_resolution_clock::now();
		writeFile(sockfd,children);

		//std::thread writeThread (writeFile,sockfd,children);
		//writeThread.join();
		dataWriteTime = getTimeElapsed(t1,std::chrono::high_resolution_clock::now());

		//Check if in closed
		t1 = std::chrono::high_resolution_clock::now();
		std::vector<Node*> nodesToAdd;
		std::vector<int> nodesToAdd_idx;
		for (unsigned int i=0; i<children.size(); i++) {
			Node *node = children[i];
			std::unordered_set<Node*,Hash,NodePointerEq>::const_iterator found = closed.find(node);

            if (found == closed.end()) {
                closed.insert(node);
                nodesToAdd.push_back(node);
                nodesToAdd_idx.push_back(i);
            } else if ((*found)->depth > node->depth) {
                (*found)->depth = node->depth;
                (*found)->parentMove = node->parentMove;
                (*found)->parent = node->parent;

                nodesToAdd.push_back(node);
                nodesToAdd_idx.push_back(i);
            } else {
                delete node;
            }
		}
		numNodesGenerated += children.size();
		checkClosedTime = getTimeElapsed(t1,std::chrono::high_resolution_clock::now());

		//Get value
		//printf("GETTING HEURISTIC\n");
		t1 = std::chrono::high_resolution_clock::now();
		std::vector<float> values(nodesToAdd_idx.size());
		std::vector<float> values_temp;

        float f;
        for (unsigned int i=0; i<children.size(); i++) {
            read(sockfd,reinterpret_cast<char*>(&f),4);
            values_temp.push_back(f);
        }
        for (unsigned int i=0; i<nodesToAdd_idx.size(); i++) {
            values[i] = values_temp[nodesToAdd_idx[i]];
        }

		if (nodesToAdd.size() > 0) {
			minValue = *std::min_element(values.begin(),values.end());
			maxValue = *std::max_element(values.begin(),values.end());
		}

		heuristicTime = getTimeElapsed(t1,std::chrono::high_resolution_clock::now());

		//Compute cost
		t1 = std::chrono::high_resolution_clock::now();
		std::vector<float> costs(nodesToAdd.size());

		#pragma omp parallel for
		for (unsigned int i=0; i<nodesToAdd.size(); i++) {
			//float heuristic = std::max(nodesToAdd[i]->heuristic, values[i]);
			float cost = values[i]*(!nodesToAdd[i]->env->isSolved()) + depthPenalty*((float) nodesToAdd[i]->depth);
			costs[i] = cost;
		}

		if (nodesToAdd.size() > 0) {
			minCost = *std::min_element(costs.begin(),costs.end());
			maxCost = *std::max_element(costs.begin(),costs.end());
		}
		costTime = getTimeElapsed(t1,std::chrono::high_resolution_clock::now());


		//Add to open
		t1 = std::chrono::high_resolution_clock::now();
		//printf("ADDING TO OPEN\n");
		for (unsigned int i=0; i<nodesToAdd.size(); i++) {
			Node *nodeToAdd = nodesToAdd[i];

			nodeToAdd->cost = costs[i];
			nodeToAdd->heuristic = values[i];

			open.push(nodeToAdd);
		}

		addToQueueTime = getTimeElapsed(t1,std::chrono::high_resolution_clock::now());

		printf("Times - remOpen: %f, exp: %f, write: %f, check: %f, heur: %f, cost: %f, add: %f, goal_p: %i\n",remOpenTime,expandingTime,dataWriteTime,checkClosedTime,heuristicTime,costTime,addToQueueTime,goal_node_found_prev);

		itrTime = getTimeElapsed(startTime,std::chrono::high_resolution_clock::now());

		printf("Iteration: %i, Min/Max - Depth: %i/%i, Heur: %.2f/%.2f, Cost: %.2f/%.2f, OpenSize: %li, ClosedSize: %li, Time: %f, Num Added: %li\n\n",searchItr,minDepth,maxDepth,minValue,maxValue,minCost,maxCost,open.size(),closed.size(),itrTime,nodesToAdd.size());

		searchItr++;
	}

	printf("SOLVED!\n");

	printf("Move nums:\n");

	Node *currNode = solvedNode;
	while (currNode->depth > 0) {
		printf("%i ",currNode->parentMove);
		currNode = currNode->parent;
	}
	printf("\n");
	printf("Nodes Generated:\n%li\n",numNodesGenerated);

	double totalTime = getTimeElapsed(searchStartTime,std::chrono::high_resolution_clock::now());
	printf("Total time:\n%f\n",totalTime);
}

int main(int argc, const char *argv[]) {
	printf("The argument supplied is %s\n", argv[1]);
	
	/* Get input from file*/
	std::string input = argv[1];
	float depthPenalty = (float) atof(argv[2]);
	int numParallel = atoi(argv[3]);
	std::string socketName = argv[4];
	std::string envName = argv[5];

	std::string str;

	/* Parse State */
	std::vector<uint8_t> init;

	std::stringstream ssin(input);
	while (ssin.good()){
		int val;
		ssin >> val;
		init.push_back((uint8_t) val);
	}

	/* Search */
	printf("State:\n");
	printArr(init);
	printf("\n");

	Environment *env = NULL;
	if (envName == "puzzle15") {
		env = new PuzzleN(init,4);
	} else if (envName == "puzzle24") {
		env = new PuzzleN(init,5);
	} else if (envName == "puzzle35") {
		env = new PuzzleN(init,6);
	} else if (envName == "puzzle48") {
		env = new PuzzleN(init,7);
	} else if (envName == "cube3") {
		env = new Cube3(init);
	} else if (envName == "cube4") {
		env = new Cube4(init);
	} else if (envName == "lightsout7") {
		env = new LightsOut(init,7);
	}
	parallelWeightedAStar(env, depthPenalty, numParallel, socketName);

	delete env;

	return 0;
}
