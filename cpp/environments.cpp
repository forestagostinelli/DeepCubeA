#include "environments.h"
#include <map>

uint8_t **getSwapZeroIdxs(int dim) {
  uint8_t **swapZeroIdxs = new uint8_t*[dim*dim];

  for (int i=0; i<(dim*dim); i++) {
    swapZeroIdxs[i] = new uint8_t[4];
  }

  for (int move=0; move<4; move++) {
    for (int i=0; i<dim; i++) {
      for (int j=0; j<dim; j++) {
        int zIdx = i*dim + j;
        bool isEligible;

        int swap_i;
        int swap_j;
        if (move == 0) { // U
            isEligible = i < (dim-1);
            swap_i = i+1;
            swap_j = j;
        } else if (move == 1) { // D
            isEligible = i > 0;
            swap_i = i-1;
            swap_j = j;
        } else if (move == 2) { // L
            isEligible = j < (dim-1);
            swap_i = i;
            swap_j = j+1;
        } else if (move == 3) { // R
            isEligible = j > 0;
            swap_i = i;
            swap_j = j-1;
        }
        if (isEligible) {
          swapZeroIdxs[zIdx][move] = (uint8_t) (swap_i*dim + swap_j);
        } else {
          swapZeroIdxs[zIdx][move] = (uint8_t) zIdx;
        }
      }
    }
  }

  return(swapZeroIdxs);
}

Environment::~Environment() {
}

/// PuzzleN
uint8_t **swapZeroIdxs4 = getSwapZeroIdxs(4);
uint8_t **swapZeroIdxs5 = getSwapZeroIdxs(5);
uint8_t **swapZeroIdxs6 = getSwapZeroIdxs(6);
uint8_t **swapZeroIdxs7 = getSwapZeroIdxs(7);

PuzzleN::PuzzleN(std::vector<uint8_t> state, uint8_t dim, uint8_t zIdx) {
	this->construct(state,dim,zIdx);
}

PuzzleN::PuzzleN(std::vector<uint8_t> state, uint8_t dim) {
	uint8_t zIdx = 0;
	for (uint8_t i=0; i<state.size(); i++) {
		if (state[i] == 0) {
			zIdx = i;
			break;
		}
	}

	this->construct(state,dim,zIdx);
}

PuzzleN::~PuzzleN() {}

void PuzzleN::construct(std::vector<uint8_t> state, uint8_t dim, uint8_t zIdx) {
	this->state = state;
	this->dim = dim;
	this->zIdx = zIdx;
	this->numTiles = dim*dim;

	if (dim == 4) {
		swapZeroIdxs = swapZeroIdxs4;
	} else if (dim == 5) {
		swapZeroIdxs = swapZeroIdxs5;
	} else if (dim == 6) {
		swapZeroIdxs = swapZeroIdxs6;
	} else if (dim == 7) {
		swapZeroIdxs = swapZeroIdxs7;
	}
}

PuzzleN *PuzzleN::getNextState(const int action) const {
	std::vector<uint8_t> newState(this->state);
	
	uint8_t swapZeroIdx = this->swapZeroIdxs[this->zIdx][action];

	uint8_t val = newState[swapZeroIdx];
	newState[this->zIdx] = val;
	newState[swapZeroIdx] = 0;

	PuzzleN *nextState = new PuzzleN(newState,this->dim,swapZeroIdx);

	return(nextState);
}

std::vector<Environment*> PuzzleN::getNextStates() const {
	std::vector<Environment*> nextStates;
	for (int i=0; i<numActions; i++) {
		nextStates.push_back(this->getNextState(i));
	}

	return(nextStates);
}

std::vector<uint8_t> PuzzleN::getState() const {
	return(this->state);
}

bool PuzzleN::isSolved() const {
	bool isSolved = true;
	for (int i=0; i < this->numTiles; i++) {
		isSolved = isSolved & (this->state[i] == ((i+1) % this->numTiles));
	}

	return(isSolved);
}

int PuzzleN::getNumActions() const {
	return(this->numActions);
}

///LightsOut
int **getMoveMat(int dim) {
  int **moveMat = new int*[dim*dim];
  for (int move=0; move<(dim*dim); move++) {
    moveMat[move] = new int[5];

		int xPos = move/dim;
		int yPos = move % dim;

		int right = xPos < (dim-1) ? move + dim : move;
		int left = xPos > 0 ? move - dim : move;
		int up = yPos < (dim-1) ? move + 1 : move;
		int down = yPos > 0 ? move - 1 : move;

		moveMat[move][0] = move;
		moveMat[move][1] = right;
		moveMat[move][2] = left;
		moveMat[move][3] = up;
		moveMat[move][4] = down;
  }

	return(moveMat);
}
int **moveMat7 = getMoveMat(7);

LightsOut::LightsOut(std::vector<uint8_t> state, uint8_t dim) {
	this->state = state;
	this->dim = dim;

	this->numActions = (this->dim)*(this->dim);

	if (dim == 7) {
		moveMat = moveMat7;
	}
}

LightsOut::~LightsOut() {}

LightsOut *LightsOut::getNextState(const int action) const {
	std::vector<uint8_t> newState(this->state);
	
	
	for (int i=0; i<5; i++) {
		newState[moveMat[action][i]] = (uint8_t) ((int) (this->state[moveMat[action][i]] + 1)) % 2;
	}

	LightsOut *nextState = new LightsOut(newState,this->dim);

	return(nextState);
}

std::vector<Environment*> LightsOut::getNextStates() const {
	std::vector<Environment*> nextStates;
	for (int i=0; i<numActions; i++) {
		nextStates.push_back(this->getNextState(i));
	}

	return(nextStates);
}

std::vector<uint8_t> LightsOut::getState() const {
	return(this->state);
}

bool LightsOut::isSolved() const {
	bool isSolved = true;
	const int numTiles = (this->dim)*(this->dim);
	for (int i=0; i<numTiles; i++) {
		isSolved = isSolved & (this->state[i] == 0);
	}

	return(isSolved);
}

int LightsOut::getNumActions() const {
	return(this->numActions);
}


/// Cube3
constexpr int Cube3::rotateIdxs_old[12][24];
constexpr int Cube3::rotateIdxs_new[12][24];

Cube3::Cube3(std::vector<uint8_t> state) {
	this->state = state;
}

Cube3::~Cube3() {}


Cube3 *Cube3::getNextState(const int action) const {
	std::vector<uint8_t> newState(this->state);
	
	for (int i=0; i<24; i++) {
		const int oldIdx = this->rotateIdxs_old[action][i];
		const int newIdx = this->rotateIdxs_new[action][i];
		newState[newIdx] = this->state[oldIdx];
	}

	Cube3 *nextState = new Cube3(newState);

	return(nextState);
}

std::vector<Environment*> Cube3::getNextStates() const {
	std::vector<Environment*> nextStates;
	for (int i=0; i<numActions; i++) {
		nextStates.push_back(this->getNextState(i));
	}

	return(nextStates);
}

std::vector<uint8_t> Cube3::getState() const {
	return(this->state);
}

bool Cube3::isSolved() const {
	bool isSolved = true;
	for (int i=0; i<54; i++) {
		isSolved = isSolved & (this->state[i] == i);
	}

	return(isSolved);
}

int Cube3::getNumActions() const {
	return(this->numActions);
}

/*** Cube4 ***/
std::vector<int> U0_n1 = {3, 7, 11, 15, 15, 14, 13, 12, 12, 8, 4, 0, 0, 1, 2, 3, 6, 10, 10, 9, 9, 5, 5, 6, 67, 71, 75, 79, 35, 39, 43, 47, 83, 87, 91, 95, 51, 55, 59, 63};
std::vector<int> U0_1 = {12, 8, 4, 0, 0, 1, 2, 3, 3, 7, 11, 15, 15, 14, 13, 12, 9, 5, 5, 6, 6, 10, 10, 9, 83, 87, 91, 95, 51, 55, 59, 63, 67, 71, 75, 79, 35, 39, 43, 47};
std::vector<int> D0_n1 = {19, 23, 27, 31, 31, 30, 29, 28, 28, 24, 20, 16, 16, 17, 18, 19, 22, 26, 26, 25, 25, 21, 21, 22, 80, 84, 88, 92, 32, 36, 40, 44, 64, 68, 72, 76, 48, 52, 56, 60};
std::vector<int> D0_1 = {28, 24, 20, 16, 16, 17, 18, 19, 19, 23, 27, 31, 31, 30, 29, 28, 25, 21, 21, 22, 22, 26, 26, 25, 64, 68, 72, 76, 48, 52, 56, 60, 80, 84, 88, 92, 32, 36, 40, 44};
std::vector<int> L0_n1 = {35, 39, 43, 47, 47, 46, 45, 44, 44, 40, 36, 32, 32, 33, 34, 35, 38, 42, 42, 41, 41, 37, 37, 38, 80, 81, 82, 83, 0, 1, 2, 3, 79, 78, 77, 76, 16, 17, 18, 19};
std::vector<int> L0_1 = {44, 40, 36, 32, 32, 33, 34, 35, 35, 39, 43, 47, 47, 46, 45, 44, 41, 37, 37, 38, 38, 42, 42, 41, 79, 78, 77, 76, 16, 17, 18, 19, 80, 81, 82, 83, 0, 1, 2, 3};
std::vector<int> R0_n1 = {51, 55, 59, 63, 63, 62, 61, 60, 60, 56, 52, 48, 48, 49, 50, 51, 54, 58, 58, 57, 57, 53, 53, 54, 67, 66, 65, 64, 12, 13, 14, 15, 92, 93, 94, 95, 28, 29, 30, 31};
std::vector<int> R0_1 = {60, 56, 52, 48, 48, 49, 50, 51, 51, 55, 59, 63, 63, 62, 61, 60, 57, 53, 53, 54, 54, 58, 58, 57, 92, 93, 94, 95, 28, 29, 30, 31, 67, 66, 65, 64, 12, 13, 14, 15};
std::vector<int> B0_n1 = {67, 71, 75, 79, 79, 78, 77, 76, 76, 72, 68, 64, 64, 65, 66, 67, 70, 74, 74, 73, 73, 69, 69, 70, 32, 33, 34, 35, 3, 7, 11, 15, 63, 62, 61, 60, 28, 24, 20, 16};
std::vector<int> B0_1 = {76, 72, 68, 64, 64, 65, 66, 67, 67, 71, 75, 79, 79, 78, 77, 76, 73, 69, 69, 70, 70, 74, 74, 73, 63, 62, 61, 60, 28, 24, 20, 16, 32, 33, 34, 35, 3, 7, 11, 15};
std::vector<int> F0_n1 = {83, 87, 91, 95, 95, 94, 93, 92, 92, 88, 84, 80, 80, 81, 82, 83, 86, 90, 90, 89, 89, 85, 85, 86, 51, 50, 49, 48, 0, 4, 8, 12, 44, 45, 46, 47, 31, 27, 23, 19};
std::vector<int> F0_1 = {92, 88, 84, 80, 80, 81, 82, 83, 83, 87, 91, 95, 95, 94, 93, 92, 89, 85, 85, 86, 86, 90, 90, 89, 44, 45, 46, 47, 31, 27, 23, 19, 51, 50, 49, 48, 0, 4, 8, 12};

std::vector<int> U1_n1 = {66, 70, 74, 78, 34, 38, 42, 46, 82, 86, 90, 94, 50, 54, 58, 62};
std::vector<int> U1_1 = {82, 86, 90, 94, 50, 54, 58, 62, 66, 70, 74, 78, 34, 38, 42, 46};
std::vector<int> D1_n1 = {81, 85, 89, 93, 33, 37, 41, 45, 65, 69, 73, 77, 49, 53, 57, 61};
std::vector<int> D1_1 = {65, 69, 73, 77, 49, 53, 57, 61, 81, 85, 89, 93, 33, 37, 41, 45};
std::vector<int> L1_n1 = {84, 85, 86, 87, 4, 5, 6, 7, 75, 74, 73, 72, 20, 21, 22, 23};
std::vector<int> L1_1 = {75, 74, 73, 72, 20, 21, 22, 23, 84, 85, 86, 87, 4, 5, 6, 7};
std::vector<int> R1_n1 = {71, 70, 69, 68, 8, 9, 10, 11, 88, 89, 90, 91, 24, 25, 26, 27};
std::vector<int> R1_1 = {88, 89, 90, 91, 24, 25, 26, 27, 71, 70, 69, 68, 8, 9, 10, 11};
std::vector<int> B1_n1 = {36, 37, 38, 39, 2, 6, 10, 14, 59, 58, 57, 56, 29, 25, 21, 17};
std::vector<int> B1_1 = {59, 58, 57, 56, 29, 25, 21, 17, 36, 37, 38, 39, 2, 6, 10, 14};
std::vector<int> F1_n1 = {55, 54, 53, 52, 1, 5, 9, 13, 40, 41, 42, 43, 30, 26, 22, 18};
std::vector<int> F1_1 = {40, 41, 42, 43, 30, 26, 22, 18, 55, 54, 53, 52, 1, 5, 9, 13};

const std::vector<std::vector<int> > Cube4::rotateIdxs_old { U0_n1, U0_1, D0_n1, D0_1, L0_n1, L0_1, R0_n1, R0_1, B0_n1, B0_1, F0_n1, F0_1, U1_n1, U1_1, D1_n1, D1_1, L1_n1, L1_1, R1_n1, R1_1, B1_n1, B1_1, F1_n1, F1_1 };

std::vector<int> U0_n1_n = {0, 1, 2, 3, 3, 7, 11, 15, 15, 14, 13, 12, 12, 8, 4, 0, 5, 6, 6, 10, 10, 9, 9, 5, 35, 39, 43, 47, 83, 87, 91, 95, 51, 55, 59, 63, 67, 71, 75, 79};
std::vector<int> U0_1_n = {0, 1, 2, 3, 3, 7, 11, 15, 15, 14, 13, 12, 12, 8, 4, 0, 5, 6, 6, 10, 10, 9, 9, 5, 35, 39, 43, 47, 83, 87, 91, 95, 51, 55, 59, 63, 67, 71, 75, 79};
std::vector<int> D0_n1_n = {16, 17, 18, 19, 19, 23, 27, 31, 31, 30, 29, 28, 28, 24, 20, 16, 21, 22, 22, 26, 26, 25, 25, 21, 32, 36, 40, 44, 64, 68, 72, 76, 48, 52, 56, 60, 80, 84, 88, 92};
std::vector<int> D0_1_n = {16, 17, 18, 19, 19, 23, 27, 31, 31, 30, 29, 28, 28, 24, 20, 16, 21, 22, 22, 26, 26, 25, 25, 21, 32, 36, 40, 44, 64, 68, 72, 76, 48, 52, 56, 60, 80, 84, 88, 92};
std::vector<int> L0_n1_n = {32, 33, 34, 35, 35, 39, 43, 47, 47, 46, 45, 44, 44, 40, 36, 32, 37, 38, 38, 42, 42, 41, 41, 37, 0, 1, 2, 3, 79, 78, 77, 76, 16, 17, 18, 19, 80, 81, 82, 83};
std::vector<int> L0_1_n = {32, 33, 34, 35, 35, 39, 43, 47, 47, 46, 45, 44, 44, 40, 36, 32, 37, 38, 38, 42, 42, 41, 41, 37, 0, 1, 2, 3, 79, 78, 77, 76, 16, 17, 18, 19, 80, 81, 82, 83};
std::vector<int> R0_n1_n = {48, 49, 50, 51, 51, 55, 59, 63, 63, 62, 61, 60, 60, 56, 52, 48, 53, 54, 54, 58, 58, 57, 57, 53, 12, 13, 14, 15, 92, 93, 94, 95, 28, 29, 30, 31, 67, 66, 65, 64};
std::vector<int> R0_1_n = {48, 49, 50, 51, 51, 55, 59, 63, 63, 62, 61, 60, 60, 56, 52, 48, 53, 54, 54, 58, 58, 57, 57, 53, 12, 13, 14, 15, 92, 93, 94, 95, 28, 29, 30, 31, 67, 66, 65, 64};
std::vector<int> B0_n1_n = {64, 65, 66, 67, 67, 71, 75, 79, 79, 78, 77, 76, 76, 72, 68, 64, 69, 70, 70, 74, 74, 73, 73, 69, 3, 7, 11, 15, 63, 62, 61, 60, 28, 24, 20, 16, 32, 33, 34, 35};
std::vector<int> B0_1_n = {64, 65, 66, 67, 67, 71, 75, 79, 79, 78, 77, 76, 76, 72, 68, 64, 69, 70, 70, 74, 74, 73, 73, 69, 3, 7, 11, 15, 63, 62, 61, 60, 28, 24, 20, 16, 32, 33, 34, 35};
std::vector<int> F0_n1_n = {80, 81, 82, 83, 83, 87, 91, 95, 95, 94, 93, 92, 92, 88, 84, 80, 85, 86, 86, 90, 90, 89, 89, 85, 0, 4, 8, 12, 44, 45, 46, 47, 31, 27, 23, 19, 51, 50, 49, 48};
std::vector<int> F0_1_n = {80, 81, 82, 83, 83, 87, 91, 95, 95, 94, 93, 92, 92, 88, 84, 80, 85, 86, 86, 90, 90, 89, 89, 85, 0, 4, 8, 12, 44, 45, 46, 47, 31, 27, 23, 19, 51, 50, 49, 48};

std::vector<int> U1_n1_n = {34, 38, 42, 46, 82, 86, 90, 94, 50, 54, 58, 62, 66, 70, 74, 78};
std::vector<int> U1_1_n = {34, 38, 42, 46, 82, 86, 90, 94, 50, 54, 58, 62, 66, 70, 74, 78};
std::vector<int> D1_n1_n = {33, 37, 41, 45, 65, 69, 73, 77, 49, 53, 57, 61, 81, 85, 89, 93};
std::vector<int> D1_1_n = {33, 37, 41, 45, 65, 69, 73, 77, 49, 53, 57, 61, 81, 85, 89, 93};
std::vector<int> L1_n1_n = {4, 5, 6, 7, 75, 74, 73, 72, 20, 21, 22, 23, 84, 85, 86, 87};
std::vector<int> L1_1_n = {4, 5, 6, 7, 75, 74, 73, 72, 20, 21, 22, 23, 84, 85, 86, 87};
std::vector<int> R1_n1_n = {8, 9, 10, 11, 88, 89, 90, 91, 24, 25, 26, 27, 71, 70, 69, 68};
std::vector<int> R1_1_n = {8, 9, 10, 11, 88, 89, 90, 91, 24, 25, 26, 27, 71, 70, 69, 68};
std::vector<int> B1_n1_n = {2, 6, 10, 14, 59, 58, 57, 56, 29, 25, 21, 17, 36, 37, 38, 39};
std::vector<int> B1_1_n = {2, 6, 10, 14, 59, 58, 57, 56, 29, 25, 21, 17, 36, 37, 38, 39};
std::vector<int> F1_n1_n = {1, 5, 9, 13, 40, 41, 42, 43, 30, 26, 22, 18, 55, 54, 53, 52};
std::vector<int> F1_1_n = {1, 5, 9, 13, 40, 41, 42, 43, 30, 26, 22, 18, 55, 54, 53, 52};

const std::vector<std::vector<int> > Cube4::rotateIdxs_new { U0_n1_n, U0_1_n, D0_n1_n, D0_1_n, L0_n1_n, L0_1_n, R0_n1_n, R0_1_n, B0_n1_n, B0_1_n, F0_n1_n, F0_1_n, U1_n1_n, U1_1_n, D1_n1_n, D1_1_n, L1_n1_n, L1_1_n, R1_n1_n, R1_1_n, B1_n1_n, B1_1_n, F1_n1_n, F1_1_n };


Cube4::Cube4(std::vector<uint8_t> state) {
	this->state = state;
}

Cube4::~Cube4() {}


Cube4 *Cube4::getNextState(const int action) const {
	std::vector<uint8_t> newState(this->state);
	
	const std::vector<int> oldIdxs = this->rotateIdxs_old[action];
	const std::vector<int> newIdxs = this->rotateIdxs_new[action];
	for (unsigned int i=0; i<oldIdxs.size(); i++) {
		const int oldIdx = oldIdxs[i];
		const int newIdx = newIdxs[i];
		newState[newIdx] = this->state[oldIdx];
	}

	Cube4 *nextState = new Cube4(newState);

	return(nextState);
}

std::vector<Environment*> Cube4::getNextStates() const {
	std::vector<Environment*> nextStates;
	for (int i=0; i<numActions; i++) {
		nextStates.push_back(this->getNextState(i));
	}

	return(nextStates);
}

std::vector<uint8_t> Cube4::getState() const {
	return(this->state);
}

bool Cube4::isSolved() const {
	bool isSolved = true;
	for (int side=0; side<6; side++) {
		int sideColor = (int) this->state[side*16]/16;
		for (int i=1; i<16; i++) {
			isSolved = isSolved & (this->state[side*16 + i]/16 == sideColor);
		}
	}

	return(isSolved);
}

int Cube4::getNumActions() const {
	return(this->numActions);
}

