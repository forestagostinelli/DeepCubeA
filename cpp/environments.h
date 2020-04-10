#include <vector>
#include <fstream>

/*** Abstract Environment Class ***/
class Environment {
  public:
		virtual ~Environment()=0;
    virtual Environment *getNextState(int action) const = 0;

    virtual std::vector<Environment*> getNextStates() const = 0;

    virtual std::vector<uint8_t> getState() const = 0;

    virtual bool isSolved() const = 0;

    virtual int getNumActions() const = 0;
};

/*** PuzzleN ***/
class PuzzleN: public Environment {
	private:
		static const int numActions = 4;
		uint8_t **swapZeroIdxs;

		std::vector<uint8_t> state;
		uint8_t dim;
		int numTiles;
		uint8_t zIdx;

    virtual void construct(std::vector<uint8_t> state, uint8_t dim, uint8_t zIdx);
	public:
		PuzzleN(std::vector<uint8_t> state, uint8_t dim, uint8_t zIdx);
		PuzzleN(std::vector<uint8_t> state, uint8_t dim);
		~PuzzleN();

		
    virtual PuzzleN *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};

/*** LightsOut ***/
class LightsOut: public Environment {
	private:
		int **moveMat;

		std::vector<uint8_t> state;
		uint8_t dim;
		int numActions;
	public:
		LightsOut(std::vector<uint8_t> state, uint8_t dim);
		~LightsOut();

		
    virtual LightsOut *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};

class Cube3: public Environment {
	private:
		static const int numActions = 12;
		static constexpr int rotateIdxs_old[12][24] = 
		{
			{2, 5, 8, 8, 7, 6, 6, 3, 0, 0, 1, 2, 38, 41, 44, 20, 23, 26, 47, 50, 53, 29, 32, 35},
			{6, 3, 0, 0, 1, 2, 2, 5, 8, 8, 7, 6, 47, 50, 53, 29, 32, 35, 38, 41, 44, 20, 23, 26},
			{11, 14, 17, 17, 16, 15, 15, 12, 9, 9, 10, 11, 45, 48, 51, 18, 21, 24, 36, 39, 42, 27, 30, 33},
			{15, 12, 9, 9, 10, 11, 11, 14, 17, 17, 16, 15, 36, 39, 42, 27, 30, 33, 45, 48, 51, 18, 21, 24},
			{20, 23, 26, 26, 25, 24, 24, 21, 18, 18, 19, 20, 45, 46, 47, 0, 1, 2, 44, 43, 42, 9, 10, 11},
			{24, 21, 18, 18, 19, 20, 20, 23, 26, 26, 25, 24, 44, 43, 42, 9, 10, 11, 45, 46, 47, 0, 1, 2},
			{29, 32, 35, 35, 34, 33, 33, 30, 27, 27, 28, 29, 38, 37, 36, 6, 7, 8, 51, 52, 53, 15, 16, 17},
			{33, 30, 27, 27, 28, 29, 29, 32, 35, 35, 34, 33, 51, 52, 53, 15, 16, 17, 38, 37, 36, 6, 7, 8},
			{38, 41, 44, 44, 43, 42, 42, 39, 36, 36, 37, 38, 18, 19, 20, 2, 5, 8, 35, 34, 33, 15, 12, 9},
			{42, 39, 36, 36, 37, 38, 38, 41, 44, 44, 43, 42, 35, 34, 33, 15, 12, 9, 18, 19, 20, 2, 5, 8},
			{47, 50, 53, 53, 52, 51, 51, 48, 45, 45, 46, 47, 29, 28, 27, 0, 3, 6, 24, 25, 26, 17, 14, 11},
			{51, 48, 45, 45, 46, 47, 47, 50, 53, 53, 52, 51, 24, 25, 26, 17, 14, 11, 29, 28, 27, 0, 3, 6}
		};

		static constexpr int rotateIdxs_new[12][24] = 
		{
			{0, 1, 2, 2, 5, 8, 8, 7, 6, 6, 3, 0, 20, 23, 26, 47, 50, 53, 29, 32, 35, 38, 41, 44},
			{0, 1, 2, 2, 5, 8, 8, 7, 6, 6, 3, 0, 20, 23, 26, 47, 50, 53, 29, 32, 35, 38, 41, 44},
			{9, 10, 11, 11, 14, 17, 17, 16, 15, 15, 12, 9, 18, 21, 24, 36, 39, 42, 27, 30, 33, 45, 48, 51},
			{9, 10, 11, 11, 14, 17, 17, 16, 15, 15, 12, 9, 18, 21, 24, 36, 39, 42, 27, 30, 33, 45, 48, 51},
			{18, 19, 20, 20, 23, 26, 26, 25, 24, 24, 21, 18, 0, 1, 2, 44, 43, 42, 9, 10, 11, 45, 46, 47},
			{18, 19, 20, 20, 23, 26, 26, 25, 24, 24, 21, 18, 0, 1, 2, 44, 43, 42, 9, 10, 11, 45, 46, 47},
			{27, 28, 29, 29, 32, 35, 35, 34, 33, 33, 30, 27, 6, 7, 8, 51, 52, 53, 15, 16, 17, 38, 37, 36},
			{27, 28, 29, 29, 32, 35, 35, 34, 33, 33, 30, 27, 6, 7, 8, 51, 52, 53, 15, 16, 17, 38, 37, 36},
			{36, 37, 38, 38, 41, 44, 44, 43, 42, 42, 39, 36, 2, 5, 8, 35, 34, 33, 15, 12, 9, 18, 19, 20},
			{36, 37, 38, 38, 41, 44, 44, 43, 42, 42, 39, 36, 2, 5, 8, 35, 34, 33, 15, 12, 9, 18, 19, 20},
			{45, 46, 47, 47, 50, 53, 53, 52, 51, 51, 48, 45, 0, 3, 6, 24, 25, 26, 17, 14, 11, 29, 28, 27},
			{45, 46, 47, 47, 50, 53, 53, 52, 51, 51, 48, 45, 0, 3, 6, 24, 25, 26, 17, 14, 11, 29, 28, 27}
		};

		std::vector<uint8_t> state;
	public:
		Cube3(std::vector<uint8_t> state);
		~Cube3();

		
    virtual Cube3 *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};


class Cube4: public Environment {
	private:
		static const int numActions = 24;
		static const std::vector<std::vector<int> > rotateIdxs_old;
		static const std::vector<std::vector<int> > rotateIdxs_new;

		std::vector<uint8_t> state;
	public:
		Cube4(std::vector<uint8_t> state);
		~Cube4();
		
    virtual Cube4 *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};


