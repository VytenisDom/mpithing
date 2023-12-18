#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

class ArrayGenerator {
public:
    static int* generateUnsortedArray(int n) {
        if (n < 0) {
            throw std::invalid_argument("Array size must be non-negative");
        }

        int* unsortedArray = new int[n];
        srand(static_cast<unsigned int>(time(nullptr)));

        for (int i = 0; i < n; i++) {
            unsortedArray[i] = rand() % (n + 1);
        }

        return unsortedArray;
    }
};

class InsertionSort {
public:
    static void insertionSort(int* array, int size) {
        for (int i = 1; i < size; ++i) {
            int key = array[i];
            int j = i - 1;

            // Move elements that are greater than key to one position ahead of their current position
            while (j >= 0 && array[j] > key) {
                array[j + 1] = array[j];
                --j;
            }

            // Place the key at its correct position in the sorted array
            array[j + 1] = key;
        }
    }

    static void parallelInsertionSort(int* array, int size, int numOfThreads) {
        int chunkSize = size / numOfThreads;
        #pragma omp parallel num_threads(numOfThreads)
        {
            #pragma omp for schedule(static, chunkSize)
            for (int i = 0; i < size; i += chunkSize) {
                int start = i;
                int end = std::min(i + chunkSize, size);
                insertionSort(array + start, end - start);
            }
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int n = 100000;
    int numOfProcesses = 8;
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);

    int* singleArray = ArrayGenerator::generateUnsortedArray(n);
    int* multiArray = new int[n];
    std::copy(singleArray, singleArray + n, multiArray);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "Generated Unsorted Arrays." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double startTimeSingle = MPI_Wtime();

    //InsertionSort::insertionSort(singleArray, n);

    double endTimeSingle = MPI_Wtime();
    double durationSingle = (endTimeSingle - startTimeSingle) * 1000; // Convert to milliseconds

    if (rank == 0) {
        std::cout << "Total execution time (single): " << durationSingle << " milliseconds." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double startTime = MPI_Wtime();
    
    InsertionSort::parallelInsertionSort(multiArray, n, numOfProcesses);

    // Gather sorted arrays from all processes to process 0
    int* sortedArray = nullptr;
    if (rank == 0) {
        sortedArray = new int[n * numOfProcesses];
    }

    MPI_Gather(multiArray, n, MPI_INT, sortedArray, n, MPI_INT, 0, MPI_COMM_WORLD);

    double endTime = MPI_Wtime();
    double duration = (endTime - startTime) * 1000; // Convert to milliseconds
    

    // Display the sorted array in process 0
    if (rank == 0) {
        std::cout << "Total execution time (multi): " << duration << " milliseconds." << std::endl;

        //std::cout << "Sorted array:" << std::endl;
        //for (int i = 0; i < n * numOfProcesses; ++i) {
        //    std::cout << sortedArray[i] << " ";
        //}
        //std::cout << std::endl;

        delete[] sortedArray;
    }

    delete[] singleArray;
    delete[] multiArray;

    MPI_Finalize();

    return 0;
}
