#include "cuda_runtime.h"

#include <stdio.h>

#include "map_reduce.cuh"


////////////////////////////////////////////////////////////////////////
//// Simple test model: number of numbers that can be divided by 10 ////
////////////////////////////////////////////////////////////////////////

namespace DB10 {

	// Depending on your application
	__device__ void mapper(int* number, int* key, int* value, const int amount_pairs_per_map) {  // Get number of inputs that can be divided by 10
		*value = 1;
		if (*number % 10 == 0) {
			*key = 1;
		}
		else {
			*key = 2;
		}
	}

	// Depending on your application
	__device__ void reducer(int* value1, int* value2, int* output) {  // Get number of inputs that can be divided by 10
		*output = *value1 + *value2;
	}

	// Prepare function pointers for the kernel (only change the template parameters)
	__device__ void(*p_mapper) (int*, int*, int*, int) = mapper;
	__device__ void(*p_reducer) (int*, int*, int*) = reducer;

	void amount_numbers_divisible_by_10() {
		// Prepare function pointers (only change the template parameters)
		void(*h_mapper) (int*, int*, int*, int);
		void(*h_reducer) (int*, int*, int*);

		cudaMemcpyFromSymbol(&h_mapper, p_mapper, sizeof(void(*) (int*, int*, int*, int)));
		cudaMemcpyFromSymbol(&h_reducer, p_reducer, sizeof(void(*) (int*, int*, int*)));

		// Input always has be to be a one dimensional array (If multidimension is needed, use structs)
		// Size has to be a constant value
		const int input_size = 10000;
		int input_array[input_size] = { 1, 2, 3, 10, 50, 45, 30, 100, 74, 30, 31 , 32 };
		for (int i = 12; i < input_size; i++) {
			if (i % 2 == 1)
				input_array[i] = 1;
			else
				input_array[i] = 10;

		}
		int* input = (int*)malloc(input_size * sizeof(int));
		std::copy(input_array, input_array + input_size, input);

		// Ouput always has be to be a one dimensional array (If multidimension is needed, use structs)
		int output_size = 4;
		int* output_keys = (int*)malloc(output_size * sizeof(int));
		int* output_values = (int*)malloc(output_size * sizeof(int));

		// Max amount of key-value pairs that can be made by 1 input a.k.a. 1 execution of the mapper
		const int amount_pairs_per_map = 1;

		// Print the input (optional)
		printf("\nInput numbers: { ");
		for (int i = 0; i < input_size; i++) {
			printf("%i, ", input[i]);
		}
		printf("\b\b }\n");

		// Template parameters: Input type, Key type, Value type
		output_size = MapReduce<int, int, int>(input, input_size, output_keys, output_values, output_size, amount_pairs_per_map, h_mapper, h_reducer);

		// Print the output (optional)
		printf("\n# numbers divisible by 10: %i\n", output_values[0]);  // index 0 are empty values, caused by the unused pairs
		printf("\n# numbers not divisible by 10: %i\n\n", output_values[1]);
	}

}  // End namespace DB10
