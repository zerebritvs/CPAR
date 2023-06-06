/*
 * Practica MPI - Grupo 17
 * Juan Antonio Pages Lopez
 * Sergio Sanz Sanz
 *
 * Probabilistic approach to locate maximum heights
 * Hill Climbing + Montecarlo
 *
 * MPI version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2021/2022
 *
 * v1.1
 * 
 * (c) 2022 Arturo Gonzalez Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<sys/time.h>
#include<omp.h>

/* Headers for the MPI assignment versions */
#include<mpi.h>                              
#include<stddef.h>                           

/* 
 * Global variables and macro to check errors in calls to MPI functions
 * The macro shows the provided message and the MPI string in case of error
 */
char mpi_error_string[ MPI_MAX_ERROR_STRING ];
int mpi_string_len;
#define MPI_CHECK( msg, mpi_call )	{ int check = mpi_call; if ( check != MPI_SUCCESS ) { MPI_Error_string( check, mpi_error_string, &mpi_string_len); fprintf(stderr,"MPI Error - %s - %s\n", msg, mpi_error_string ); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE ); } }


#define	PRECISION	10000

/* 
 * Structure to represent a climbing searcher 
 * 	This structure can be changed and/or optimized by the students
 */
typedef struct {
	int id;				// Searcher identifier
	int pos_row, pos_col;		// Position in the grid
	int steps;			// Steps count
	int follows;			// When it finds an explored trail, who searched that trail
} Searcher;


/* 
 * Function to get wall time
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/* 
 * Macro function to simplify accessing with two coordinates to a flattened array
 * 	This macro-function can be changed and/or optimized by the students
 */
#define accessMat( arr, exp1, exp2 )	arr[ (int)(exp1) * columns + (int)(exp2) ]



/*
 * Function: Generate height for a given position
 * 	This function can be changed and/or optimized by the students
 */
int get_height( int x, int y, int rows, int columns, float x_min, float x_max, float y_min, float y_max  ) {
	/* Calculate the coordinates of the point in the ranges */
	float x_coord = x_min + ( (x_max - x_min) / rows ) * x;
	float y_coord = y_min + ( (y_max - y_min) / columns ) * y;
	/* Compute function value */
	float value = 2 * sin(x_coord) * cos(y_coord/2) + log( fabs(y_coord - M_PI_2) );
	/* Transform to fixed point precision */
	int fixed_point = (int)( PRECISION * value );
	return fixed_point;
}

/*
 * Function: Climbing step
 * 	This function can be changed and/or optimized by the students
 */
int climbing_step(int my_begin, int my_size, int rows, int columns, Searcher *searchers, int search, int *heights, int *trails, int *tainted, float x_min, float x_max, float y_min, float y_max, int *alturas_arriba, int *alturas_abajo) {
	int search_flag = 0;
	

	/* Annotate one step more, landing counts as the first step */
	searchers[ search ].steps ++;

	/* Get starting position */
	int pos_row = searchers[ search ].pos_row;
	int pos_col = searchers[ search ].pos_col;
	
	int my_pos_row = pos_row - my_begin; //Calculamos la posicion local que usará cada proceso para acceder a las distintas matrices

	/* Stop if searcher finds another trail */
	int check;
	check = accessMat( tainted, my_pos_row, pos_col );
	accessMat( tainted, my_pos_row, pos_col ) = 1;

	if ( check != 0 ) {
		searchers[ search ].follows = accessMat( trails, my_pos_row, pos_col );
		search_flag = 1;
	}
	else {
		/* Annotate the trail */
		accessMat( trails, my_pos_row, pos_col ) = searchers[search].id;

		/* Compute the height */
		accessMat( heights, my_pos_row, pos_col ) = get_height( pos_row, pos_col, rows, columns, x_min, x_max, y_min, y_max );

		/* Locate the highest climbing direction */
		float local_max = accessMat( heights, my_pos_row, pos_col );
		int climbing_direction = 0;
		//Comprobacion si el buscador se sale por arriba, guardaremos estas alturas obtenidas en un array distinto de heights, concretamente en alturas_arriba y con ello obtendremos el maximo local
		if ( pos_row > 0 ) {
			//Comprueba que esa resta no es la fila 0, ya que en ese caso, al hacer el step (pos_row -1) se sale de la zona local
			if(my_pos_row != 0){
				/* Compute the height in the neighbor if needed */
				if ( accessMat( heights, my_pos_row-1, pos_col ) == INT_MIN ) 
					accessMat( heights, my_pos_row-1, pos_col ) = get_height( pos_row-1, pos_col, rows, columns, x_min, x_max, y_min, y_max );

				/* Annotate the travelling direction if higher */
				if ( accessMat( heights, my_pos_row-1, pos_col ) > local_max ) {
					climbing_direction = 1;
					local_max = accessMat( heights, my_pos_row-1, pos_col );
				}
			}else{
				if ( accessMat( alturas_arriba, 0, pos_col ) == INT_MIN ) 
					accessMat( alturas_arriba, 0, pos_col ) = get_height( pos_row-1, pos_col, rows, columns, x_min, x_max, y_min, y_max );
				if ( accessMat( alturas_arriba, 0, pos_col ) > local_max ) {
					climbing_direction = 1;
					local_max = accessMat( alturas_arriba, 0, pos_col );
				}
				
			}
		}

		//Comprobacion si el buscador se sale por abajo, guardaremos estas alturas obtenidas en un array distinto de heights, concretamente en alturas_abajo y con ello obtendremos el maximo local
		if ( pos_row < rows-1 ) {
			//Comprueba que esa resta no es la ultima fila, ya que en ese caso, al hacer el step (pos_row +1) se sale de la zona local
			if(my_pos_row != my_size-1){
				/* Compute the height in the neighbor if needed */
				if ( accessMat( heights, my_pos_row+1, pos_col ) == INT_MIN )
					accessMat( heights, my_pos_row+1, pos_col ) = get_height( pos_row+1, pos_col, rows, columns, x_min, x_max, y_min, y_max );

				/* Annotate the travelling direction if higher */
				if ( accessMat( heights, my_pos_row+1, pos_col ) > local_max ) {
					climbing_direction = 2;
					local_max = accessMat( heights, my_pos_row+1, pos_col );
				}
			}else{
				if ( accessMat( alturas_abajo, 0, pos_col ) == INT_MIN ) 
					accessMat( alturas_abajo, 0, pos_col ) = get_height( pos_row+1, pos_col, rows, columns, x_min, x_max, y_min, y_max );
				if ( accessMat( alturas_abajo, 0, pos_col ) > local_max ) {
					climbing_direction = 2;
					local_max = accessMat( alturas_abajo, 0, pos_col );
				}
			}
		
		}
		if ( pos_col > 0 ) {
			/* Compute the height in the neighbor if needed */
			if ( accessMat( heights, my_pos_row, pos_col-1 ) == INT_MIN ) 
				accessMat( heights, my_pos_row, pos_col-1 ) = get_height( pos_row, pos_col-1, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( accessMat( heights, my_pos_row, pos_col-1 ) > local_max ) {
				climbing_direction = 3;
				local_max = accessMat( heights, my_pos_row, pos_col-1 );
			}
		}
		if ( pos_col < columns-1 ) {
			/* Compute the height in the neighbor if needed */
			if ( accessMat( heights, my_pos_row, pos_col+1 ) == INT_MIN ) 
				accessMat( heights, my_pos_row, pos_col+1 ) = get_height( pos_row, pos_col+1, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( accessMat( heights, my_pos_row, pos_col+1 ) > local_max ) {
				climbing_direction = 4;
				local_max = accessMat( heights, my_pos_row, pos_col+1 );
			}
		}

		/* Stop if local maximum is reached */
		if ( climbing_direction == 0 ) {
			searchers[ search ].follows = searchers[ search ].id;
			search_flag = 1;
		}

		/* Move in the chosen direction: 0 does not change coordinates */
		switch( climbing_direction ) {
			case 1: pos_row--; break;
			case 2: pos_row++; break;
			case 3: pos_col--; break;
			case 4: pos_col++; break;
		}
		searchers[ search ].pos_row = pos_row;
		searchers[ search ].pos_col = pos_col;
		
		//Comprobar si se sale en la iteracion por abajo
		if(pos_row < my_begin){
			search_flag = -1;
		}
		//Comprobar si se sale en la iteracion por arriba
		if(pos_row >= my_begin + my_size){
			search_flag = -2;
		}
	}

	/* Return a flag to indicate if search should stop */
	return search_flag;
}


#ifdef DEBUG
/* 
 * Function: Print the current state of the simulation 
 */
void print_heights( int rows, int columns, int *heights ) {
	/* 
	 * You don't need to optimize this function, it is only for pretty 
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;
	printf("Heights:\n");
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		printf("|");
		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( heights, i, j ) != INT_MIN ) 
				printf(" %6d", accessMat( heights, i, j ) );
			else
				printf("       ");
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n\n");
}

void print_trails( int rows, int columns, int *trails ) {
	/* 
	 * You don't need to optimize this function, it is only for pretty 
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;
	printf("Trails:\n");
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		printf("|");
		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( trails, i, j ) != -1 ) 
				printf("%7d", accessMat( trails, i, j ) );
			else
				printf("       ", accessMat( trails, i, j ) );
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n\n");
}
#endif // DEBUG

/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<rows> <columns> <x_min> <x_max> <y_min> <y_max> <searchers_density> <short_rnd1> <short_rnd2> <short_rnd3>\n");
	fprintf(stderr,"\n");
}

/*
 * Funcion creada para obtener el minimo entre dos enteros
 */
int min (int x, int y){
	if (x>y) {
		return (y);
	}  else  {
		return (x);
	}
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	// This eliminates the buffer of stdout, forcing the messages to be printed immediately
	setbuf(stdout,NULL);

	int i,j;

	// Simulation data
	int rows, columns;		// Matrix sizes
	float x_min, x_max;		// Limits of the terrain x coordinates
	float y_min, y_max;		// Limits of the terrain y coordinates

	float searchers_density;	// Density of hill climbing searchers
	unsigned short random_seq[3];	// Status of the random sequence

	int *heights = NULL;		// Heights of the terrain points
	int *trails = NULL;		// Searchers trace and trails
	int *tainted = NULL;		// Position found in a search
	int *follows = NULL;		// Compacted list of searchers "follows"
	int num_searchers;		// Number of searchers
	Searcher *searchers = NULL;	// Searchers data
	int *total_steps = NULL;	// Annotate accumulated steps to local maximums

	/* 0. Initialize MPI */
	MPI_Init( &argc, &argv );
	int rank;	
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc != 11) {
		fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	/* 1.2. Read argument values */
	rows = atoi( argv[1] );
	columns = atoi( argv[2] );
	x_min = atof( argv[3] );
	x_max = atof( argv[4] );
	y_min = atof( argv[5] );
	y_max = atof( argv[6] );
	searchers_density = atof( argv[7] );

	/* 1.3. Read random sequences initializer */
	for( i=0; i<3; i++ ) {
		random_seq[i] = (unsigned short)atoi( argv[8+i] );
	}


#ifdef DEBUG
	/* 1.4. Print arguments */
	if ( rank == 0 ) {
		printf("Arguments, Rows: %d, Columns: %d\n", rows, columns);
		printf("Arguments, x_range: ( %d, %d ), y_range( %d, %d )\n", x_min, x_max, y_min, y_max );
		printf("Arguments, searchers_density: %f\n", searchers_density );
		printf("Arguments, Init Random Sequence: %hu,%hu,%hu\n", random_seq[0], random_seq[1], random_seq[2]);
		printf("\n");
	}
#endif // DEBUG


	/* 2. Start global timer */
	MPI_CHECK( "Clock: Start-Barrier ", MPI_Barrier( MPI_COMM_WORLD ) );
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	/* Statistical data */
	int num_local_max = 0;
	int max_accum_steps = INT_MIN;
	int total_tainted = 0;
	int max_height = INT_MIN;
	int maximo_local = INT_MIN; //Almacena la altura maxima local en cada proceso
	int tainted_locales = 0;  //Almacena tainted local para cada proceso	
	int num_procs;
	MPI_Comm_size ( MPI_COMM_WORLD , &num_procs ) ;

	int *alturas_arriba = NULL; //Array de enteros utilizado para almacenar las alturas de los buscadores que salen por arriba de la zona local
	int *alturas_abajo = NULL; //Array de enteros utilizado para almacenar las alturas de los buscadores que salen por abajo de la zona local

	//La matriz ha sido dividida para los procesos por filas, obteniendo los correspondientes size y begin
	int size_local_rows = rows/num_procs ;
	int begin_local_rows = rank * size_local_rows ;
	int resto=rows%num_procs;
	
	if(resto != 0){
		if(rank<resto)size_local_rows++;
		begin_local_rows=begin_local_rows + min(rank,resto);
	}
                           

	/* 3. Initialization */
	/* 3.1. Memory allocation */
	num_searchers = (int)( rows * columns * searchers_density );

	searchers = (Searcher *)malloc( sizeof(Searcher) * num_searchers ); 
	total_steps = (int *)malloc( sizeof(int) * num_searchers ); 
	follows = (int *)malloc( sizeof(int) * num_searchers );

	Searcher *searchersLocales = NULL;
	Searcher *searchersCorrectos = NULL;
	Searcher *searchersArriba = NULL;
	Searcher *searchersAbajo = NULL;

	searchersLocales = (Searcher *)malloc( sizeof(Searcher) * num_searchers ); 
	searchersCorrectos = (Searcher *)malloc( sizeof(Searcher) * num_searchers ); 
	searchersArriba = (Searcher *)malloc( sizeof(Searcher) * num_searchers ); 
	searchersAbajo = (Searcher *)malloc( sizeof(Searcher) * num_searchers );

	
	if (searchers == NULL || total_steps == NULL || follows == NULL || searchersLocales == NULL || searchersCorrectos == NULL || searchersArriba == NULL || searchersAbajo == NULL ) {
		fprintf(stderr,"-- Error allocating searchers structures for size: %d\n", num_searchers );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	heights = (int *)malloc( sizeof(int) * (size_t)size_local_rows * (size_t)columns );
	trails = (int *)malloc( sizeof(int) * (size_t)size_local_rows * (size_t)columns );
	tainted = (int *)malloc( sizeof(int) * (size_t)size_local_rows * (size_t)columns );
	alturas_arriba = (int *)malloc( sizeof(int) * (size_t)columns * (size_t)(1) );
	alturas_abajo = (int *)malloc( sizeof(int) * (size_t)columns * (size_t)(1) );
	if ( heights == NULL || trails == NULL || tainted == NULL ) {
		fprintf(stderr,"-- Error allocating terrain structures for size: %d x %d \n", rows, columns );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	
	//Creacion del tipo derivado MPI_Searcher, para permitir el envio de buscadores entre procesos
	int          blocklengths[5] = {1,1,1,1,1};
    	MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    	MPI_Aint     offsets[5] ={offsetof(Searcher, id), offsetof(Searcher, pos_row), offsetof(Searcher, pos_col), offsetof(Searcher, steps), offsetof(Searcher, follows)};

	MPI_Datatype MPI_Searcher;
	
	MPI_Type_create_struct(5, blocklengths, offsets, types, &MPI_Searcher);    	
	MPI_Type_commit(&MPI_Searcher);
	

	/* 3.2. Terrain initialization */
	
	/*Inicializacion de arrays para alturas*/
	for(int k=0; k<columns; k++ ) {
			accessMat( alturas_arriba, 0, k) = INT_MIN;
			accessMat( alturas_abajo, 0, k) = INT_MIN;
		}


	for( i=0; i<size_local_rows; i++ ) {
		for( j=0; j<columns; j++ ) {
			accessMat( heights, i, j) = INT_MIN;
			accessMat( trails, i, j  ) = -1;
			accessMat( tainted, i, j  ) = 0;
		}
	}

	/* 3.3. Searchers initialization */
	int search;
	int numLocales=0;
	for( search = 0; search < num_searchers; search++ ) {
		searchers[ search ].id = search;
		searchers[ search ].pos_row = (int)( rows * erand48( random_seq ) );
		searchers[ search ].pos_col = (int)( columns * erand48( random_seq ) );
		searchers[ search ].steps = 0;
		searchers[ search ].follows = -1;
		total_steps[ search ] = 0;
		
		//Comprobacion para almacenar los searchers locales de cada proceso a partir de sus coordenadas
		if(searchers[ search ].pos_row>=begin_local_rows && searchers[ search ].pos_row<(size_local_rows+begin_local_rows) ){
			searchersLocales[numLocales] = searchers[search];
			numLocales++;
		}
	}
	
	/* 4. Compute searchers climbing trails */
	
	int correctos = 0;
	int abajo = 0;
	int arriba = 0;
	int flag_envios=0; //Flag para comprobar que todos los envios han sido completados
	while(!flag_envios){
		
		for( search = 0; search < numLocales; search++ ) {
			int search_flag = 0;
			while( ! search_flag ) {	
					search_flag = climbing_step( begin_local_rows, size_local_rows ,rows, columns, searchersLocales, search, heights, trails, tainted, x_min, x_max, y_min, y_max, alturas_arriba, alturas_abajo);
				
			}

			if(search_flag == -1){ //Comprueba si el searcher se pasa por arriba
				searchersArriba[arriba]=searchersLocales[ search ];
				arriba++;
			}

			if(search_flag == -2){ //Comprueba si el searcher se pasa por abajo
				searchersAbajo[abajo]=searchersLocales[ search ];
				abajo++;
			} 
			

			if(search_flag == 1){// Si el searcher pertenece al proceso correcto
				searchersCorrectos[correctos]=searchersLocales[ search ];
				correctos++;
			}	
		}
	
		int sizeRecibo = 0;
		numLocales = 0;
		MPI_Request requestArriba;
		MPI_Request requestAbajo;
		/*Envios hacia proceso siguiente y anterior de los buscadores*/
		if(rank != 0){
			MPI_CHECK("Enviamos arriba", MPI_Isend(searchersArriba, arriba, MPI_Searcher, rank-1, 2000, MPI_COMM_WORLD, &requestArriba));
		}
		if(rank != num_procs-1){
			MPI_CHECK("Enviamos abajo", MPI_Isend(searchersAbajo, abajo, MPI_Searcher, rank+1, 2000, MPI_COMM_WORLD, &requestAbajo));
		}
		
		/*Recepciones de los buscadores en sus correspondientes procesos*/
		if(rank!=0){
			MPI_Status status;
			MPI_CHECK("Recibimos de arriba", MPI_Recv(searchersLocales, num_searchers, MPI_Searcher,rank-1,2000, MPI_COMM_WORLD, &status));
			MPI_Get_count(&status, MPI_Searcher, &sizeRecibo);
			numLocales+=sizeRecibo;
		}
		
		
		if(rank!=num_procs-1){
			MPI_Status status;
			MPI_CHECK("Recibimos de abajo", MPI_Recv(&searchersLocales[numLocales], num_searchers - numLocales, MPI_Searcher,rank+1,2000, MPI_COMM_WORLD, &status));
			MPI_Get_count(&status, MPI_Searcher, &sizeRecibo);
			numLocales+=sizeRecibo;
				
		}
		
		/*Espera a la finalizacion de las recepciones*/
		if(rank != num_procs-1){
			MPI_Wait(&requestAbajo, MPI_STATUS_IGNORE);
			abajo = 0;
		}
		
		if(rank != 0){
			MPI_Wait(&requestArriba, MPI_STATUS_IGNORE);
			arriba = 0;
		}
		
		
		
		int finish=0;	
		/*Reduce para comprobar que finalizan todos los envios*/	
		MPI_CHECK("Reduccion para comprobar que acaba",MPI_Allreduce(&correctos, &finish, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
		if(finish==num_searchers){flag_envios=1;}
		
	}
		
	int acabadosAux = correctos; //Variable auxiliar que se usa mas tarde, ya que correctos es modificado
	
#ifdef DEBUG
/* Print computed heights at the end of the search.
 * You can ignore this debug feature.
 * If you want to use this functionality in parallel processes BEWARE: You should 
 * modify it to sincronize the processes to print in order, and each process should
 * prints only its part */
print_heights( rows, columns, heights );
#endif
	
	/*Todos los procesos envian sus buscadores al proceso 0 y la variable del tamaño del array*/
	if(rank != 0){
		MPI_CHECK("Enviamos a proceso el tamaño", MPI_Send(&correctos, 1, MPI_INT, 0, 2000, MPI_COMM_WORLD));
		MPI_CHECK("Enviamos a proceso 0", MPI_Send(searchersCorrectos, correctos, MPI_Searcher, 0, 2000, MPI_COMM_WORLD));
	}else{
		Searcher *auxiliar = NULL;
		auxiliar = (Searcher *)malloc( sizeof(Searcher) * (num_searchers) );
		int recibidos = correctos;
		//Estando en el proceso 0, metemos en las primeras posiciones estos buscadores de 0
	
		for (int i = 0; i<correctos; i++){
			searchers[i]=searchersCorrectos[i];
		}

		//Recibimos los searchers de los demás procesos y los añadimos al mismo array
		
		for (int i=1;i<num_procs;i++){
			MPI_Status statusFinal;
			MPI_CHECK("Recibimos la variable correcto", MPI_Recv(&correctos, 1, MPI_INT,i,2000, MPI_COMM_WORLD, &statusFinal));
			MPI_CHECK("Recibimos en proceso 0", MPI_Recv(auxiliar, correctos, MPI_Searcher,i,2000, MPI_COMM_WORLD, &statusFinal));
			
			for(int k=0; k<correctos;k++){
				searchers[recibidos + k] = auxiliar[k];
			}
			recibidos+=correctos;	
		}
	
		free(auxiliar);												
	/* 5. Compute the leading follower of each searcher */	
		for( search = 0; search < num_searchers; search++ ) {	
			follows[ searchers[ search ].id ] = searchers[ search ].follows;
		}

		for( search = 0; search < num_searchers; search++ ) {
			int search_flag = 0;
			int parent = searchers[ search ].id;
			int follows_to = follows[ parent ];
			while( ! search_flag ) {
				if ( follows_to == parent ){
					search_flag = 1;
				}
				else {
					parent = follows_to;
					follows_to = follows[ parent ];
				}
			}
			searchers[ search ].follows = follows_to;
		}
		
		/* 6. Compute accumulated trail steps to each maximum */
		for( search = 0; search < num_searchers; search++ ) {
			int pos_max = searchers[ search ].follows;
			total_steps[ pos_max ] += searchers[ search ].steps;		
		}
		
		/* 7. Compute statistical data */
		for( search = 0; search < num_searchers; search++ ) {
			
			int id = searchers[ search ].id;
			/* Maximum of accumulated trail steps to a local maximum */
			if ( max_accum_steps < total_steps[ id ] ) 
				max_accum_steps = total_steps[ id ];

			/* If this searcher found a maximum, check the maximum value */
			if ( searchers[ search ].follows == id ) {
				num_local_max++;	
			}
		}
	}
		/*La busqueda de la altura maxima la realiza localmente cada proceso*/
		for( search = 0; search < acabadosAux; search++ ) {
			int id2 = searchersCorrectos[ search ].id;
			if ( searchersCorrectos[ search ].follows == id2 ) {
				int pos_row = searchersCorrectos[ search ].pos_row-begin_local_rows;
				int pos_col = searchersCorrectos[ search ].pos_col;
				if ( maximo_local < accessMat( heights, pos_row, pos_col ) ) 
					maximo_local = accessMat( heights, pos_row, pos_col );
			}
		}
		
		
		/*Liberacion de memoria dinamica*/	
		MPI_Type_free(&MPI_Searcher);
		free( searchersLocales );
		free( searchersCorrectos );
		free( searchersArriba );
		free( searchersAbajo );
		free( alturas_arriba );
		free( alturas_abajo );

		/*MPI_Reduce para obtener el maximo de alturas del global*/
		MPI_CHECK("Maximo de heigths todos los procesos", MPI_Reduce(&maximo_local, &max_height, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD));

	
		/*La suma de tainted se realiza localmente en cada proceso*/
		for( i=0; i<size_local_rows; i++ ) {
			for( j=0; j<columns; j++ ) {
				if ( accessMat( tainted, i, j ) == 1 ) 
					tainted_locales++;
			}
		}
		
		/*MPI_Reduce para obtener la suma total global de tainted con todos los procesos*/
		MPI_CHECK("Maximo de tainted todos los procesos", MPI_Reduce(&tainted_locales, &total_tainted, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
	
	
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	MPI_CHECK( "End-Barrier", MPI_Barrier( MPI_COMM_WORLD ) );
	ttotal = cp_Wtime() - ttotal;

	/* 6. Output for leaderboard */
	if ( rank == 0 ) { 
		printf("\n");
		/* 6.1. Total computation time */
		printf("Time: %lf\n", ttotal );

		/* 6.2. Results: Statistics */
		printf("Result: %d, %d, %d, %d\n\n", 
			num_local_max,
			max_height,
			max_accum_steps,
			total_tainted );
	}
			
	/* 7. Free resources */	
	free( searchers );
	free( total_steps );
	free( follows );
	free( heights );
	free( trails );
	free( tainted );

	/* 8. End */
	MPI_Finalize();
	return 0;
}
