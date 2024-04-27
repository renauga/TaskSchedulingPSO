#include <bits/stdc++.h>
#include <fstream>
#include <omp.h>

using namespace std;

// Define the graph structure
struct Task {
  int id;
  int weight;
  vector<int> dependencies;
  vector<int> communicationCosts; // Communication costs with other tasks
};

// Define the particle structure
struct Particle {
    vector<long double> position; // Assignment of tasks to processors
    vector<long double> velocity; // Velocity for each task assignment
    vector<long double> personal_best; // Personal best position
    int personal_best_fitness; // Personal best fitness value
};


int N; // Number of tasks
vector<Task> input_tasks;
const int M = 4;        // Number of processors
const int MAX_ITER = 10; // Maximum number of iterations
const int NUM_OF_PARTICLES = 100; // Number of particles in PSO
const long double MIN_POS = 0, MAX_POS = N-1;

// Evaluate makespan of a solution
// priority: priority[i] is the priority of i'th task in the order of tasks
int evaluateMakespan(const vector<long double> &priority) {
    priority_queue<pair<int,int>> pq;
    vector<int> deps_left(N, 0);
    vector<vector<int>> dependents(N);
    for(int i=0;i<N;i++) {
        if(input_tasks[i].dependencies.empty()) {
            pq.push({priority[i], i});
            continue;
        }
        deps_left[i] = input_tasks[i].dependencies.size();
        for(int dependency:input_tasks[i].dependencies) {
            dependents[dependency].push_back(i);
        }
    }
    priority_queue<pair<int,int>> events; // {-finishTime, task_id}
    vector<int> finishTimesForProcessor(M, 0), finishProc(N, -1), finishTimesForTask(N, -1); // Finish times for each processor
    while(!events.empty() || !pq.empty()) {
        if(!pq.empty()) {
            int task_id = pq.top().second;
            pq.pop();
            int nax = -1e9, nax2 = -1e9, nax_processor_id = -1;
            for(int dependency:input_tasks[task_id].dependencies) {
                int comm_time = finishTimesForTask[dependency] + input_tasks[task_id].communicationCosts[dependency];
                if(comm_time >= nax) {
                    nax2 = nax;
                    nax = comm_time;
                    nax_processor_id = finishProc[dependency];
                } else nax2 = max(nax2, comm_time);
            }

            // find best processor
            int best_start_time = 1e9, best_processor = -1;
            for(int i=0;i<M;i++) {
                int start_time;
                if(i == nax_processor_id) {
                    start_time = max(finishTimesForProcessor[i], nax2);
                } else start_time = max(finishTimesForProcessor[i], nax);
                if(start_time < best_start_time) {
                    best_start_time = start_time;
                    best_processor = i;
                }
            }
            finishProc[task_id] = best_processor;
            finishTimesForProcessor[best_processor] = best_start_time + input_tasks[task_id].weight;
            finishTimesForTask[task_id] = best_start_time + input_tasks[task_id].weight;
            events.push({-finishTimesForProcessor[best_processor], task_id});
        } else {
            int curr_time = -events.top().first;
            while(!events.empty() && -events.top().first == curr_time) {
                int completed_task = events.top().second;
                events.pop();
                for(int dependent: dependents[completed_task]) {
                    deps_left[dependent]--;
                    if(deps_left[dependent] == 0) pq.push({priority[dependent], dependent});
                }
            }
        }
    }

    if(*min_element(finishProc.begin(), finishProc.end()) == -1) {
        cout<<"some tasks weren't completed, cycle present" << endl;
        exit(-1);
    }
    // Maximum finish time among all processors is the makespan
    return *max_element(finishTimesForProcessor.begin(), finishTimesForProcessor.end());
}

long double PSO(long double W, long double U1, long double U2, int number_of_threads) {
    vector<Particle> particles(NUM_OF_PARTICLES);
    vector<long double> global_best;
    int global_best_fitness = 1e9;

#pragma omp parallel for num_threads(number_of_threads) schedule(dynamic)
    for(Particle &particle:particles) {
        particle.position.resize(N);
        particle.velocity.resize(N);
        for(int j=0;j<N;j++) {
            particle.position[j] = MIN_POS + (long double)rand()/RAND_MAX*(MAX_POS-MIN_POS);
            long double min_vel = MIN_POS-particle.position[j], max_vel = MAX_POS-particle.position[j];
            particle.velocity[j] = min_vel + (long double)rand()/RAND_MAX*(max_vel-min_vel);
        }
        particle.personal_best = particle.position;
        particle.personal_best_fitness = evaluateMakespan(particle.position);
    }
    for(Particle &particle:particles) {
        if(particle.personal_best_fitness < global_best_fitness) {
            global_best = particle.personal_best;
            global_best_fitness = particle.personal_best_fitness;
        }
    }
    // find optimal solution
    for(int iter=1;iter<MAX_ITER;iter++) {
        // update each particle, can be parallelized
#pragma omp parallel for num_threads(number_of_threads) schedule(dynamic)
        for(Particle &particle:particles) {
            // update velocity
            long double c1 = (long double)rand()/RAND_MAX, c2 = (long double)rand()/RAND_MAX;
            for(int i=0;i<N;i++) {
                particle.velocity[i] = W*particle.velocity[i]
                    + c1*U1*(particle.personal_best[i]-particle.position[i])
                    + c2*U2*(global_best[i]-particle.position[i]);
            }
            for(int i=0;i<N;i++) {
                particle.position[i] += particle.velocity[i];
                particle.position[i] = min(particle.position[i], MAX_POS);
                particle.position[i] = max(particle.position[i], MIN_POS);
            }
            int new_personal_fitness = evaluateMakespan(particle.position);
            if(new_personal_fitness < particle.personal_best_fitness) {
                particle.personal_best = particle.position;
                particle.personal_best_fitness = new_personal_fitness;
            }
        }
        for(Particle &particle:particles) {
            if(particle.personal_best_fitness < global_best_fitness) {
                global_best = particle.personal_best;
                global_best_fitness = particle.personal_best_fitness;
            }
        }
    }
    return global_best_fitness;
}

void take_input(string filename) {
    ifstream fin;
    fin.open(filename, ios::in);
    if(!fin) {
        cerr<<"couldn't open file: "<< filename <<endl;
        exit(-1);
    }
    int comm_cost;
    fin >> N >> comm_cost;
    input_tasks.resize(N);
    for(int i=0;i<N;i++) {
        fin >> input_tasks[i].id >> input_tasks[i].weight;
        input_tasks[i].communicationCosts.resize(N, 0);
        int num_dep;
        fin >> num_dep;
        input_tasks[i].dependencies.resize(num_dep);
        for(int j=0;j<num_dep;j++) {
            int dep;
            fin >> dep;
            input_tasks[i].dependencies[j] = dep;
            input_tasks[i].communicationCosts[dep] = comm_cost;
        }
    }
}

int main(int argc, char **argv) {
    cout << endl << endl;
    srand(time(NULL));
    string filename = argv[1];
    take_input(filename);
    // cout << "Filename: " << filename << endl;
    int number_of_threads = stoi(argv[2]);
    cout << "Calculating..." << endl;
    long double final_answer = 1e9;
    double start_time = omp_get_wtime();
    for(long double W=0.1;W<=1;W+=0.1) {
        for(long double U1=0.1;U1<=1;U1+=0.1) {
            for(long double U2=0.1;U2<=1;U2+=0.1) {
                long double fitness = PSO(W, U1, U2, number_of_threads);
                final_answer = min(final_answer, fitness);
                // cout << "W: " << W << ", U1: " << U1 << ", U2: " << U2 << ", Global Best Fitness: " << fitness << endl;
            }
        }
    }
    cout << "Final best answer: " << final_answer << endl;
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
    cout << endl << endl;
    return 0;
}
