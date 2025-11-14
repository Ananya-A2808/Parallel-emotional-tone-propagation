// parallel_update.cpp
// Usage:
// cpp/bin/parallel_update graph.txt states.txt output_states.txt history.txt T [alpha] [threads]
#include <bits/stdc++.h>
#include <omp.h>
#include <iomanip>
using namespace std;

int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0] << " graph.txt states.txt output_states.txt history.txt T [alpha] [threads]\n";
        return 1;
    }
    string graph_file = argv[1];
    string states_file = argv[2];
    string out_states_file = argv[3];
    string history_file = argv[4];
    int T = stoi(argv[5]);
    double alpha = 0.3;
    if (argc >= 7) alpha = stod(argv[6]);
    int threads = 0;
    if (argc >= 8) threads = stoi(argv[7]);

    // disable OpenMP dynamic thread adjustment and set threads if requested
    omp_set_dynamic(0);
    if (threads > 0) omp_set_num_threads(threads);

    // Read graph: skip empty lines and comments (#)
    ifstream gf(graph_file);
    if (!gf.is_open()) { cerr << "Cannot open graph file: " << graph_file << endl; return 2; }
    long long N_ll = 0, M_ll = 0;
    // read header robustly
    {
        string line;
        while (getline(gf, line)) {
            if (line.size()==0) continue;
            // skip comments
            if (line[0]=='#') continue;
            stringstream ss(line);
            if (ss >> N_ll >> M_ll) break;
        }
    }
    if (N_ll <= 0) { cerr << "Invalid N in graph file\n"; return 2; }
    const size_t N = static_cast<size_t>(N_ll);
    const size_t M = static_cast<size_t>(M_ll);

    vector<pair<int,int>> edges;
    edges.reserve(M);
    // read edges lines
    string line;
    while (getline(gf, line)) {
        if (line.size()==0) continue;
        if (line[0]=='#') continue;
        stringstream ss(line);
        long long u_ll, v_ll;
        if (!(ss >> u_ll >> v_ll)) continue;
        edges.emplace_back(static_cast<int>(u_ll), static_cast<int>(v_ll));
    }
    gf.close();

    if (edges.size() != M) {
        // Warning but continue if counts differ (some edgelists omit header)
        cerr << "Warning: expected M=" << M << " edges, but read " << edges.size() << ". Proceeding.\n";
    }

    // Build incoming neighbors CSR
    vector<int> indeg(N, 0);
    for (auto &e: edges) {
        int v = e.second;
        if (v < 0 || static_cast<size_t>(v) >= N) continue;
        indeg[v]++;
    }
    vector<size_t> offsets(N+1, 0);
    for (size_t i=0;i<N;i++) offsets[i+1] = offsets[i] + indeg[i];
    vector<int> nbrs(offsets[N], -1);
    vector<size_t> pos = offsets;
    for (auto &e: edges) {
        int u = e.first, v = e.second;
        if (v < 0 || static_cast<size_t>(v) >= N) continue;
        nbrs[pos[v]++] = u;
    }

    // Read states
    ifstream sf(states_file);
    if (!sf.is_open()) { cerr << "Cannot open states file: " << states_file << endl; return 3; }
    vector<double> states(N, 0.0);
    size_t read_count = 0;
    while (read_count < N && getline(sf, line)) {
        if (line.size()==0) continue;
        if (line[0]=='#') continue;
        stringstream ss(line);
        double val;
        if (!(ss >> val)) continue;
        states[read_count++] = val;
    }
    sf.close();
    if (read_count < N) {
        cerr << "states file had fewer than N entries (" << read_count << " < " << N << ")\n";
        return 4;
    }

    // Optimize chunk size: larger chunks reduce overhead for small graphs
    // For large graphs, use smaller chunks for better load balance
    // For high thread counts, use larger chunks to reduce contention
    int used_threads = (threads > 0) ? threads : omp_get_max_threads();
    int chunk = 1;
    const char* chunk_env = getenv("OMP_CHUNK_SIZE");
    if (chunk_env && atoi(chunk_env) > 0) {
        chunk = atoi(chunk_env);
    } else {
        // Auto-tune chunk size based on graph size and thread count
        // For high thread counts (>4), use larger chunks to reduce memory contention
        if (N < 1000) {
            chunk = std::max(1, (int)(N / (used_threads * 4)));
        } else if (N < 10000) {
            // For medium graphs, adjust chunk based on thread count
            if (used_threads <= 4) {
                chunk = std::max(1, (int)(N / (used_threads * 8)));
            } else {
                // For >4 threads, use larger chunks to reduce memory bandwidth contention
                chunk = std::max(1, (int)(N / (used_threads * 4)));
            }
        } else {
            // For large graphs, use smaller chunks but still account for thread count
            if (used_threads <= 4) {
                chunk = std::max(1, (int)(N / (used_threads * 16)));
            } else {
                // For >4 threads on large graphs, use medium chunks
                chunk = std::max(1, (int)(N / (used_threads * 8)));
            }
        }
    }

    // Align arrays to cache line boundaries to avoid false sharing
    // Use separate arrays per thread for reduction to minimize contention
    vector<double> new_states(N, 0.0);
    vector<double> history;
    history.reserve(T);

    double t_start = omp_get_wtime();
    int print_interval = std::max(1, T / 20);  // Print ~20 progress updates
    
    cerr << "[Parallel] Starting simulation: " << T << " steps, " << N << " nodes, alpha=" << alpha << ", threads=" << used_threads << "\n";
    cerr.flush();
    
    for (int t=0; t<T; ++t) {
        double global_sum = 0.0;
        
        // Choose scheduling strategy based on graph size and thread count
        // For high thread counts, dynamic scheduling can help with load imbalance
        if (N < 500) {
            // Use guided scheduling for small graphs to minimize overhead
            #pragma omp parallel for reduction(+:global_sum) schedule(guided)
            for (size_t v=0; v<N; ++v) {
                size_t start = offsets[v];
                size_t end = offsets[v+1];
                if (start == end) {
                    new_states[v] = states[v];
                } else {
                    double sum = 0.0;
                    for (size_t j=start; j<end; ++j) sum += states[ static_cast<size_t>(nbrs[j]) ];
                    double avg = sum / double(end - start);
                    new_states[v] = (1.0 - alpha) * states[v] + alpha * avg;
                }
                global_sum += new_states[v];
            }
        } else if (used_threads > 4) {
            // For high thread counts, use dynamic scheduling to better handle load imbalance
            // This helps when memory bandwidth becomes a bottleneck
            #pragma omp parallel for reduction(+:global_sum) schedule(dynamic, chunk)
            for (size_t v=0; v<N; ++v) {
                size_t start = offsets[v];
                size_t end = offsets[v+1];
                if (start == end) {
                    new_states[v] = states[v];
                } else {
                    double sum = 0.0;
                    for (size_t j=start; j<end; ++j) sum += states[ static_cast<size_t>(nbrs[j]) ];
                    double avg = sum / double(end - start);
                    new_states[v] = (1.0 - alpha) * states[v] + alpha * avg;
                }
                global_sum += new_states[v];
            }
        } else {
            // Use static scheduling with optimized chunk for lower thread counts
            #pragma omp parallel for reduction(+:global_sum) schedule(static,chunk)
            for (size_t v=0; v<N; ++v) {
                size_t start = offsets[v];
                size_t end = offsets[v+1];
                if (start == end) {
                    new_states[v] = states[v];
                } else {
                    double sum = 0.0;
                    for (size_t j=start; j<end; ++j) sum += states[ static_cast<size_t>(nbrs[j]) ];
                    double avg = sum / double(end - start);
                    new_states[v] = (1.0 - alpha) * states[v] + alpha * avg;
                }
                global_sum += new_states[v];
            }
        }
        states.swap(new_states);
        history.push_back(global_sum / double(N));
        
        // Progress logging
        if (t % print_interval == 0 || t == T - 1) {
            double elapsed = omp_get_wtime() - t_start;
            double progress = 100.0 * (t + 1) / T;
            double rate = (t + 1) / elapsed;
            double eta = (T - t - 1) / rate;
            double avg_sentiment = history.back();
            cerr << "\r[Parallel] Step " << (t+1) << "/" << T << " (" << fixed << setprecision(1) << progress 
                 << "%) | Avg sentiment: " << setprecision(6) << avg_sentiment 
                 << " | Elapsed: " << setprecision(1) << elapsed << "s"
                 << " | ETA: " << setprecision(1) << eta << "s"
                 << " | Rate: " << setprecision(0) << rate << " steps/s" << flush;
        }
    }
    
    cerr << "\n";  // New line after progress
    double t_end = omp_get_wtime();
    double total_time = t_end - t_start;
    cerr << "[Parallel] Completed " << T << " steps in " << fixed << setprecision(2) << total_time 
         << "s (" << setprecision(0) << (T / total_time) << " steps/s)\n";

    // write history
    ofstream hf(history_file);
    if (!hf.is_open()) { cerr << "Cannot write history file\n"; return 5; }
    for (double x: history) hf << x << "\n";
    hf.close();

    // write final states
    ofstream of(out_states_file);
    if (!of.is_open()) { cerr << "Cannot write output states file\n"; return 6; }
    for (size_t i=0;i<N;i++) of << states[i] << "\n";
    of.close();

    // Already printed detailed info above
    return 0;
}
