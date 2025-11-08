// parallel_update.cpp
// Usage:
// cpp/bin/parallel_update graph.txt states.txt output_states.txt history.txt T [alpha] [threads]
#include <bits/stdc++.h>
#include <omp.h>
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

    vector<double> new_states(N, 0.0);
    vector<double> history;
    history.reserve(T);

    double t_start = omp_get_wtime();
    for (int t=0; t<T; ++t) {
        double global_sum = 0.0;
        #pragma omp parallel for reduction(+:global_sum) schedule(static)
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
        states.swap(new_states);
        history.push_back(global_sum / double(N));
    }
    double t_end = omp_get_wtime();

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

    cerr << "Ran T=" << T << " steps in " << (t_end - t_start) << " sec. Threads: " << omp_get_max_threads() << "\n";
    return 0;
}
