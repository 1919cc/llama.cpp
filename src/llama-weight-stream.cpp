#include "llama-weight-stream.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#define LLAMA_WS_SUPPORTED 1
#endif

namespace {

constexpr size_t WS_ALIGN = 4096;           // O_DIRECT offset / length / buffer alignment
[[maybe_unused]] constexpr size_t WS_MIN_TENSOR = 64 << 10;  // smaller tensors stay mmap-ed (they stay in the page cache)
[[maybe_unused]] constexpr size_t WS_MERGE_GAP = 1 << 20;    // read through gaps smaller than this between tensors
constexpr size_t WS_CHUNK = 4 << 20;        // bytes per pread

size_t align_down(size_t v) { return v / WS_ALIGN * WS_ALIGN; }
size_t align_up(size_t v)   { return (v + WS_ALIGN - 1) / WS_ALIGN * WS_ALIGN; }

struct ws_tensor {
    ggml_tensor * t;
    void *        orig;     // the mmap-ed address
    int           file;
    size_t        off;      // offset in the file
    size_t        slot_off; // offset in the slot buffer
};

struct ws_run {
    int    file;
    size_t lo, hi;   // aligned file range
    size_t slot_off;
};

struct ws_layer {
    std::vector<ws_tensor> tensors;
    std::vector<ws_run>    runs;
    size_t                 bytes = 0;
};

struct ws_map {
    uintptr_t   start, end;
    size_t      off;
    std::string path;
};

std::vector<ws_map> read_maps() {
    std::vector<ws_map> maps;
    FILE * f = fopen("/proc/self/maps", "r");
    if (!f) {
        return maps;
    }
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        unsigned long start, end, off;
        char perms[8], path[3072] = { 0 };
        unsigned int dmaj, dmin;
        unsigned long inode;
        if (sscanf(line, "%lx-%lx %7s %lx %x:%x %lu %3071[^\n]", &start, &end, perms, &off, &dmaj, &dmin, &inode, path) == 8 && path[0] == '/') {
            maps.push_back({ (uintptr_t) start, (uintptr_t) end, (size_t) off, path });
        }
    }
    fclose(f);
    return maps;
}

} // namespace

struct llama_weight_stream {
    struct op_data {
        llama_weight_stream * ws;
        int                   il;
    };

    std::vector<ws_layer>    layers; // by layer index, empty = not streamed
    std::vector<op_data>     ops;
    std::vector<int>         order;  // streamed layers, in execution order
    std::vector<std::string> paths;
    std::vector<int>         fds;

    void * slot_mem[2]     = { nullptr, nullptr };
    size_t slot_size       = 0;
    int    slot_layer[2]   = { -1, -1 };
    int    slot_pending[2] = { 0, 0 };   // chunks still being read into the slot
    int    redirected      = -1;         // layer whose tensors point into a slot

    struct chunk {
        int    slot;
        int    fd;
        size_t off;
        size_t len;
        char * dst;
    };

    std::mutex              mtx;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::deque<chunk>       queue;
    std::vector<std::thread> workers;
    bool                    stop = false;

    // stats
    size_t bytes_read = 0;
    double wait_ms    = 0;

    ~llama_weight_stream() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
            if (redirected >= 0) {
                restore(redirected);
            }
        }
        cv_work.notify_all();
        for (auto & t : workers) {
            t.join();
        }
#ifdef LLAMA_WS_SUPPORTED
        for (int fd : fds) {
            close(fd);
        }
#endif
        free(slot_mem[0]);
        free(slot_mem[1]);
        if (bytes_read > 0) {
            LLAMA_LOG_INFO("%s: streamed %.1f GiB, waited %.1f s for reads\n", __func__, bytes_read / 1073741824.0, wait_ms / 1000.0);
        }
    }

    void worker() {
#ifdef LLAMA_WS_SUPPORTED
        for (;;) {
            chunk c;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_work.wait(lock, [&] { return stop || !queue.empty(); });
                if (stop && queue.empty()) {
                    return;
                }
                c = queue.front();
                queue.pop_front();
            }
            size_t done = 0;
            while (done < c.len) {
                const ssize_t n = pread(c.fd, c.dst + done, c.len - done, (off_t) (c.off + done));
                if (n <= 0) {
                    break; // end of file (the last run is rounded up past it) or an error: the tensors end before
                }
                done += (size_t) n;
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                bytes_read += done;
                if (--slot_pending[c.slot] == 0) {
                    cv_done.notify_all();
                }
            }
        }
#endif
    }

    // with mtx held
    void start_load(int s, int il) {
        slot_layer[s] = il;
        char * base = (char *) slot_mem[s];
        for (const auto & r : layers[il].runs) {
            for (size_t o = r.lo; o < r.hi; o += WS_CHUNK) {
                queue.push_back({ s, fds[r.file], o, std::min(WS_CHUNK, r.hi - o), base + r.slot_off + (o - r.lo) });
                slot_pending[s]++;
            }
        }
        cv_work.notify_all();
    }

    void redirect(int il, int s) {
        for (auto & t : layers[il].tensors) {
            t.t->data = (char *) slot_mem[s] + t.slot_off;
        }
        redirected = il;
    }

    void restore(int il) {
        for (auto & t : layers[il].tensors) {
            t.t->data = t.orig;
        }
        redirected = -1;
    }

    int slot_of(int il) const {
        return slot_layer[0] == il ? 0 : slot_layer[1] == il ? 1 : -1;
    }

    void enter_layer(int il) {
        const int64_t t0 = ggml_time_us();
        std::unique_lock<std::mutex> lock(mtx);
        if (redirected >= 0 && redirected != il) {
            restore(redirected);
        }
        int s = slot_of(il);
        if (s < 0) {
            // not loaded or loading: take a slot that is not busy (wait for one)
            cv_done.wait(lock, [&] { return slot_pending[0] == 0 || slot_pending[1] == 0; });
            s = slot_pending[0] == 0 ? 0 : 1;
            start_load(s, il);
        }
        cv_done.wait(lock, [&] { return slot_pending[s] == 0; });
        redirect(il, s);

        // read the next streamed layer (the first one again after the last) into the other slot
        const size_t pos  = std::find(order.begin(), order.end(), il) - order.begin();
        const int    next = order[(pos + 1) % order.size()];
        const int    o    = 1 - s;
        if (next != il && slot_layer[o] != next) {
            cv_done.wait(lock, [&] { return slot_pending[o] == 0; });
            start_load(o, next);
        }
        wait_ms += (ggml_time_us() - t0) / 1000.0;
    }
};

namespace {

std::mutex                                                                   g_ws_mutex;
std::map<const llama_model *, std::unique_ptr<llama_weight_stream>>         g_ws;
std::map<const llama_model *, bool>                                          g_ws_failed;

std::unique_ptr<llama_weight_stream> ws_create(const llama_model & model) {
#ifndef LLAMA_WS_SUPPORTED
    GGML_UNUSED(model);
    return nullptr;
#else
    const auto maps = read_maps();
    auto ws = std::make_unique<llama_weight_stream>();
    const int n_layer = (int) model.hparams.n_layer();
    ws->layers.resize(n_layer);

    std::map<std::string, int> file_idx;
    for (const auto & it : model.tensors_by_name) {
        ggml_tensor * t = it.second;
        int il = -1;
        if (sscanf(it.first.c_str(), "blk.%d.", &il) != 1 || il < 0 || il >= n_layer) {
            continue;
        }
        if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer) || ggml_nbytes(t) < WS_MIN_TENSOR) {
            continue;
        }
        const uintptr_t a = (uintptr_t) t->data;
        auto m = std::find_if(maps.begin(), maps.end(), [&](const ws_map & mp) { return a >= mp.start && a + ggml_nbytes(t) <= mp.end; });
        if (m == maps.end()) {
            continue; // not file-backed (copied into anonymous memory)
        }
        auto fi = file_idx.find(m->path);
        if (fi == file_idx.end()) {
            fi = file_idx.emplace(m->path, (int) ws->paths.size()).first;
            ws->paths.push_back(m->path);
        }
        ws->layers[il].tensors.push_back({ t, t->data, fi->second, m->off + (a - m->start), 0 });
    }

    for (const auto & p : ws->paths) {
        const int fd = open(p.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) {
            LLAMA_LOG_WARN("%s: cannot open %s with O_DIRECT, weight streaming disabled\n", __func__, p.c_str());
            for (int f : ws->fds) {
                close(f);
            }
            ws->fds.clear();
            return nullptr;
        }
        ws->fds.push_back(fd);
    }

    for (int il = 0; il < n_layer; ++il) {
        auto & L = ws->layers[il];
        if (L.tensors.empty()) {
            continue;
        }
        std::sort(L.tensors.begin(), L.tensors.end(), [](const ws_tensor & a, const ws_tensor & b) {
            return a.file != b.file ? a.file < b.file : a.off < b.off;
        });
        for (auto & t : L.tensors) {
            const size_t lo = align_down(t.off);
            const size_t hi = align_up(t.off + ggml_nbytes(t.t));
            if (!L.runs.empty() && L.runs.back().file == t.file && lo <= L.runs.back().hi + WS_MERGE_GAP) {
                L.runs.back().hi = std::max(L.runs.back().hi, hi);
            } else {
                const size_t slot_off = L.runs.empty() ? 0 : L.runs.back().slot_off + (L.runs.back().hi - L.runs.back().lo);
                L.runs.push_back({ t.file, lo, hi, slot_off });
            }
            t.slot_off = L.runs.back().slot_off + (t.off - L.runs.back().lo);
        }
        L.bytes = L.runs.back().slot_off + (L.runs.back().hi - L.runs.back().lo);
        ws->slot_size = std::max(ws->slot_size, L.bytes);
        ws->order.push_back(il);
    }
    if (ws->order.empty()) {
        return nullptr;
    }

    for (int s = 0; s < 2; ++s) {
        if (posix_memalign(&ws->slot_mem[s], WS_ALIGN, ws->slot_size) != 0) {
            LLAMA_LOG_WARN("%s: cannot allocate %zu MiB slot buffers, weight streaming disabled\n", __func__, ws->slot_size >> 20);
            return nullptr;
        }
    }

    ws->ops.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        ws->ops[il] = { ws.get(), il };
    }

    const char * nt = getenv("LLAMA_WEIGHT_STREAM_THREADS");
    const int n_threads = nt ? std::max(1, atoi(nt)) : 4;
    for (int i = 0; i < n_threads; ++i) {
        ws->workers.emplace_back([p = ws.get()] { p->worker(); });
    }

    LLAMA_LOG_INFO("%s: streaming %zu layers from %zu file(s), 2 x %.1f MiB buffers, %d I/O threads\n",
                   __func__, ws->order.size(), ws->paths.size(), ws->slot_size / 1048576.0, n_threads);
    return ws;
#endif
}

} // namespace

llama_weight_stream * llama_weight_stream_get(const llama_model & model) {
    const char * e = getenv("LLAMA_WEIGHT_STREAM");
    if (!e || atoi(e) <= 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_ws_mutex);
    auto it = g_ws.find(&model);
    if (it != g_ws.end()) {
        return it->second.get();
    }
    if (g_ws_failed.count(&model)) {
        return nullptr;
    }
    auto ws = ws_create(model);
    if (!ws) {
        g_ws_failed[&model] = true;
        return nullptr;
    }
    return g_ws.emplace(&model, std::move(ws)).first->second.get();
}

void llama_weight_stream_release(const llama_model & model) {
    std::lock_guard<std::mutex> lock(g_ws_mutex);
    g_ws.erase(&model);
    g_ws_failed.erase(&model);
}

void * llama_weight_stream_layer(llama_weight_stream * ws, int il) {
    if (!ws || il < 0 || il >= (int) ws->layers.size() || ws->layers[il].tensors.empty()) {
        return nullptr;
    }
    return &ws->ops[il];
}

void llama_weight_stream_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * od = (llama_weight_stream::op_data *) userdata;
    od->ws->enter_layer(od->il);
}

namespace {

// one pool for the whole process: the page cache is shared, so are the reads that fill it
class llama_page_filler {
public:
    static llama_page_filler & get() {
        // never destroyed: the detached workers wait on cv_work until the process exits
        static llama_page_filler * f = new llama_page_filler();
        return *f;
    }

    void fill(const std::vector<std::pair<const void *, size_t>> & ranges, bool wait) {
#ifdef LLAMA_WS_SUPPORTED
        auto req = std::make_shared<request>();
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (const auto & r : ranges) {
                const uintptr_t a = (uintptr_t) r.first;
                const ws_map * m = find_map(a, r.second);
                if (!m) {
                    continue;
                }
                const int fd = fd_of(m->path);
                if (fd < 0) {
                    continue;
                }
                const size_t off = m->off + (a - m->start);
                const size_t lo  = align_down(off);
                const size_t hi  = align_up(off + r.second);
                for (size_t o = lo; o < hi; o += WS_CHUNK) {
                    queue.push_back({ req, fd, o, std::min(WS_CHUNK, hi - o) });
                    req->pending++;
                }
            }
            if (req->pending == 0) {
                return;
            }
        }
        cv_work.notify_all();
        if (wait) {
            std::unique_lock<std::mutex> lock(mtx);
            cv_done.wait(lock, [&] { return req->pending == 0; });
        }
#else
        GGML_UNUSED(ranges);
        GGML_UNUSED(wait);
#endif
    }

private:
    struct request {
        int pending = 0;
    };
    struct chunk {
        std::shared_ptr<request> req;
        int    fd;
        size_t off;
        size_t len;
        char * dst = nullptr; // read into this buffer; nullptr: into a scratch buffer, for the page cache only
    };

public:
    // reads [src, src + len) of mmap-ed files into dst buffers (buffered: page cache hits are served from RAM)
    bool read_into(const std::vector<llama_read_req> & reqs) {
#ifdef LLAMA_WS_SUPPORTED
        auto req = std::make_shared<request>();
        {
            std::lock_guard<std::mutex> lock(mtx);
            // resolve all requests first: nothing may be queued into the caller's buffers when we return false
            std::vector<std::pair<int, size_t>> src(reqs.size());
            for (size_t i = 0; i < reqs.size(); ++i) {
                const uintptr_t a = (uintptr_t) reqs[i].src;
                const ws_map * m = find_map(a, reqs[i].len);
                const int fd = m ? fd_of(m->path) : -1;
                if (fd < 0) {
                    return false;
                }
                src[i] = { fd, m->off + (a - m->start) };
            }
            for (size_t i = 0; i < reqs.size(); ++i) {
                const auto & r = reqs[i];
                for (size_t o = 0; o < r.len; o += WS_CHUNK) {
                    queue.push_back({ req, src[i].first, src[i].second + o, std::min(WS_CHUNK, r.len - o), (char *) r.dst + o });
                    req->pending++;
                }
            }
            if (req->pending == 0) {
                return true;
            }
        }
        cv_work.notify_all();
        std::unique_lock<std::mutex> lock(mtx);
        cv_done.wait(lock, [&] { return req->pending == 0; });
        return true;
#else
        GGML_UNUSED(reqs);
        return false;
#endif
    }

private:

    llama_page_filler() {
        const char * nt = getenv("LLAMA_FILL_THREADS");
        const int n = nt ? std::max(1, atoi(nt)) : 6;
        for (int i = 0; i < n; ++i) {
            std::thread([this] { run(); }).detach();
        }
    }

    // with mtx held; re-reads /proc/self/maps when an address is not in the cached copy (a model was loaded since)
    const ws_map * find_map(uintptr_t a, size_t n) {
        for (int pass = 0; pass < 2; ++pass) {
            for (const auto & m : maps) {
                if (a >= m.start && a + n <= m.end) {
                    return &m;
                }
            }
            if (pass == 0) {
                maps = read_maps();
            }
        }
        return nullptr;
    }

    int fd_of(const std::string & path) {
#ifdef LLAMA_WS_SUPPORTED
        auto it = fds.find(path);
        if (it == fds.end()) {
            it = fds.emplace(path, open(path.c_str(), O_RDONLY | O_CLOEXEC)).first;
        }
        return it->second;
#else
        GGML_UNUSED(path);
        return -1;
#endif
    }

    [[noreturn]] void run() {
        std::vector<char> buf(WS_CHUNK);
        for (;;) {
            chunk c;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_work.wait(lock, [&] { return !queue.empty(); });
                c = std::move(queue.front());
                queue.pop_front();
            }
#ifdef LLAMA_WS_SUPPORTED
            // a buffered read leaves the data in the page cache, which is all we want from it
            size_t done = 0;
            while (done < c.len) {
                char * to = c.dst ? c.dst + done : buf.data();
                const ssize_t r = pread(c.fd, to, c.len - done, (off_t) (c.off + done));
                if (r <= 0) {
                    break;
                }
                done += (size_t) r;
            }
#endif
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (--c.req->pending == 0) {
                    cv_done.notify_all();
                }
            }
        }
    }

    std::mutex                 mtx;
    std::condition_variable    cv_work;
    std::condition_variable    cv_done;
    std::deque<chunk>          queue;
    std::vector<ws_map>        maps;
    std::map<std::string, int> fds;
};

} // namespace

bool llama_read_into(const std::vector<llama_read_req> & reqs) {
    return reqs.empty() || llama_page_filler::get().read_into(reqs);
}

void llama_page_fill(const std::vector<std::pair<const void *, size_t>> & ranges, bool wait) {
    if (!ranges.empty()) {
        llama_page_filler::get().fill(ranges, wait);
    }
}
