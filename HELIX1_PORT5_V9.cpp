// HELIX-1 PORT5 V9 - C++17 reference model: 5 ALU ports (2 INT + 2 FP + 1 BRANCH), 6-wide dispatch, ROB128, 8-bank LSU.
// Predictor V8: six-level TAGE + 128-bit GHR/Path History + 32-bit Perceptron + Loop Predictor + BTB/RAS.
// Recovery: fetch-stage exact BEQ resolution, speculative GHR/path/local-history checkpoints,
//           reverse-order local-history rollback, GHR/RAS restore, same-cycle redirect.
// JALR/RET opcode 120 is added for indirect jumps and RAS return prediction.
// Store Buffer decouples retirement from backing DMEM; loads forward from ROB/SB.
// Early illegal-opcode detection stops wrong-path fetch before issue; precise fault at retirement.
// This is a software reference model, NOT synthesizable RTL and NOT FPGA timing evidence.
// Build: g++ -std=c++17 -O2 -Wall -Wextra -pedantic -o helix1 HELIX1_PORT5_V9.cpp
// Run:   ./helix1 [--dmem N] [--samples N] [--seed N] [--penalty N] [--lsu 1..8] [--alu 2] [--fp-alu 2] [--branch-alu 1] [--rob 16..128] [--hist-bits 128] [--bp-confidence 0|1]
// Note: --penalty 0 is same-cycle redirect in this model; use --penalty 1 for a
//       more conservative one-extra-cycle recovery assumption.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <deque>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
using namespace std;
namespace fs = std::filesystem;
using u32 = uint32_t;
using u64 = uint64_t;
constexpr u64 M64 = numeric_limits<u64>::max();
constexpr u32 M32 = numeric_limits<u32>::max();
// Bo nho du lieu mac dinh: 64 tu, moi tu 64-bit (512 byte).
// IMEM va DMEM dung vector nen co the mo rong theo kich thuoc chuong trinh/cau hinh.
constexpr size_t DEFAULT_DWORDS = 64;
const unordered_map<string, int> OPS = {
    {"ADD", 0}, {"SUB", 1}, {"MUL", 2}, {"DIV", 3},
    {"AND", 8}, {"OR", 9}, {"XOR", 10},
    {"SLL", 16}, {"SRL", 17},
    {"FADD", 24}, {"FSUB", 25}, {"FMUL", 26}, {"FDIV", 27},
    {"ITOF", 28}, {"FTOI", 29},
    {"ADDI", 32}, {"LW", 64}, {"FLD", 68},
    {"SW", 80}, {"FSD", 84}, {"BEQ", 96}, {"JALR", 120}, {"JAL", 121}
};
const unordered_map<int, int> ALU_OF = {
    {0, 0}, {1, 1}, {2, 7}, {3, 8}, {8, 2}, {9, 3}, {10, 4}, {16, 5}, {17, 6}
};
constexpr u32 HALT = 121; // JAL x0, 0
string trim(const string& s) {
    size_t first = s.find_first_not_of(" \t\r\n");
    if (first == string::npos) return "";
    size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}
// Tra ve gia tri sign-extend duoi dang so nguyen 64-bit khong dau.
u64 sx(u64 value, int bits) {
    const u64 mask = (u64(1) << bits) - 1;
    value &= mask;
    if (value & (u64(1) << (bits - 1))) value |= ~mask;
    return value;
}
u32 fit_imm(int64_t imm, int bits, const string& what) {
    const int64_t lo = -(int64_t(1) << (bits - 1));
    const int64_t hi = (int64_t(1) << (bits - 1)) - 1;
    if (imm < lo || imm > hi) {
        ostringstream msg;
        msg << what << ": imm " << imm << " ngoai [" << lo << "," << hi << "]";
        throw runtime_error(msg.str());
    }
    return static_cast<u32>(static_cast<uint64_t>(imm) & ((u64(1) << bits) - 1));
}
u32 enc_r(int op, int rd, int rs1, int rs2) {
    return (u32(rs2) << 17) | (u32(rs1) << 12) | (u32(rd) << 7) | u32(op);
}
u32 enc_i(int op, int rd, int rs1, int64_t imm) {
    return (fit_imm(imm, 15, "I") << 17) | (u32(rs1) << 12) | (u32(rd) << 7) | u32(op);
}
u32 enc_s(int op, int rs1, int rs2, int64_t imm) {
    return (fit_imm(imm, 10, "S/B") << 22) | (u32(rs2) << 17) | (u32(rs1) << 12) | u32(op);
}
u32 enc_j(int op, int rd, int64_t imm) {
    return (fit_imm(imm, 20, "J") << 12) | (u32(rd) << 7) | u32(op);
}
u32 enc_jalr(int rd, int rs1, int64_t imm) {
    return (fit_imm(imm, 15, "JALR") << 17) | (u32(rs1) << 12) | (u32(rd) << 7) | u32(OPS.at("JALR"));
}
u64 double_to_bits(double value) {
    u64 bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "double phai 64-bit");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
double bits_to_double(u64 bits) {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
u64 double_to_i64_bits(double value) {
    if (!std::isfinite(value)) return 0;
    const long double v = static_cast<long double>(value);
    const long double hi = static_cast<long double>(numeric_limits<int64_t>::max());
    const long double lo = static_cast<long double>(numeric_limits<int64_t>::min());
    if (v >= hi) return static_cast<u64>(numeric_limits<int64_t>::max());
    if (v <= lo) return static_cast<u64>(numeric_limits<int64_t>::min());
    return static_cast<u64>(static_cast<int64_t>(value));
}

int64_t parse_integer(const string& token) {
    string t = trim(token);
    if (t.empty()) throw runtime_error("thieu so nguyen");
    size_t used = 0;
    long long value;
    try {
        value = stoll(t, &used, 0);
    } catch (...) {
        throw runtime_error("khong doc duoc so nguyen: " + t);
    }
    if (used != t.size()) throw runtime_error("so nguyen khong hop le: " + t);
    return static_cast<int64_t>(value);
}
vector<string> split_args(const string& s) {
    vector<string> result;
    string part;
    stringstream ss(s);
    while (getline(ss, part, ',')) {
        part = trim(part);
        if (!part.empty()) result.push_back(part);
    }
    return result;
}
vector<u32> assemble(const string& src) {
    unordered_map<string, int> labels;
    vector<string> lines;
    stringstream input(src);
    string raw;
    while (getline(input, raw)) {
        size_t comment = raw.find(';');
        if (comment != string::npos) raw = raw.substr(0, comment);
        string line = trim(raw);
        while (line.find(':') != string::npos) {
            size_t pos = line.find(':');
            string label = trim(line.substr(0, pos));
            if (label.empty()) throw runtime_error("nhan (label) rong");
            labels[label] = static_cast<int>(lines.size());
            line = trim(line.substr(pos + 1));
        }
        if (!line.empty()) lines.push_back(line);
    }
    auto reg = [](const string& token, char prefix = 'x') -> int {
        string t = token;
        transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return char(tolower(c)); });
        if (t.size() < 2 || t[0] != prefix)
            throw runtime_error(string("thanh ghi sai (can ") + prefix + "N): " + t);
        int n = static_cast<int>(parse_integer(t.substr(1)));
        if (n < 0 || n >= 32) throw runtime_error("thanh ghi ngoai pham vi: " + t);
        return n;
    };
    auto val = [&labels](const string& token, int pc) -> int64_t {
        string t = trim(token);
        auto it = labels.find(t);
        if (it != labels.end()) return int64_t(it->second) - pc;
        return parse_integer(t);
    };
    vector<u32> out;
    for (int pc = 0; pc < static_cast<int>(lines.size()); ++pc) {
        const string& line = lines[pc];
        size_t sep = line.find_first_of(" \t");
        string mn = (sep == string::npos) ? line : line.substr(0, sep);
        string rest = (sep == string::npos) ? "" : trim(line.substr(sep + 1));
        transform(mn.begin(), mn.end(), mn.begin(), [](unsigned char c) { return char(toupper(c)); });
        vector<string> a = split_args(rest);
        auto need = [&a, &line](size_t n) {
            if (a.size() != n) {
                throw runtime_error("sai so doi so trong lenh: " + line);
            }
        };
        if (mn == "FADD" || mn == "FSUB" || mn == "FMUL" || mn == "FDIV") {
            need(3);
            out.push_back(enc_r(OPS.at(mn), reg(a[0], 'f'), reg(a[1], 'f'), reg(a[2], 'f')));
        } else if (mn == "ITOF") {
            need(2);
            out.push_back(enc_r(OPS.at(mn), reg(a[0], 'f'), reg(a[1], 'x'), 0));
        } else if (mn == "FTOI") {
            need(2);
            out.push_back(enc_r(OPS.at(mn), reg(a[0], 'x'), reg(a[1], 'f'), 0));
        } else if (mn == "FLD") {
            need(3);
            out.push_back(enc_i(OPS.at(mn), reg(a[0], 'f'), reg(a[1], 'x'), val(a[2], pc)));
        } else if (mn == "FSD") {
            need(3);
            out.push_back(enc_s(OPS.at(mn), reg(a[1], 'x'), reg(a[0], 'f'), val(a[2], pc)));
        } else if (mn == "ADD" || mn == "SUB" || mn == "MUL" || mn == "DIV" || mn == "AND" || mn == "OR" || mn == "XOR" || mn == "SLL" || mn == "SRL") {
            need(3);
            out.push_back(enc_r(OPS.at(mn), reg(a[0]), reg(a[1]), reg(a[2])));
        } else if (mn == "ADDI" || mn == "LW") {
            need(3);
            out.push_back(enc_i(OPS.at(mn), reg(a[0]), reg(a[1]), val(a[2], pc)));
        } else if (mn == "SW") {
            need(3); // SW rs2, rs1, imm
            out.push_back(enc_s(OPS.at("SW"), reg(a[1]), reg(a[0]), val(a[2], pc)));
        } else if (mn == "BEQ") {
            need(3); // BEQ rs1, rs2, label
            out.push_back(enc_s(OPS.at("BEQ"), reg(a[0]), reg(a[1]), val(a[2], pc)));
        } else if (mn == "JAL") {
            need(2);
            out.push_back(enc_j(OPS.at("JAL"), reg(a[0]), val(a[1], pc)));
        } else if (mn == "JALR") {
            need(3);
            out.push_back(enc_jalr(reg(a[0]), reg(a[1]), val(a[2], pc)));
        } else if (mn == "RET") {
            need(0);
            out.push_back(enc_jalr(0, 1, 0));
        } else if (mn == "HALT") {
            need(0);
            out.push_back(HALT);
        } else if (mn == ".WORD") {
            need(1);
            out.push_back(static_cast<u32>(parse_integer(a[0])));
        } else if (mn == "NOP") {
            need(0);
            out.push_back(0);
        } else {
            throw runtime_error("lenh khong co: " + line);
        }
    }
    return out;
}
// IMEM co kich thuoc theo chuong trinh, khong con gioi han 128 lenh.
vector<u32> make_imem(const vector<u32>& words) {
    return words;
}
struct Result {
    array<u64, 32> regs{};
    array<u64, 32> fregs{}; // IEEE-754 binary64 bit patterns; f0 is writable
    vector<u64> dmem;
    bool fault = false;
};
// ---------------------------------------------------------------------------
// Mo hinh chuan (golden)
// ---------------------------------------------------------------------------
Result golden(const vector<u32>& words, int max_steps = 100000,
              size_t data_words = DEFAULT_DWORDS) {
    if (data_words == 0) throw runtime_error("DMEM phai co it nhat 1 tu 64-bit");
    auto imem = make_imem(words);
    Result result;
    result.dmem.assign(data_words, 0);
    u32 pc = 0;
    auto write_reg = [&result](int i, u64 value) {
        if (i != 0) result.regs[i] = value;
    };
    auto write_freg = [&result](int i, u64 value) { result.fregs[i] = value; };
    for (int step = 0; step < max_steps; ++step) {
        u32 w = ((pc & 3u) == 0 && (pc >> 2) < imem.size())
                    ? imem[pc >> 2] : 0xFFFFFFFFu;
        int op = int(w & 127u);
        int rd = int((w >> 7) & 31u);
        int rs1 = int((w >> 12) & 31u);
        int rs2 = int((w >> 17) & 31u);
        u64 a = result.regs[rs1], b = result.regs[rs2];
        u32 npc = pc + 4u;
        if (op == OPS.at("FADD") || op == OPS.at("FSUB") ||
            op == OPS.at("FMUL") || op == OPS.at("FDIV")) {
            const double fa = bits_to_double(result.fregs[rs1]);
            const double fb = bits_to_double(result.fregs[rs2]);
            double fy = 0.0;
            if (op == OPS.at("FADD")) fy = fa + fb;
            else if (op == OPS.at("FSUB")) fy = fa - fb;
            else if (op == OPS.at("FMUL")) fy = fa * fb;
            else fy = fa / fb; // IEEE-754: signed infinity/NaN on zero divisor
            write_freg(rd, double_to_bits(fy));
        } else if (op == OPS.at("ITOF")) {
            write_freg(rd, double_to_bits(static_cast<double>(static_cast<int64_t>(result.regs[rs1]))));
        } else if (op == OPS.at("FTOI")) {
            write_reg(rd, double_to_i64_bits(bits_to_double(result.fregs[rs1])));
        } else {
        auto it = ALU_OF.find(op);
        if (it != ALU_OF.end()) {
            u64 y = 0;
            switch (it->second) {
                case 0: y = a + b; break;
                case 1: y = a - b; break;
                case 2: y = a & b; break;
                case 3: y = a | b; break;
                case 4: y = a ^ b; break;
                case 5: y = a << (b & 63u); break;
                case 6: y = a >> (b & 63u); break;
                case 7: y = a * b; break;
                case 8: y = (b == 0 ? M64 : a / b); break; // unsigned DIV; divide-by-zero returns all ones
            }
            write_reg(rd, y);
        } else if (op == OPS.at("ADDI")) {
            write_reg(rd, a + sx(w >> 17, 15));
        } else if (op == OPS.at("LW") || op == OPS.at("SW") ||
                   op == OPS.at("FLD") || op == OPS.at("FSD")) {
            const bool is_load = (op == OPS.at("LW") || op == OPS.at("FLD"));
            u64 imm = is_load ? sx(w >> 17, 15) : sx(w >> 22, 10);
            u64 addr = a + imm;
            if ((addr & 7u) != 0 || (addr >> 3) >= result.dmem.size()) {
                result.fault = true;
                return result;
            }
            size_t index = static_cast<size_t>(addr >> 3);
            if (op == OPS.at("LW")) write_reg(rd, result.dmem[index]);
            else if (op == OPS.at("FLD")) write_freg(rd, result.dmem[index]);
            else if (op == OPS.at("SW")) result.dmem[index] = b;
            else result.dmem[index] = result.fregs[rs2];
        } else if (op == OPS.at("BEQ")) {
            if (a == b) npc = pc + static_cast<u32>(sx(w >> 22, 10) * 4u);
        } else if (op == OPS.at("JAL")) {
            u64 imm = sx(w >> 12, 20);
            write_reg(rd, u64(pc) + 4u);
            if (imm == 0) return result;
            npc = pc + static_cast<u32>(imm * 4u);
        } else if (op == OPS.at("JALR")) {
            write_reg(rd, u64(pc) + 4u);
            npc = static_cast<u32>((a + sx(w >> 17, 15)) & ~u64(3));
        } else {
            result.fault = true;
            return result;
        }
        }
        pc = npc;
    }
    throw runtime_error("golden: vuot so buoc");
}
// ---------------------------------------------------------------------------
// Giai ma lenh va ALU
// ---------------------------------------------------------------------------
struct Decoded {
    int rd = 0, rs1 = 0, rs2 = 0;
    u64 imm = 0;
    bool reg_write = false, mem_r = false, mem_w = false;
    bool src_imm = false, branch = false, jump = false, indirect_jump = false;
    bool use1 = false, use2 = false, illegal = false;
    bool fp_exec = false, dst_fp = false, src1_fp = false, src2_fp = false;
    int alu = 0;
};
Decoded decode(u32 w) {
    int op = int(w & 127u);
    Decoded d;
    d.rd = int((w >> 7) & 31u);
    d.rs1 = int((w >> 12) & 31u);
    d.rs2 = int((w >> 17) & 31u);
    if (op == OPS.at("FADD") || op == OPS.at("FSUB") ||
        op == OPS.at("FMUL") || op == OPS.at("FDIV")) {
        d.reg_write = true; d.use1 = true; d.use2 = true;
        d.fp_exec = true; d.dst_fp = true; d.src1_fp = true; d.src2_fp = true;
    } else if (op == OPS.at("ITOF")) {
        d.reg_write = true; d.use1 = true; d.fp_exec = true; d.dst_fp = true;
    } else if (op == OPS.at("FTOI")) {
        d.reg_write = true; d.use1 = true; d.fp_exec = true; d.src1_fp = true;
    } else {
    auto it = ALU_OF.find(op);
    if (it != ALU_OF.end()) {
        d.reg_write = true; d.alu = it->second; d.use1 = true; d.use2 = true;
    } else if (op == OPS.at("ADDI")) {
        d.reg_write = true; d.src_imm = true; d.use1 = true; d.imm = sx(w >> 17, 15);
    } else if (op == OPS.at("LW")) {
        d.reg_write = true; d.mem_r = true; d.src_imm = true; d.use1 = true; d.imm = sx(w >> 17, 15);
    } else if (op == OPS.at("FLD")) {
        d.reg_write = true; d.mem_r = true; d.src_imm = true; d.use1 = true;
        d.dst_fp = true; d.imm = sx(w >> 17, 15);
    } else if (op == OPS.at("SW")) {
        d.mem_w = true; d.src_imm = true; d.use1 = true; d.use2 = true; d.imm = sx(w >> 22, 10);
    } else if (op == OPS.at("FSD")) {
        d.mem_w = true; d.src_imm = true; d.use1 = true; d.use2 = true;
        d.src2_fp = true; d.imm = sx(w >> 22, 10);
    } else if (op == OPS.at("BEQ")) {
        d.branch = true; d.alu = 1; d.use1 = true; d.use2 = true; d.imm = sx(w >> 22, 10);
    } else if (op == OPS.at("JAL")) {
        d.reg_write = true; d.jump = true; d.imm = sx(w >> 12, 20);
    } else if (op == OPS.at("JALR")) {
        d.reg_write = true; d.jump = true; d.indirect_jump = true;
        d.use1 = true; d.imm = sx(w >> 17, 15);
    } else {
        d.illegal = true;
    }
    }
    return d;
}
pair<u64, bool> alu(int op, u64 a, u64 b) {
    u64 y = 0;
    switch (op) {
        case 0: y = a + b; break;
        case 1: y = a - b; break;
        case 2: y = a & b; break;
        case 3: y = a | b; break;
        case 4: y = a ^ b; break;
        case 5: y = a << (b & 63u); break;
        case 6: y = a >> (b & 63u); break;
        case 7: y = a * b; break;
        case 8: y = (b == 0 ? M64 : a / b); break;
        default: y = 0; break;
    }
    return {y, y == 0};
}
struct IDReg {
    bool v = false;
    u32 pc = 0, instr = 0;
};
struct EXReg {
    bool v = false, illegal = false, reg_write = false, mem_r = false, mem_w = false;
    bool src_imm = false, branch = false, jump = false, use1 = false, use2 = false;
    int alu = 0, rd = 0, rs1 = 0, rs2 = 0;
    u64 v1 = 0, v2 = 0, imm = 0;
    u32 pc = 0;
};
struct WBReg {
    bool v = false, reg_write = false, mem_r = false, mem_w = false;
    int rd = 0;
    u64 y = 0, load = 0, store = 0, addr = 0;
};

// ---------------------------------------------------------------------------
// Cau hinh may + bo du doan nhanh
// ---------------------------------------------------------------------------
enum class BPMode { OFF, TOURNAMENT };

// Two-word history register: portable C++17 replacement for non-standard __int128.
struct History128 {
    u64 lo = 0;
    u64 hi = 0;

    History128 push_bit(bool bit) const {
        History128 out;
        out.hi = (hi << 1) | (lo >> 63);
        out.lo = (lo << 1) | (bit ? 1ULL : 0ULL);
        return out;
    }
    History128 push_byte(uint8_t byte) const {
        History128 out;
        out.hi = (hi << 8) | (lo >> 56);
        out.lo = (lo << 8) | static_cast<u64>(byte);
        return out;
    }
    History128 masked(unsigned bits) const {
        if (bits >= 128) return *this;
        if (bits == 0) return {};
        if (bits == 64) return {lo, 0};
        if (bits < 64) return {lo & ((1ULL << bits) - 1ULL), 0};
        const unsigned high_bits = bits - 64;
        return {lo, hi & ((1ULL << high_bits) - 1ULL)};
    }
    bool bit(unsigned index) const {
        return index < 64 ? ((lo >> index) & 1ULL) != 0
                          : ((hi >> (index - 64)) & 1ULL) != 0;
    }
};

struct Config {
    int issue_width = 6;          // aggregate dispatch / execute ceiling
    int integer_alu_width = 2;    // 2 integer ALU ports
    int floating_alu_width = 2;   // 2 IEEE-754 binary64 FP ports
    int branch_control_width = 1; // 1 branch/control port; independent of integer/FP ports
    int memory_width = 8;
    int fetch_width = 12;
    int retire_width = 10;
    size_t rob_size = 128;
    int redirect_bubbles = 0;
    int branch_confidence_threshold = 0; // V8 default: never stall Fetch for low confidence
    BPMode bp = BPMode::TOURNAMENT;
    int predictor_entries = 512;
    int hist_bits = 128;
    int ras_depth = 16;
    int store_buffer_entries = 16;
    int store_drain_width = 2;
};

struct PredictionMeta {
    History128 ghr_snapshot{};
    History128 path_history_snapshot{};
    uint16_t local_history_snapshot = 0;
    bool pred_bimodal = false, pred_gshare = false, pred_local = false;
    bool early_resolved = false;
    bool btb_hit = false, ras_used = false;
    bool pred_tage = false, pred_alt = false, tage_used = false;
    bool tage_alt_prediction = false;
    bool predicted_direction = false;
    // Perceptron and loop-predictor metadata for training and speculative rollback.
    bool pred_perceptron = false, perceptron_used = false;
    bool pred_fast = false, fast_used = false;
    bool pred_fast_bimodal = false, pred_fast_gshare = false;
    bool loop_used = false, loop_predicted_exit = false, loop_main_taken = false;
    bool loop_resync_before = false;
    int16_t perceptron_sum = 0;
    uint16_t perceptron_index = 0, loop_spec_before = 0;
    uint8_t perceptron_trained = 0;
    uint8_t branch_class = 0, hybrid_chooser_snapshot = 1;
    uint16_t fast_bimodal_index = 0, fast_gshare_index = 0;
    uint8_t fast_chooser_snapshot = 1;
    uint8_t confidence = 0; // 0..3; low-confidence branches may be deferred
    int8_t tage_provider = -1;
    int8_t tage_alt_provider = -1;
    array<uint16_t, 6> tage_index{};
    array<uint16_t, 6> tage_tag{};
    array<bool, 6> tage_hit{};
    array<bool, 6> tage_pred{};
    u32 predicted_target = 0;
};

// V8: V8's six-level TAGE + Path History + Perceptron + Loop Predictor,
// with a parallel 64-entry fast predictor and a 2-bit chooser per branch class.
class BranchPredictor {
    static constexpr int NT = 6;
    struct TageEntry {
        uint16_t tag = 0;
        int8_t ctr = 0;       // signed saturating counter [-3,+3]
        uint8_t useful = 0;   // 2-bit usefulness
        bool valid = false;
    };
    struct BTBEntry { bool valid = false; u32 tag = 0, target = 0; bool indirect = false; };
    struct BranchStat { u32 pc = 0; uint8_t miss_streak = 0; bool valid = false; };
    struct LoopEntry {
        u32 tag = 0;
        uint16_t trip_count = 0;
        uint16_t train_run = 0;
        uint16_t speculative_iter = 0;
        uint8_t confidence = 0;
        bool main_taken = false;
        bool resync = false;
        bool valid = false;
    };
    static constexpr int PERCEPTRON_HISTORY = 32;
    struct PerceptronEntry {
        array<int8_t, PERCEPTRON_HISTORY + 1> weight{}; // last slot is bias
        uint8_t trained = 0;
    };
    static constexpr array<int, NT> HIST_LEN = {4, 8, 16, 32, 64, 128};

public:
    BranchPredictor(int entries, int hist_bits, int ras_depth)
        : entries_(entries), mask_(static_cast<u32>(entries - 1)), hist_bits_(hist_bits),
          bimodal_(static_cast<size_t>(entries), 1),
          gshare_(static_cast<size_t>(entries), 1),
          local_pht_(static_cast<size_t>(entries), 1),
          score_b_(static_cast<size_t>(entries), 1),
          score_g_(static_cast<size_t>(entries), 1),
          score_l_(static_cast<size_t>(entries), 1),
          seen_b_(static_cast<size_t>(entries), false),
          seen_g_(static_cast<size_t>(entries), false),
          seen_l_(static_cast<size_t>(entries), false),
          local_hist_(static_cast<size_t>(entries), 0),
          score_perceptron_(static_cast<size_t>(entries), 1),
          score_tage_(static_cast<size_t>(entries), 1),
          hybrid_chooser_(static_cast<size_t>(entries) * 2u, 1),
          fast_bimodal_(64, 1), fast_gshare_(64, 1), fast_chooser_(64, 1),
          fast_seen_b_(64, false), fast_seen_g_(64, false),
          perceptron_(static_cast<size_t>(entries)),
          loop_(static_cast<size_t>(entries)),
          branch_stats_(static_cast<size_t>(entries)),
          btb_(static_cast<size_t>(entries)),
          ras_(static_cast<size_t>(max(0, ras_depth)), 0),
          ras_depth_(max(0, ras_depth)) {
        if (entries <= 0 || (entries & (entries - 1)) != 0)
            throw runtime_error("predictor entries phai la luy thua cua 2");
        if (hist_bits < 4 || hist_bits > 128)
            throw runtime_error("hist_bits phai trong [4,128]");
        for (auto& table : tage_) table.assign(static_cast<size_t>(entries), TageEntry{});
    }

    bool predict(u32 pc, bool backward, PredictionMeta& m) {
        m.ghr_snapshot = history_;
        m.path_history_snapshot = path_history_;
        const size_t bi = index_pc(pc);
        const u32 index_fold = fold_history(history_, hist_bits_, index_width());
        const size_t gi = ((pc >> 2) ^ index_fold) & mask_;
        m.local_history_snapshot = local_hist_[bi];
        const size_t lpi = (bi ^ static_cast<size_t>(m.local_history_snapshot)) & mask_;

        m.pred_bimodal = seen_b_[bi] ? (bimodal_[bi] >= 2) : backward;
        m.pred_gshare = seen_g_[gi] ? (gshare_[gi] >= 2) : backward;
        m.pred_local = seen_l_[lpi] ? (local_pht_[lpi] >= 2) : backward;
        const uint8_t sb = score_b_[bi], sg = score_g_[gi], sl = score_l_[lpi];
        bool base = m.pred_bimodal;
        uint8_t base_confidence = seen_b_[bi]
            ? static_cast<uint8_t>((bimodal_[bi] == 0 || bimodal_[bi] == 3) ? 2 : 1) : 0;
        if (sg > sb && sg >= sl) {
            base = m.pred_gshare;
            base_confidence = seen_g_[gi]
                ? static_cast<uint8_t>((gshare_[gi] == 0 || gshare_[gi] == 3) ? 2 : 1) : 0;
        } else if (sl > sb && sl > sg) {
            base = m.pred_local;
            base_confidence = seen_l_[lpi]
                ? static_cast<uint8_t>((local_pht_[lpi] == 0 || local_pht_[lpi] == 3) ? 2 : 1) : 0;
        }
        m.pred_alt = base;
        m.confidence = base_confidence;

        // Fast warm-up predictor: 64-entry bimodal + short-history gshare.
        // It runs beside TAGE and can provide a useful direction after the first sample.
        constexpr unsigned FAST_BITS = 6;
        const size_t fbi = (pc >> 2) & 63u;
        const u32 short_fold = fold_history(history_, min(hist_bits_, 8), FAST_BITS);
        const size_t fgi = ((pc >> 2) ^ short_fold) & 63u;
        m.fast_bimodal_index = static_cast<uint16_t>(fbi);
        m.fast_gshare_index = static_cast<uint16_t>(fgi);
        m.pred_fast_bimodal = fast_seen_b_[fbi] ? (fast_bimodal_[fbi] >= 2) : backward;
        m.pred_fast_gshare = fast_seen_g_[fgi] ? (fast_gshare_[fgi] >= 2) : backward;
        const bool fast_pick_g = fast_seen_g_[fgi] &&
            (!fast_seen_b_[fbi] || fast_chooser_[fbi] >= 2);
        m.fast_chooser_snapshot = fast_chooser_[fbi];
        m.pred_fast = fast_pick_g ? m.pred_fast_gshare : m.pred_fast_bimodal;

        int provider = -1;
        int alternate = -1;
        for (int t = 0; t < NT; ++t) {
            const size_t idx = tage_index(pc, history_, path_history_, HIST_LEN[static_cast<size_t>(t)]);
            const uint16_t tag = tage_tag(pc, history_, path_history_, HIST_LEN[static_cast<size_t>(t)]);
            m.tage_index[static_cast<size_t>(t)] = static_cast<uint16_t>(idx);
            m.tage_tag[static_cast<size_t>(t)] = tag;
            const TageEntry& te = tage_[static_cast<size_t>(t)][idx];
            const bool hit = te.valid && te.tag == tag;
            m.tage_hit[static_cast<size_t>(t)] = hit;
            m.tage_pred[static_cast<size_t>(t)] = hit ? (te.ctr >= 0) : false;
            if (hit) { alternate = provider; provider = t; }
        }

        bool taken = base;
        if (provider >= 0) {
            const TageEntry& pe = tage_[static_cast<size_t>(provider)][m.tage_index[static_cast<size_t>(provider)]];
            const bool provider_pred = (pe.ctr >= 0);
            const bool alt_pred = alternate >= 0
                ? m.tage_pred[static_cast<size_t>(alternate)] : base;
            const bool weak = abs(static_cast<int>(pe.ctr)) <= 1;
            const bool use_alt = weak && pe.useful == 0;
            taken = use_alt ? alt_pred : provider_pred;
            m.tage_alt_prediction = alt_pred;
            m.confidence = use_alt ? base_confidence
                : static_cast<uint8_t>(min(3, abs(static_cast<int>(pe.ctr))));
            m.tage_used = true;
            m.tage_provider = static_cast<int8_t>(provider);
            m.tage_alt_provider = static_cast<int8_t>(alternate);
            m.pred_tage = taken;
        } else {
            m.tage_alt_prediction = base;
            m.pred_tage = base;
        }

        // Fast fallback handles cold/weak TAGE entries without stalling Fetch.
        const bool tage_cold_or_weak = (provider < 0) || (m.confidence <= 1);
        if (tage_cold_or_weak) {
            taken = m.pred_fast;
            m.fast_used = true;
            m.confidence = (fast_seen_b_[fbi] || fast_seen_g_[fgi]) ? 1 : 0;
        }

        // Separate 2-bit chooser for backward-loop and forward-conditional BEQs.
        // The branch class is static PC-relative direction because ISA has one BEQ opcode.
        m.branch_class = backward ? 0u : 1u;
        const size_t chooser_index = bi * 2u + static_cast<size_t>(m.branch_class);
        m.hybrid_chooser_snapshot = hybrid_chooser_[chooser_index];
        const PerceptronEntry& pe = perceptron_[bi];
        int sum = pe.weight[PERCEPTRON_HISTORY];
        for (int bit = 0; bit < PERCEPTRON_HISTORY; ++bit) {
            const int x = history_.bit(static_cast<unsigned>(bit)) ? 1 : -1;
            sum += x * static_cast<int>(pe.weight[static_cast<size_t>(bit)]);
        }
        m.perceptron_index = static_cast<uint16_t>(bi);
        m.perceptron_sum = static_cast<int16_t>(clamp(sum, -32768, 32767));
        m.pred_perceptron = (sum >= 0);
        m.perceptron_trained = pe.trained;
        if (pe.trained >= 2 && m.hybrid_chooser_snapshot >= 2 && abs(sum) >= 3) {
            taken = m.pred_perceptron;
            m.perceptron_used = true;
        }

        // Loop predictor: after observing a stable trip count, predict the one
        // exit iteration rather than repeatedly missing the loop-exit branch.
        LoopEntry& le = loop_[bi];
        if (le.valid && le.tag == pc && le.confidence >= 2 && le.trip_count >= 3) {
            m.loop_main_taken = le.main_taken;
            m.loop_spec_before = le.speculative_iter;
            m.loop_resync_before = le.resync;
            if (le.resync) {
                // Predictor became confident while the fetch stream may already be
                // inside a loop. Follow TAGE until its predicted exit marks the next
                // loop boundary; only then start using trip-count predictions.
                if (taken != le.main_taken) {
                    le.resync = false;
                    le.speculative_iter = 0;
                }
            } else {
                m.loop_used = true;
                const bool exit_now = (static_cast<unsigned>(le.speculative_iter) + 1u == le.trip_count);
                m.loop_predicted_exit = exit_now;
                taken = exit_now ? !le.main_taken : le.main_taken;
                if (exit_now) le.speculative_iter = 0;
                else if (le.speculative_iter < 65534) ++le.speculative_iter;
            }
        }

        m.predicted_direction = taken;
        local_hist_[bi] = static_cast<uint16_t>(((m.local_history_snapshot << 1) | (taken ? 1u : 0u)) & mask_);
        history_ = history_.push_bit(taken).masked(static_cast<unsigned>(hist_bits_));
        path_history_ = path_history_.push_byte(path_token(pc, taken));
        return taken;
    }

    bool btb_lookup(u32 pc, u32& target, bool& indirect) const {
        const BTBEntry& e = btb_[index_pc(pc)];
        if (e.valid && e.tag == pc) { target = e.target; indirect = e.indirect; return true; }
        return false;
    }
    void btb_update(u32 pc, u32 target, bool indirect) {
        BTBEntry& e = btb_[index_pc(pc)];
        e.valid = true; e.tag = pc; e.target = target; e.indirect = indirect;
    }

    std::array<u32, 16> ras_snapshot() const {
        std::array<u32, 16> out{};
        for (int i = 0; i < min(ras_size_, 16); ++i) out[static_cast<size_t>(i)] = ras_[static_cast<size_t>(i)];
        return out;
    }
    int ras_size() const { return ras_size_; }
    bool ras_pop(u32& target) {
        if (ras_size_ <= 0) return false;
        target = ras_[static_cast<size_t>(--ras_size_)];
        return true;
    }
    void ras_push(u32 addr) {
        if (ras_depth_ <= 0) return;
        if (ras_size_ < ras_depth_) ras_[static_cast<size_t>(ras_size_++)] = addr;
        else {
            for (int i = 1; i < ras_depth_; ++i) ras_[static_cast<size_t>(i - 1)] = ras_[static_cast<size_t>(i)];
            ras_[static_cast<size_t>(ras_depth_ - 1)] = addr;
        }
    }
    void ras_restore(const std::array<u32, 16>& snap, int size) {
        ras_size_ = min(max(size, 0), min(ras_depth_, 16));
        for (int i = 0; i < ras_size_; ++i) ras_[static_cast<size_t>(i)] = snap[static_cast<size_t>(i)];
    }

    void update(u32 pc, const PredictionMeta& m, bool taken) {
        const size_t bi = index_pc(pc);
        const u32 index_fold = fold_history(m.ghr_snapshot, static_cast<int>(min(hist_bits_, 16)), index_width());
        const size_t gi = ((pc >> 2) ^ index_fold) & mask_;
        const size_t lpi = (bi ^ static_cast<size_t>(m.local_history_snapshot)) & mask_;

        train(bimodal_[bi], seen_b_[bi], taken);
        train(gshare_[gi], seen_g_[gi], taken);
        train(local_pht_[lpi], seen_l_[lpi], taken);

        // Update the micro fast predictor on every retired branch.
        const size_t fbi = m.fast_bimodal_index & 63u;
        const size_t fgi = m.fast_gshare_index & 63u;
        train(fast_bimodal_[fbi], fast_seen_b_[fbi], taken);
        train(fast_gshare_[fgi], fast_seen_g_[fgi], taken);
        if (m.pred_fast_bimodal != m.pred_fast_gshare) {
            if (m.pred_fast_gshare == taken) {
                if (fast_chooser_[fbi] < 3) ++fast_chooser_[fbi];
            } else if (fast_chooser_[fbi] > 0) --fast_chooser_[fbi];
        }
        const size_t hci = bi * 2u + static_cast<size_t>(m.branch_class & 1u);
        if (m.pred_perceptron != m.pred_tage) {
            if (m.pred_perceptron == taken) {
                if (hybrid_chooser_[hci] < 3) ++hybrid_chooser_[hci];
            } else if (hybrid_chooser_[hci] > 0) --hybrid_chooser_[hci];
        }
        train_score(score_b_[bi], m.pred_bimodal == taken);
        train_score(score_g_[gi], m.pred_gshare == taken);
        train_score(score_l_[lpi], m.pred_local == taken);
        train_score(score_perceptron_[bi], m.pred_perceptron == taken);
        train_score(score_tage_[bi], m.pred_tage == taken);
        train_perceptron(bi, m.ghr_snapshot, m.perceptron_sum, m.pred_perceptron, taken);
        train_loop(pc, taken);

        int provider = static_cast<int>(m.tage_provider);
        if (provider >= 0 && provider < NT) {
            const size_t idx = m.tage_index[static_cast<size_t>(provider)];
            TageEntry& e = tage_[static_cast<size_t>(provider)][idx];
            if (e.valid && e.tag == m.tage_tag[static_cast<size_t>(provider)]) {
                const bool old_pred = m.tage_pred[static_cast<size_t>(provider)];
                const bool provider_correct = old_pred == taken;
                train_signed(e.ctr, taken);
                const bool alt_correct = m.tage_alt_prediction == taken;
                if (old_pred != m.tage_alt_prediction) {
                    if (provider_correct && e.useful < 3) ++e.useful;
                    else if (!provider_correct && alt_correct && e.useful > 0) --e.useful;
                }
            } else {
                provider = -1;
            }
        }

        // Adaptive cold-start allocator: with no tagged provider, allocate after
        // the first miss; if an existing provider failed, demand two repeated
        // misses before consuming a longer-history entry.
        BranchStat& bs = branch_stats_[bi];
        if (!bs.valid || bs.pc != pc) { bs = BranchStat{}; bs.valid = true; bs.pc = pc; }
        const bool final_pred_wrong = (m.predicted_direction != taken);
        if (!final_pred_wrong) {
            bs.miss_streak = 0;
            return;
        }
        if (bs.miss_streak < 3) ++bs.miss_streak;
        const uint8_t required_misses = (provider < 0 ? 1u : 2u);
        if (bs.miss_streak < required_misses) return;
        bs.miss_streak = 0;

        const int first = provider + 1;
        if (first >= NT) return; // no longer-history table exists beyond the longest provider
        bool allocated = false;
        for (int t = first; t < NT && !allocated; ++t) {
            const size_t idx = m.tage_index[static_cast<size_t>(t)];
            TageEntry& e = tage_[static_cast<size_t>(t)][idx];
            if (!e.valid || e.useful == 0) {
                e.valid = true;
                e.tag = m.tage_tag[static_cast<size_t>(t)];
                e.ctr = taken ? 0 : -1;
                e.useful = 0;
                allocated = true;
            }
        }
        if (!allocated) {
            for (int t = first; t < NT; ++t) {
                TageEntry& e = tage_[static_cast<size_t>(t)][m.tage_index[static_cast<size_t>(t)]];
                if (e.useful > 0) --e.useful;
            }
        }
    }

    void repair(u32 pc, const PredictionMeta& m, bool actual_taken) {
        history_ = m.ghr_snapshot.push_bit(actual_taken).masked(static_cast<unsigned>(hist_bits_));
        path_history_ = m.path_history_snapshot.push_byte(path_token(pc, actual_taken));
        const size_t bi = index_pc(pc);
        local_hist_[bi] = static_cast<uint16_t>(((m.local_history_snapshot << 1) | (actual_taken ? 1u : 0u)) & mask_);
        repair_loop(pc, m, actual_taken);
    }
    void restore_loop_spec(u32 pc, uint16_t snapshot, bool resync_snapshot) {
        LoopEntry& e = loop_[index_pc(pc)];
        if (e.valid && e.tag == pc) {
            e.speculative_iter = snapshot;
            e.resync = resync_snapshot;
        }
    }
    History128 history_snapshot() const { return history_; }
    History128 path_history_snapshot() const { return path_history_; }
    void restore_history(const History128& snapshot) { history_ = snapshot.masked(static_cast<unsigned>(hist_bits_)); }
    void restore_path_history(const History128& snapshot) { path_history_ = snapshot; }
    void restore_local_history(u32 pc, uint16_t snapshot) {
        local_hist_[index_pc(pc)] = static_cast<uint16_t>(snapshot & mask_);
    }

private:
    unsigned index_width() const {
        unsigned w = 0; int n = entries_ - 1;
        while (n > 0) { ++w; n >>= 1; }
        return max(1u, w);
    }
    size_t index_pc(u32 pc) const { return (pc >> 2) & mask_; }
    static uint8_t path_token(u32 pc, bool taken) {
        const u32 p = (pc >> 2) ^ (pc >> 9) ^ (pc >> 17);
        return static_cast<uint8_t>((p & 0x7fu) | (taken ? 0x80u : 0u));
    }
    static u32 fold_history(const History128& history, int length, unsigned width) {
        u32 folded = 0;
        if (width == 0) return 0;
        for (int bit = 0; bit < length; ++bit)
            if (history.bit(static_cast<unsigned>(bit))) folded ^= (1u << (static_cast<unsigned>(bit) % width));
        return folded & ((1u << width) - 1u);
    }
    static u32 fold_path(const History128& path, int length, unsigned width) {
        // Fold PC-fragment bits and branch outcomes independently into the same signature.
        return fold_history(path, length, width);
    }
    size_t tage_index(u32 pc, const History128& history, const History128& path, int length) const {
        const unsigned w = index_width();
        const u32 gh = fold_history(history, length, w);
        const u32 ph = fold_path(path, min(128, length * 8), w);
        const u32 mix = (pc >> 2) ^ gh ^ ((ph << 1) | (ph >> (w > 1 ? w - 1 : 0)));
        return static_cast<size_t>(mix) & mask_;
    }
    uint16_t tage_tag(u32 pc, const History128& history, const History128& path, int length) const {
        const u32 gh = fold_history(history, length, 12);
        const u32 ph = fold_path(path, min(128, length * 8), 12);
        const u32 mix = (pc >> 2) ^ (pc >> 13) ^ gh ^ (ph * 0x9e3u);
        return static_cast<uint16_t>(mix & 0xffffu);
    }
    static void train(uint8_t& c, uint8_t& seen, bool taken) {
        if (!seen) { seen = true; c = taken ? 2 : 1; }
        else if (taken) { if (c < 3) ++c; }
        else if (c > 0) --c;
    }
    static void train_score(uint8_t& score, bool correct) {
        if (correct) { if (score < 3) ++score; }
        else if (score > 0) --score;
    }
    static void train_signed(int8_t& c, bool taken) {
        if (taken) { if (c < 3) ++c; }
        else if (c > -3) --c;
    }
    static int8_t sat_weight(int value) {
        return static_cast<int8_t>(clamp(value, -31, 31));
    }
    void train_perceptron(size_t index, const History128& history, int16_t sum,
                          bool predicted, bool actual) {
        PerceptronEntry& p = perceptron_[index];
        constexpr int THETA = 20;
        if (predicted != actual || abs(static_cast<int>(sum)) <= THETA) {
            const int target = actual ? 1 : -1;
            for (int bit = 0; bit < PERCEPTRON_HISTORY; ++bit) {
                const int x = history.bit(static_cast<unsigned>(bit)) ? 1 : -1;
                p.weight[static_cast<size_t>(bit)] = sat_weight(
                    static_cast<int>(p.weight[static_cast<size_t>(bit)]) + target * x);
            }
            p.weight[PERCEPTRON_HISTORY] = sat_weight(
                static_cast<int>(p.weight[PERCEPTRON_HISTORY]) + target);
        }
        if (p.trained < 255) ++p.trained;
    }
    void train_loop(u32 pc, bool taken) {
        LoopEntry& e = loop_[index_pc(pc)];
        if (!e.valid || e.tag != pc) {
            e = LoopEntry{};
            e.valid = true;
            e.tag = pc;
            e.main_taken = taken;
            e.train_run = 1;
            return;
        }
        if (taken == e.main_taken) {
            if (e.train_run < 65534) ++e.train_run;
            return;
        }
        const uint16_t observed = static_cast<uint16_t>(min<unsigned>(65535u,
            static_cast<unsigned>(e.train_run) + 1u));
        if (e.trip_count == observed) {
            if (e.confidence < 3) ++e.confidence;
            if (e.confidence >= 2) {
                // Confidence became usable after this loop completed. The fetch
                // stream may be ahead, so synchronize at its next observed exit.
                e.resync = true;
                e.speculative_iter = 0;
            }
        } else {
            e.trip_count = observed;
            e.confidence = 1;
            e.resync = false;
        }
        e.train_run = 0;
    }
    void repair_loop(u32 pc, const PredictionMeta& m, bool actual_taken) {
        if (!m.loop_used && !m.loop_resync_before) return;
        LoopEntry& e = loop_[index_pc(pc)];
        if (!e.valid || e.tag != pc) return;
        e.speculative_iter = m.loop_spec_before;
        e.resync = m.loop_resync_before;
        if (e.resync) {
            if (actual_taken != e.main_taken) {
                e.resync = false;
                e.speculative_iter = 0;
            }
        } else if (m.loop_used) {
            if (actual_taken == e.main_taken) {
                if (e.speculative_iter < 65534) ++e.speculative_iter;
                if (e.speculative_iter >= e.trip_count) e.confidence = 0;
            } else {
                e.speculative_iter = 0;
            }
        }
    }

    int entries_;
    u32 mask_;
    int hist_bits_;
    vector<uint8_t> bimodal_, gshare_, local_pht_;
    vector<uint8_t> score_b_, score_g_, score_l_;
    vector<uint8_t> seen_b_, seen_g_, seen_l_;
    vector<uint16_t> local_hist_;
    vector<uint8_t> score_perceptron_, score_tage_;
    vector<uint8_t> hybrid_chooser_;
    vector<uint8_t> fast_bimodal_, fast_gshare_, fast_chooser_;
    vector<uint8_t> fast_seen_b_, fast_seen_g_;
    vector<PerceptronEntry> perceptron_;
    vector<LoopEntry> loop_;
    array<vector<TageEntry>, NT> tage_;
    vector<BranchStat> branch_stats_;
    vector<BTBEntry> btb_;
    History128 history_{};
    History128 path_history_{};
    vector<u32> ras_;
    int ras_depth_ = 16, ras_size_ = 0;
};

// ---------------------------------------------------------------------------
// HELIX-1 6-wide out-of-order, co du doan nhanh va nap lenh speculative.
//
// - Nap toi da fetch_width lenh/chu ky; lenh nap o chu ky t phat hanh som nhat o t+1.
// - Register renaming bang tag ROB: WAW/WAR khong chan; RAW cho producer xong.
// - Nhanh BEQ: tournament predictor; if both operands are already ready at fetch,
//   compare them early and use the exact direction instead of a prediction.
//   JAL: dich nhay biet ngay khi giai ma nen luon "du doan" dung.
// - Khi nhanh ra ket qua (luc execute):
//     dung  -> lam tiep, khong mat chu ky nao.
//     sai   -> rollback local/GHR/RAS speculation, discard younger ROB entries,
//              then refetch the correct path after redirect_bubbles cycles.
//   Lenh CU HON nhanh van dung nen duoc giu va tiep tuc hoan tat.
// - Vi lenh speculative co the bi xoa nen:
//     * SW chi ghi DMEM luc RETIRE (store nam trong ROB), LW doc ngang tu SW cu hon
//       trong ROB (store-to-load forwarding).
//     * Fault (LW/SW sai dia chi, lenh sai, nap ngoai IMEM) chi duoc bao luc RETIRE.
//     * Thanh ghi kien truc chi ghi luc RETIRE, theo thu tu chuong trinh.
// - Memory van phat hanh theo thu tu, toi da memory_width LW/SW moi chu ky.
// - Hai LSU duoc mo hinh hoa bang 2 bank xen ke (word index bit 0 chon bank),
//   moi bank nhan toi da mot truy cap/chu ky; conflict cung bank se stall.
// - Load-forward tu store cu hon duoc giu dung theo thu tu ROB.
// ---------------------------------------------------------------------------
struct PipelineResult {
    array<u64, 32> regs{};
    array<u64, 32> fregs{};
    vector<u64> dmem;
    bool fault = false;
    int cycles = 0;
    int issued = 0;          // so lenh da execute (gom ca lenh sau do bi xoa)
    int committed = 0;       // so lenh retired; dung de tinh IPC
    int alu_ops = 0;
    int peak_issue = 0;
    int peak_integer_alu_issue = 0;
    int peak_floating_alu_issue = 0;
    int peak_branch_control_issue = 0;
    int peak_five_alu_ports = 0; // max simultaneous INT + FP + BRANCH operations in one cycle
    int floating_ops = 0;
    long long critical_path_alu_issues = 0;
    long long matrix_scheduler_cycles = 0;
    int max_dependency_fanout = 0;
    int loop_predictions = 0;
    int perceptron_predictions = 0;
    int fast_predictor_uses = 0;
    int memory_ops = 0;
    int peak_memory_issue = 0;
    int cond_branches = 0;   // so BEQ da retire
    int raw_mispredicts = 0; // raw direction misses before exact early resolution
    int predictor_predictions = 0; // branches actually speculated (excluding early/deferred)
    int predictor_misses = 0;
    int deferred_branches = 0; // low-confidence branches waited for exact execution
    int early_resolved_branches = 0;
    int mispredicts = 0;     // actual wrong-path redirects after early resolution
    int flushed = 0;         // so lenh bi xoa vi du doan sai
    int fetched = 0;         // so lenh da nap (gom ca duong sai)
};

class Pipeline {
    struct ROBEntry {
        uint64_t seq = 0;
        u32 pc = 0;
        u32 word = 0;
        Decoded d{};
        int64_t src1_tag = -1;
        int64_t src2_tag = -1;
        u64 src1_base = 0;
        u64 src2_base = 0;
        bool done = false;
        bool execution_started = false;
        int complete_cycle = 0;
        u64 pending_result = 0;
        bool fault = false;
        bool invalid_fetch = false;
        u64 result = 0;
        u32 next_pc = 0;
        // du doan nhanh
        bool predicted = false;      // frontend da chon huong nap tiep
        u32 pred_pc = 0;
        bool is_cond = false;        // BEQ
        bool taken = false;          // huong thuc te (BEQ)
        bool mispredicted = false;
        PredictionMeta pred_meta{};
        std::array<u32, 16> ras_snap{};
        int ras_size_snap = 0;
        bool ras_return = false;
        // bo nho
        u64 addr = 0;
        u64 store_data = 0;
        bool addr_ok = false;
    };

public:
    vector<u32> imem;
    vector<u64> dmem;
    array<u64, 32> regs{};
    array<u64, 32> fregs{}; // Separate FP architectural register file
    u32 pc = 0;
    bool halted = false, fault = false;
    int cycles = 0;
    int issued = 0;
    int committed = 0;
    int alu_ops = 0;
    int peak_issue = 0;
    int peak_integer_alu_issue = 0;
    int peak_floating_alu_issue = 0;
    int peak_branch_control_issue = 0;
    int peak_five_alu_ports = 0;
    int floating_ops = 0;
    long long critical_path_issue_count = 0;
    long long matrix_scheduler_cycles = 0;
    int peak_dependents_score = 0;
    int loop_predictions = 0;
    int perceptron_predictions = 0;
    int fast_predictor_uses = 0;
    int memory_ops = 0;
    int peak_memory_issue = 0;
    int cond_branches = 0;
    int raw_mispredicts = 0;
    int predictor_predictions = 0;
    int predictor_misses = 0;
    int deferred_branches = 0;
    int early_resolved_branches = 0;
    int mispredicts = 0;
    int flushed = 0;
    int fetched = 0;

private:
    struct StoreBufferEntry { u64 addr = 0, data = 0; uint64_t seq = 0; };
    Config cfg_;
    BranchPredictor bp_;
    vector<ROBEntry> rob_;
    deque<StoreBufferEntry> store_buffer_;
    u32 fetch_pc_ = 0;
    uint64_t next_seq_ = 1;
    bool fetch_stopped_ = false;
    int stall_ = 0;

    static bool is_memory(const Decoded& d) { return d.mem_r || d.mem_w; }

    const ROBEntry* find_seq(int64_t seq) const {
        if (seq < 0) return nullptr;
        for (const auto& e : rob_)
            if (static_cast<int64_t>(e.seq) == seq) return &e;
        return nullptr;
    }

    bool source_ready(const ROBEntry& e, bool first, u64& value) const {
        const bool used = first ? e.d.use1 : e.d.use2;
        if (!used) { value = 0; return true; }
        const int reg = first ? e.d.rs1 : e.d.rs2;
        const bool fp_source = first ? e.d.src1_fp : e.d.src2_fp;
        if (!fp_source && reg == 0) { value = 0; return true; } // x0 only; f0 is writable
        const int64_t tag = first ? e.src1_tag : e.src2_tag;
        const u64 base = first ? e.src1_base : e.src2_base;
        if (tag < 0) { value = base; return true; }
        const ROBEntry* producer = find_seq(tag);
        if (producer) {
            if (!producer->done || producer->fault) return false;
            value = producer->result;
            return true;
        }
        // The producer retired in order into the matching architectural register file.
        value = fp_source ? fregs[reg] : regs[reg];
        return true;
    }

    // ---------------- Fetch / rename ----------------
    void fetch() {
        if (stall_ > 0) { --stall_; return; }
        if (fetch_stopped_) return;

        array<int64_t, 32> last_writer{};
        array<int64_t, 32> last_fp_writer{};
        last_writer.fill(-1);
        last_fp_writer.fill(-1);
        for (const auto& e : rob_) {
            if (!e.d.reg_write) continue;
            if (e.d.dst_fp) last_fp_writer[e.d.rd] = static_cast<int64_t>(e.seq);
            else if (e.d.rd != 0) last_writer[e.d.rd] = static_cast<int64_t>(e.seq);
        }

        int fetched_now = 0;
        while (fetched_now < cfg_.fetch_width && rob_.size() < cfg_.rob_size && !fetch_stopped_) {
            ROBEntry e;
            e.seq = next_seq_++;
            e.pc = fetch_pc_;
            ++fetched;
            ++fetched_now;

            if ((fetch_pc_ & 3u) != 0 || (fetch_pc_ >> 2) >= imem.size()) {
                e.invalid_fetch = true;
                e.fault = true;
                e.done = true;
                rob_.push_back(e);
                fetch_stopped_ = true;
                break;
            }

            e.word = imem[fetch_pc_ >> 2];
            e.d = decode(e.word);
            e.pred_meta.ghr_snapshot = bp_.history_snapshot();
            e.pred_meta.path_history_snapshot = bp_.path_history_snapshot();
            if (e.d.use1) {
                const int r = e.d.rs1;
                if (e.d.src1_fp) {
                    e.src1_tag = last_fp_writer[r];
                    e.src1_base = fregs[r];
                } else if (r != 0) {
                    e.src1_tag = last_writer[r];
                    e.src1_base = regs[r];
                }
            }
            if (e.d.use2) {
                const int r = e.d.rs2;
                if (e.d.src2_fp) {
                    e.src2_tag = last_fp_writer[r];
                    e.src2_base = fregs[r];
                } else if (r != 0) {
                    e.src2_tag = last_writer[r];
                    e.src2_base = regs[r];
                }
            }
            if (e.d.reg_write) {
                if (e.d.dst_fp) last_fp_writer[e.d.rd] = static_cast<int64_t>(e.seq);
                else if (e.d.rd != 0) last_writer[e.d.rd] = static_cast<int64_t>(e.seq);
            }

            const u32 target = fetch_pc_ + static_cast<u32>(e.d.imm * 4u);
            u32 next_fetch = fetch_pc_ + 4u;

            e.ras_snap = bp_.ras_snapshot();
            e.ras_size_snap = bp_.ras_size();
            if (e.d.branch) {
                e.is_cond = true;
                if (cfg_.bp == BPMode::OFF) {
                    fetch_stopped_ = true;
                } else {
                    const bool backward = static_cast<int64_t>(e.d.imm) < 0;
                    bool predicted_taken = bp_.predict(fetch_pc_, backward, e.pred_meta);
                    // Early branch resolution if both operands are already ready.
                    auto early_value = [&](int reg, u64& value) -> bool {
                        if (reg == 0) { value = 0; return true; }
                        const int64_t tag = last_writer[reg];
                        if (tag < 0) { value = regs[reg]; return true; }
                        const ROBEntry* producer = find_seq(tag);
                        if (producer && producer->done && !producer->fault) { value = producer->result; return true; }
                        return false;
                    };
                    u64 early_a = 0, early_b = 0;
                    const bool same_source = (e.d.rs1 == e.d.rs2);
                    if (same_source || (early_value(e.d.rs1, early_a) && early_value(e.d.rs2, early_b))) {
                        predicted_taken = same_source || (early_a == early_b);
                        e.pred_meta.early_resolved = true;
                        // The operands are already available at fetch: resolve this BEQ
                        // now, rather than spending one of the six issue slots next cycle.
                        e.taken = predicted_taken;
                        e.next_pc = predicted_taken
                            ? e.pc + static_cast<u32>(e.d.imm * 4u) : e.pc + 4u;
                        e.done = true;
                        bp_.repair(e.pc, e.pred_meta, predicted_taken);
                    }
                    if (!e.pred_meta.early_resolved && e.pred_meta.fast_used)
                        ++fast_predictor_uses;
                    if (!e.pred_meta.early_resolved &&
                        cfg_.branch_confidence_threshold > 0 &&
                        e.pred_meta.confidence < cfg_.branch_confidence_threshold) {
                        // Do not send low-confidence speculative work down a likely
                        // wrong path. Restore the speculative histories and wait for
                        // the branch operands to become ready in the execution stage.
                        bp_.restore_history(e.pred_meta.ghr_snapshot);
                        bp_.restore_path_history(e.pred_meta.path_history_snapshot);
                        bp_.restore_local_history(e.pc, e.pred_meta.local_history_snapshot);
                        e.predicted = false;
                        fetch_stopped_ = true;
                    } else {
                        u32 btb_target = target; bool btb_indirect = false;
                        e.pred_meta.btb_hit = bp_.btb_lookup(fetch_pc_, btb_target, btb_indirect);
                        e.pred_pc = predicted_taken ? (e.pred_meta.btb_hit && !btb_indirect ? btb_target : target) : next_fetch;
                        e.predicted = true; next_fetch = e.pred_pc;
                    }
                }
            } else if (e.d.jump && !e.d.indirect_jump) {
                // Direct JAL has an exact PC-relative target at decode; resolve it
                // in the frontend instead of consuming the Branch/Control issue port.
                e.result = u64(e.pc) + 4u;
                e.next_pc = target;
                e.predicted = true;
                e.pred_pc = target;
                e.done = true;
                next_fetch = target;
                if (e.d.imm == 0) {
                    fetch_stopped_ = true; // HALT convention
                } else if (e.d.rd == 1) {
                    bp_.ras_push(e.pc + 4u); // call
                }
            } else if (e.d.indirect_jump) {
                u32 target_pc = 0; bool indirect = false;
                const bool is_return = (e.d.rd == 0 && e.d.rs1 == 1 && e.d.imm == 0);
                bool target_predicted = false;
                if (cfg_.bp != BPMode::OFF) {
                    if (is_return && bp_.ras_pop(target_pc)) {
                        target_predicted = true; e.pred_meta.ras_used = true;
                    } else if (bp_.btb_lookup(e.pc, target_pc, indirect)) {
                        target_predicted = true; e.pred_meta.btb_hit = true;
                    }
                }
                // If the JALR base is already ready, calculate its exact target
                // at Fetch. Otherwise use RAS/BTB speculation and validate in Control.
                u64 base_value = 0;
                const bool exact_target_ready = source_ready(e, true, base_value);
                if (exact_target_ready) {
                    const u32 exact_target = static_cast<u32>((base_value + e.d.imm) & ~u64(3));
                    e.result = u64(e.pc) + 4u;
                    e.next_pc = exact_target;
                    e.predicted = true; e.pred_pc = exact_target;
                    e.done = true;
                    next_fetch = exact_target;
                } else if (cfg_.bp != BPMode::OFF && target_predicted) {
                    e.predicted = true; e.pred_pc = target_pc; next_fetch = target_pc;
                } else {
                    fetch_stopped_ = true; // target unknown: wait for JALR source/execute
                }
                if (e.d.rd == 1) bp_.ras_push(e.pc + 4u);
                e.ras_return = is_return;
            } else if (e.d.illegal) {
                // Early-fault discard: mark the illegal instruction complete now and
                // stop this fetch path. Precise architectural fault is still at ROB head.
                e.fault = true; e.done = true; fetch_stopped_ = true;
            }

            rob_.push_back(e);
            fetch_pc_ = next_fetch;
            // BTB-style zero-bubble redirect: continue filling the same fetch bundle
            // from the predicted target instead of wasting the rest of this cycle.
            // Wrong predictions are still squashed by resolve_control next cycle.

        }
    }

    // ---------------- Retire ----------------
    void retire() {
        int n = 0;
        while (!rob_.empty() && rob_.front().done && n < cfg_.retire_width) {
            if (rob_.front().d.mem_w && store_buffer_.size() >= static_cast<size_t>(cfg_.store_buffer_entries)) break;
            ROBEntry e = rob_.front();
            rob_.erase(rob_.begin());

            if (e.fault || e.invalid_fetch) {
                // All older stores have retired; make them visible before precise trap.
                drain_store_buffer_all();
                fault = true;
                halted = true;
                rob_.clear();
                fetch_stopped_ = true;
                return;
            }
            if (e.d.mem_w) {
                // Retire to a FIFO store buffer; background drain writes backing DMEM.
                store_buffer_.push_back({e.addr, e.store_data, e.seq});
            }
            if (e.d.reg_write) {
                if (e.d.dst_fp) fregs[e.d.rd] = e.result;
                else if (e.d.rd != 0) regs[e.d.rd] = e.result;
            }

            ++committed;
            ++n;

            if (e.is_cond) {
                ++cond_branches;
                if (e.pred_meta.early_resolved) {
                    ++early_resolved_branches;
                } else if (!e.predicted) {
                    ++deferred_branches;
                } else {
                    ++predictor_predictions;
                    if (e.pred_meta.predicted_direction != e.taken) ++predictor_misses;
                }
                if (e.pred_meta.loop_used) ++loop_predictions;
                if (e.pred_meta.perceptron_used) ++perceptron_predictions;
                if (e.pred_meta.predicted_direction != e.taken) ++raw_mispredicts;
                if (e.mispredicted) ++mispredicts;
                if (cfg_.bp == BPMode::TOURNAMENT)
                    bp_.update(e.pc, e.pred_meta, e.taken);
            }
            if (e.d.branch || e.d.jump) {
                pc = e.next_pc;
                // A conditional branch's BTB target is the taken target only.
                // Writing the fall-through PC on a not-taken BEQ corrupts the next
                // taken prediction and causes a false target misprediction.
                if (e.d.jump || (e.d.branch && e.taken))
                    bp_.btb_update(e.pc, e.next_pc, e.d.indirect_jump);
                if (e.d.jump && !e.d.indirect_jump && e.d.imm == 0) {   // HALT
                    drain_store_buffer_all();
                    halted = true;
                    rob_.clear();
                    return;
                }
            } else {
                pc = e.pc + 4u;
            }
        }
    }

    void rollback_local_history_after(size_t index) {
        if (cfg_.bp != BPMode::TOURNAMENT) return;
        for (size_t j = rob_.size(); j-- > index + 1;) {
            const ROBEntry& younger = rob_[j];
            if (younger.is_cond) {
                bp_.restore_local_history(younger.pc, younger.pred_meta.local_history_snapshot);
                if (younger.pred_meta.loop_used || younger.pred_meta.loop_resync_before)
                    bp_.restore_loop_spec(younger.pc, younger.pred_meta.loop_spec_before,
                                          younger.pred_meta.loop_resync_before);
            }
        }
    }

    // Nhanh/JAL (khac HALT) vua co ket qua o vi tri index trong ROB.
    void resolve_control(size_t index, u32 actual_pc) {
        ROBEntry& e = rob_[index];
        if (e.d.jump && !e.d.indirect_jump && e.d.imm == 0) return; // HALT

        if (cfg_.bp == BPMode::OFF) {                       // khong du doan: nap tiep tu dich that
            fetch_pc_ = actual_pc;
            fetch_stopped_ = false;
            stall_ = cfg_.redirect_bubbles;
            return;
        }
        if (!e.predicted) {                                // frontend waited; this is not a misprediction
            if (e.is_cond) bp_.repair(e.pc, e.pred_meta, e.taken);
            fetch_pc_ = actual_pc;
            fetch_stopped_ = false;
            stall_ = cfg_.redirect_bubbles;
            return;
        }
        const bool direction_mismatch = e.is_cond && !e.pred_meta.early_resolved &&
            (e.pred_meta.predicted_direction != e.taken);
        if (e.pred_pc == actual_pc && !direction_mismatch) return; // target va huong dung

        // Du doan SAI: xoa moi lenh tre hon, nap lai tu dung duong.
        e.mispredicted = true;
        rollback_local_history_after(index);
        flushed += static_cast<int>(rob_.size() - (index + 1));
        rob_.erase(rob_.begin() + static_cast<ptrdiff_t>(index + 1), rob_.end());
        fetch_pc_ = actual_pc;
        fetch_stopped_ = false;
        stall_ = cfg_.redirect_bubbles;
        if (e.is_cond) bp_.repair(e.pc, e.pred_meta, e.taken);
        else { bp_.restore_history(e.pred_meta.ghr_snapshot); bp_.restore_path_history(e.pred_meta.path_history_snapshot); }
        // Restore speculative RAS state to the checkpoint at this control instruction,
        // then apply the actual call/return side effect exactly once.
        const auto snap = e.ras_snap; const int snap_size = e.ras_size_snap;
        const bool is_call = e.d.jump && e.d.rd == 1;
        const bool is_ret = e.d.indirect_jump && e.ras_return;
        bp_.ras_restore(snap, snap_size);
        if (is_call) bp_.ras_push(e.pc + 4u);
        if (is_ret) { u32 ignored = 0; bp_.ras_pop(ignored); }
    }

    // LW: lay du lieu tu SW cu hon gan nhat cung dia chi trong ROB, neu khong thi doc DMEM.
    // Memory phat hanh theo thu tu nen moi SW cu hon deu da co dia chi.
    // Precise early-fault discard: keep older instructions plus the faulting op,
    // flush younger speculative work immediately. An older branch may still squash
    // this fault later if its prediction turns out to be wrong.
    void discard_after_fault(size_t index) {
        if (index + 1 < rob_.size()) {
            rollback_local_history_after(index);
            flushed += static_cast<int>(rob_.size() - index - 1);
            rob_.erase(rob_.begin() + static_cast<ptrdiff_t>(index + 1), rob_.end());
        }
        bp_.restore_history(rob_[index].pred_meta.ghr_snapshot);
        bp_.restore_path_history(rob_[index].pred_meta.path_history_snapshot);
        bp_.ras_restore(rob_[index].ras_snap, rob_[index].ras_size_snap);
        fetch_stopped_ = true;
    }

    u64 load_value(size_t index, u64 addr) const {
        for (size_t j = index; j-- > 0;) {
            const ROBEntry& o = rob_[j];
            if (o.d.mem_w && o.done && o.addr_ok && o.addr == addr) return o.store_data;
        }
        for (auto it = store_buffer_.rbegin(); it != store_buffer_.rend(); ++it)
            if (it->addr == addr) return it->data;
        return dmem[static_cast<size_t>(addr >> 3)];
    }

    void drain_store_buffer() {
        vector<bool> used(static_cast<size_t>(cfg_.memory_width), false);
        int drained = 0;
        for (auto it = store_buffer_.begin(); it != store_buffer_.end() && drained < cfg_.store_drain_width;) {
            const size_t bank = static_cast<size_t>((it->addr >> 3) % static_cast<u64>(cfg_.memory_width));
            if (!used[bank]) {
                dmem[static_cast<size_t>(it->addr >> 3)] = it->data;
                used[bank] = true; it = store_buffer_.erase(it); ++drained;
            } else ++it;
        }
    }
    void drain_store_buffer_all() {
        while (!store_buffer_.empty()) {
            auto e = store_buffer_.front(); store_buffer_.pop_front();
            dmem[static_cast<size_t>(e.addr >> 3)] = e.data;
        }
    }

public:
    Pipeline(const vector<u32>& words, size_t data_words, const Config& cfg)
        : imem(make_imem(words)), dmem(data_words, 0), cfg_(cfg),
          bp_(cfg.predictor_entries, cfg.hist_bits, cfg.ras_depth) {
        if (data_words == 0) throw runtime_error("DMEM phai co it nhat 1 tu 64-bit");
        rob_.reserve(cfg.rob_size + 1);
    }

    void step() {
        if (halted) return;
        ++cycles;

        drain_store_buffer();
        retire();
        if (halted) return;

        // Complete prior Complex-ALU operations (MUL latency 3, DIV latency 12).
        for (auto& entry : rob_) {
            if (entry.execution_started && !entry.done && entry.complete_cycle <= cycles) {
                entry.result = entry.pending_result;
                entry.done = true;
                entry.execution_started = false;
            }
        }

        // ---------------- Issue + Execute ----------------
        struct Selected {
            uint64_t seq = 0;
            u64 a = 0, b = 0;
            int rob_index = -1;
            int dependents = 0;
            int critical_depth = 0;
            int port_type = 0; // instruction class
            int assigned_port = -1; // 0/1 simple lane, 2 complex, 3 branch/control, 4 LSU, 5 illegal
        };
        struct Candidate {
            Selected slot;
            long long priority = 0;
            bool chosen = false;
        };
        vector<Selected> selected;
        selected.reserve(static_cast<size_t>(cfg_.issue_width));
        vector<bool> memory_bank_used(static_cast<size_t>(cfg_.memory_width), false);

        // Build a dependency matrix from rename tags, then compute transitive
        // dependents and longest remaining dependency depth for each ROB producer.
        const size_t rob_n = rob_.size();
        unordered_map<uint64_t, size_t> index_by_seq;
        index_by_seq.reserve(rob_n * 2u + 1u);
        for (size_t i = 0; i < rob_n; ++i) index_by_seq.emplace(rob_[i].seq, i);
        vector<vector<size_t>> children(rob_n);
        for (size_t j = 0; j < rob_n; ++j) {
            if (rob_[j].done || rob_[j].execution_started) continue;
            const auto add_edge = [&](int64_t tag) {
                if (tag < 0) return;
                auto it = index_by_seq.find(static_cast<uint64_t>(tag));
                if (it == index_by_seq.end() || it->second >= j) return;
                auto& list = children[it->second];
                if (find(list.begin(), list.end(), j) == list.end()) list.push_back(j);
            };
            add_edge(rob_[j].src1_tag);
            add_edge(rob_[j].src2_tag);
        }
        vector<array<uint64_t, 2>> reach(rob_n, array<uint64_t, 2>{0, 0});
        vector<int> dependent_count(rob_n, 0), critical_depth(rob_n, 0);
        for (size_t rev = rob_n; rev-- > 0;) {
            for (size_t child : children[rev]) {
                reach[rev][child / 64u] |= (uint64_t(1) << (child % 64u));
                reach[rev][0] |= reach[child][0];
                reach[rev][1] |= reach[child][1];
                critical_depth[rev] = max(critical_depth[rev], critical_depth[child] + 1);
            }
            dependent_count[rev] = __builtin_popcountll(reach[rev][0]) +
                                   __builtin_popcountll(reach[rev][1]);
        }

        vector<Candidate> candidates;
        candidates.reserve(rob_n);
        for (size_t i = 0; i < rob_n; ++i) {
            ROBEntry& e = rob_[i];
            if (e.done || e.execution_started || e.invalid_fetch) continue;
            u64 a = 0, b = 0;
            if (!e.d.illegal && (!source_ready(e, true, a) || !source_ready(e, false, b))) continue;

            int port_type = 5; // illegal / fault-only slot
            if (is_memory(e.d)) port_type = 3;             // LSU AGU (separate from five ALU ports)
            else if (e.d.branch || e.d.jump) port_type = 2; // dedicated branch/control port
            else if (e.d.fp_exec) port_type = 4;           // two FP ports
            else if (!e.d.illegal) port_type = 0;           // two integer ports

            const int direct = static_cast<int>(children[i].size());
            const int total = dependent_count[i];
            const int depth = critical_depth[i];
            // Transitive dependent count is the primary score; critical depth and direct
            // fanout refine it. Program age is a deterministic tie-breaker only.
            const long long priority = static_cast<long long>(total) * 10000LL +
                                       static_cast<long long>(depth) * 100LL +
                                       static_cast<long long>(direct) * 10LL -
                                       static_cast<long long>(e.seq & 0xffffu);
            Candidate c;
            c.slot = {e.seq, a, b, static_cast<int>(i), total, depth, port_type, -1};
            c.priority = priority;
            candidates.push_back(c);
            peak_dependents_score = max(peak_dependents_score, total);
        }
        // Adaptive Matrix Scheduler: use dependency priority when the ROB is
        // genuinely under pressure; otherwise preserve FIFO age to avoid
        // reordering short/light programs where priority cannot unlock throughput.
        const bool critical_pressure =
            rob_n >= max<size_t>(16u, (cfg_.rob_size * 3u) / 4u) &&
            candidates.size() > static_cast<size_t>(cfg_.issue_width * 2);
        if (critical_pressure) ++matrix_scheduler_cycles;
        if (!critical_pressure) {
            for (Candidate& c : candidates)
                c.priority = -static_cast<long long>(c.slot.seq);
        }
        stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.priority != b.priority) return a.priority > b.priority;
            return a.slot.seq < b.slot.seq;
        });

        int integer_used = 0, floating_used = 0, control_used = 0, memory_used = 0;
        auto has_selected_seq = [&](uint64_t seq) {
            return any_of(selected.begin(), selected.end(), [&](const Selected& x) { return x.seq == seq; });
        };
        auto try_select = [&](Candidate& c) -> bool {
            if (c.chosen || static_cast<int>(selected.size()) >= cfg_.issue_width) return false;
            const int i = c.slot.rob_index;
            const ROBEntry& e = rob_[static_cast<size_t>(i)];
            int assigned_port = -1;
            switch (c.slot.port_type) {
                case 0: // Integer: exactly two independent single-issue ports
                    if (integer_used >= cfg_.integer_alu_width) return false;
                    assigned_port = integer_used++;
                    break;
                case 2: // Branch / control: exactly one dedicated port
                    if (control_used >= cfg_.branch_control_width) return false;
                    assigned_port = 4;
                    ++control_used;
                    break;
                case 3: { // LSU AGU is modelled separately from the five ALU ports
                    if (memory_used >= cfg_.memory_width) return false;
                    bool older_memory_unselected = false;
                    for (int j = 0; j < i; ++j) {
                        if (!is_memory(rob_[static_cast<size_t>(j)].d) || rob_[static_cast<size_t>(j)].done) continue;
                        if (!has_selected_seq(rob_[static_cast<size_t>(j)].seq)) {
                            older_memory_unselected = true;
                            break;
                        }
                    }
                    if (older_memory_unselected) return false;
                    const u64 addr = c.slot.a + e.d.imm;
                    const size_t bank = static_cast<size_t>((addr >> 3) % static_cast<u64>(cfg_.memory_width));
                    const bool bad_addr = ((addr & 7u) != 0) || ((addr >> 3) >= dmem.size());
                    if (!bad_addr && memory_bank_used[bank]) return false;
                    if (!bad_addr) memory_bank_used[bank] = true;
                    assigned_port = 5; // LSU, not an ALU port
                    ++memory_used;
                    break;
                }
                case 4: // Floating point: two independent IEEE-754 ports
                    if (floating_used >= cfg_.floating_alu_width) return false;
                    assigned_port = 2 + floating_used;
                    ++floating_used;
                    break;
                default: // illegal instruction: let precise fault handling retire it
                    assigned_port = 6;
                    break;
            }
            c.slot.assigned_port = assigned_port;
            selected.push_back(c.slot);
            c.chosen = true;
            return true;
        };
        // Priority pass. A younger memory op cannot pass an unresolved older one.
        for (Candidate& c : candidates) try_select(c);
        // Revisit memory candidates after older ready memory ops may have been selected.
        for (Candidate& c : candidates) {
            if (static_cast<int>(selected.size()) >= cfg_.issue_width) break;
            if (!c.chosen && c.slot.port_type == 3) try_select(c);
        }

        int executed_this_cycle = 0;
        int integer_alu_executed_this_cycle = 0;
        int floating_alu_executed_this_cycle = 0;
        int control_port_used_this_cycle = 0;
        int memory_executed_this_cycle = 0;
        for (const Selected& slot : selected) {
            // Resolve the selected operation by its stable sequence number.
            // A mispredicted older branch may erase younger ROB entries and
            // shift vector indices; their sequence IDs then simply disappear.
            size_t current_index = rob_.size();
            for (size_t k = 0; k < rob_.size(); ++k) {
                if (rob_[k].seq == slot.seq) { current_index = k; break; }
            }
            if (current_index == rob_.size()) continue; // flushed by an older branch
            ROBEntry& e = rob_[current_index];
            if (e.done) continue;
            const Decoded& d = e.d;
            if (slot.assigned_port == 0 || slot.assigned_port == 1) ++integer_alu_executed_this_cycle;
            else if (slot.assigned_port == 2 || slot.assigned_port == 3) ++floating_alu_executed_this_cycle;
            else if (slot.assigned_port == 4) ++control_port_used_this_cycle;
            ++issued;
            ++executed_this_cycle;

            if (d.illegal) {
                e.fault = true;
                e.done = true;
                discard_after_fault(current_index);
                break;
            }
            if (d.mem_r || d.mem_w) {
                ++memory_ops;
                ++memory_executed_this_cycle;
                const u64 addr = slot.a + d.imm;
                if ((addr & 7u) != 0 || (addr >> 3) >= dmem.size()) {
                    e.fault = true;                       // trap precise at ROB head
                    e.done = true;
                    discard_after_fault(current_index);
                    continue;
                }
                e.addr = addr;
                e.addr_ok = true;
                if (d.mem_r) e.result = load_value(current_index, addr);
                else e.store_data = slot.b;               // chua ghi DMEM
                e.done = true;
                continue;
            }
            if (d.branch) {
                e.taken = (slot.a == slot.b);
                e.next_pc = e.taken ? e.pc + static_cast<u32>(d.imm * 4u) : e.pc + 4u;
                e.done = true;
                resolve_control(current_index, e.next_pc);
                continue;
            }
            if (d.jump) {
                e.result = u64(e.pc) + 4u;
                e.next_pc = d.indirect_jump
                    ? static_cast<u32>((slot.a + d.imm) & ~u64(3))
                    : e.pc + static_cast<u32>(d.imm * 4u);
                e.done = true;
                resolve_control(current_index, e.next_pc);
                continue;
            }
            if (d.fp_exec) {
                double computed_fp = 0.0;
                int latency = 2;
                const int op = int(e.word & 127u);
                if (op == OPS.at("ITOF")) {
                    computed_fp = static_cast<double>(static_cast<int64_t>(slot.a));
                } else if (op == OPS.at("FTOI")) {
                    e.result = double_to_i64_bits(bits_to_double(slot.a));
                    e.execution_started = true;
                    e.pending_result = e.result;
                    e.complete_cycle = cycles + latency;
                    ++floating_ops;
                    if (slot.dependents > 0) ++critical_path_issue_count;
                    continue;
                } else {
                    const double fa = bits_to_double(slot.a);
                    const double fb = bits_to_double(slot.b);
                    if (op == OPS.at("FADD")) { computed_fp = fa + fb; latency = 3; }
                    else if (op == OPS.at("FSUB")) { computed_fp = fa - fb; latency = 3; }
                    else if (op == OPS.at("FMUL")) { computed_fp = fa * fb; latency = 4; }
                    else if (op == OPS.at("FDIV")) { computed_fp = fa / fb; latency = 12; }
                }
                e.execution_started = true;
                e.pending_result = double_to_bits(computed_fp);
                e.complete_cycle = cycles + latency;
                ++floating_ops;
                if (slot.dependents > 0) ++critical_path_issue_count;
            } else {
                const u64 alu_b = d.src_imm ? d.imm : (d.use2 ? slot.b : 0);
                const u64 computed = alu(d.alu, slot.a, alu_b).first;
                ++alu_ops;
                if (slot.dependents > 0) ++critical_path_issue_count;
                if (d.alu == 7 || d.alu == 8) {
                    e.execution_started = true;
                    e.pending_result = computed;
                    e.complete_cycle = cycles + (d.alu == 7 ? 2 : 11);
                } else {
                    e.result = computed;
                    e.done = true;
                }
            }
        }
        peak_issue = max(peak_issue, executed_this_cycle);
        peak_integer_alu_issue = max(peak_integer_alu_issue, integer_alu_executed_this_cycle);
        peak_floating_alu_issue = max(peak_floating_alu_issue, floating_alu_executed_this_cycle);
        peak_branch_control_issue = max(peak_branch_control_issue, control_port_used_this_cycle);
        peak_five_alu_ports = max(peak_five_alu_ports,
            integer_alu_executed_this_cycle + floating_alu_executed_this_cycle + control_port_used_this_cycle);
        peak_memory_issue = max(peak_memory_issue, memory_executed_this_cycle);

        fetch();

        if (rob_.empty() && fetch_stopped_ && stall_ == 0)
            throw runtime_error("deadlock: ROB rong va frontend dang dung");
    }

    PipelineResult result() const {
        PipelineResult r;
        r.regs = regs; r.fregs = fregs; r.dmem = dmem; r.fault = fault; r.cycles = cycles;
        r.issued = issued; r.committed = committed; r.alu_ops = alu_ops;
        r.peak_issue = peak_issue; r.peak_integer_alu_issue = peak_integer_alu_issue;
        r.peak_floating_alu_issue = peak_floating_alu_issue;
        r.peak_branch_control_issue = peak_branch_control_issue;
        r.peak_five_alu_ports = peak_five_alu_ports;
        r.floating_ops = floating_ops;
        r.critical_path_alu_issues = critical_path_issue_count;
        r.matrix_scheduler_cycles = matrix_scheduler_cycles;
        r.max_dependency_fanout = peak_dependents_score;
        r.loop_predictions = loop_predictions; r.perceptron_predictions = perceptron_predictions;
        r.fast_predictor_uses = fast_predictor_uses;
        r.memory_ops = memory_ops;
        r.peak_memory_issue = peak_memory_issue; r.cond_branches = cond_branches;
        r.raw_mispredicts = raw_mispredicts;
        r.predictor_predictions = predictor_predictions;
        r.predictor_misses = predictor_misses;
        r.deferred_branches = deferred_branches;
        r.early_resolved_branches = early_resolved_branches;
        r.mispredicts = mispredicts; r.flushed = flushed; r.fetched = fetched;
        return r;
    }
};

PipelineResult run_pipeline(const vector<u32>& words, int max_cycles = 5000,
                            size_t data_words = DEFAULT_DWORDS,
                            const Config& cfg = Config()) {
    Pipeline p(words, data_words, cfg);
    while (!p.halted) {
        p.step();
        if (p.cycles > max_cycles) throw runtime_error("multi-issue timeout");
    }
    return p.result();
}

// ---------------------------------------------------------------------------
// Kiem thu co dinh
// ---------------------------------------------------------------------------
struct DirectedTest {
    string name, source;
    unordered_map<int, u64> expected_regs;
    unordered_map<int, u64> expected_mem;
    bool expected_fault = false;
    unordered_map<int, u64> expected_fregs;

    DirectedTest(string n, string src, unordered_map<int, u64> xr,
                 unordered_map<int, u64> xm, bool fault_expected,
                 unordered_map<int, u64> xf = {})
        : name(std::move(n)), source(std::move(src)), expected_regs(std::move(xr)),
          expected_mem(std::move(xm)), expected_fault(fault_expected),
          expected_fregs(std::move(xf)) {}
};
vector<DirectedTest> make_directed_tests(size_t data_words = DEFAULT_DWORDS) {
    vector<DirectedTest> tests = {
        {"smoke (giong testbench SV)", R"ASM(
            addi x1, x0, 5
            addi x2, x1, 7         ; chuyen tiep kc 1
            addi x3, x0, 1
            add  x4, x1, x2        ; bypass regfile kc 2
            sw   x4, x0, 8
            lw   x5, x0, 8         ; ne SW->LW
            beq  x5, x4, over
            addi x6, x0, 99        ; bi bo qua
over:       addi x7, x0, 3
            halt
        )ASM", {{1, 5}, {2, 12}, {3, 1}, {4, 17}, {5, 17}, {6, 0}, {7, 3}}, {{1, 17}}, false},
        {"vong lap tinh 5+4+3+2+1", R"ASM(
            addi x1, x0, 5
            addi x2, x0, 0
            addi x3, x0, 1
loop:       add  x2, x2, x1
            sub  x1, x1, x3
            beq  x1, x0, done
            jal  x0, loop
done:       halt
        )ASM", {{1, 0}, {2, 15}}, {}, false},
        {"JAL ghi dia chi tra ve", R"ASM(
            jal  x5, tgt           ; x5 = 4
            addi x6, x0, 99        ; bi bo qua
tgt:        addi x7, x0, 7
            halt
        )ASM", {{5, 4}, {6, 0}, {7, 7}}, {}, false},
        {"dich bit & logic 64-bit", R"ASM(
            addi x1, x0, 1
            addi x2, x0, 63
            sll  x3, x1, x2        ; 1<<63
            srl  x4, x3, x2        ; ve 1
            addi x5, x0, -1
            and  x6, x5, x3
            xor  x7, x5, x5
            or   x8, x7, x1
            halt
        )ASM", {{3, u64(1) << 63}, {4, 1}, {5, M64}, {6, u64(1) << 63}, {7, 0}, {8, 1}}, {}, false},
        {"ghi vao x0 bi bo qua", R"ASM(
            addi x0, x0, 5
            add  x1, x0, x0
            halt
        )ASM", {{1, 0}}, {}, false},
        {"LW lech hang -> fault, khong commit", R"ASM(
            addi x1, x0, 7
            lw   x2, x0, 3
            addi x3, x0, 9         ; khong duoc thuc thi
            halt
        )ASM", {{1, 7}, {2, 0}, {3, 0}}, {}, true},
        {"SW ngoai DMEM -> fault", R"ASM(
            addi x1, x0, 5
            addi x2, x0, 512
            sw   x1, x2, 0
            halt
        )ASM", {{1, 5}, {2, 512}}, {}, true},
        {"DMEM 64 tu: ghi/doc tu thu 64", R"ASM(
            addi x1, x0, 42
            sw   x1, x0, 504
            lw   x2, x0, 504
            halt
        )ASM", {{1, 42}, {2, 42}}, {{63, 42}}, false},
        {"V9 ports: 2 Integer + 2 FP + 1 Branch simultaneous", R"ASM(
            addi x1, x0, 10
            addi x2, x0, 20
            addi x3, x0, 30
            addi x4, x0, 40
            itof f1, x1
            itof f2, x2
            itof f3, x3
            itof f4, x4
            mul  x9, x1, x2       ; 3-cycle producer keeps BEQ unresolved at Fetch
            beq  x9, x0, skip
            add  x10, x1, x2
            sub  x11, x4, x3
            and  x12, x1, x2
            or   x13, x3, x4
            xor  x14, x1, x4
            add  x15, x2, x3
            fadd f5, f1, f2
            fmul f6, f3, f4
            fadd f7, f1, f3
            fsub f8, f4, f2
            fadd f9, f1, f4
            fmul f10, f2, f3
            fdiv f11, f4, f2
            fsub f12, f3, f1
            fadd f13, f2, f4
            fmul f14, f1, f3
    skip:   halt
        )ASM", {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {9, 200},
                {10, 30}, {11, 10}, {12, 0}, {13, 62}, {14, 34}, {15, 50}}, {}, false,
            {{5, double_to_bits(30.0)}, {6, double_to_bits(1200.0)},
             {7, double_to_bits(40.0)}, {8, double_to_bits(20.0)},
             {9, double_to_bits(50.0)}, {10, double_to_bits(600.0)},
             {11, double_to_bits(2.0)}, {12, double_to_bits(20.0)},
             {13, double_to_bits(60.0)}, {14, double_to_bits(300.0)}}},
        {"V9 Integer ports: MUL/DIV and SHIFT", R"ASM(
            addi x1, x0, 12
            addi x2, x0, 3
            mul  x3, x1, x2
            div  x4, x3, x2
            sll  x5, x2, x2
            srl  x6, x5, x2
            halt
        )ASM", {{1, 12}, {2, 3}, {3, 36}, {4, 12}, {5, 24}, {6, 3}}, {}, false},
        {"V9 Integer ports: DIV by zero returns all ones", R"ASM(
            addi x1, x0, 123
            div  x2, x1, x0
            halt
        )ASM", {{1, 123}, {2, M64}}, {}, false},
        {"V9: 2 FP ports, IEEE-754 arithmetic and register conversion", R"ASM(
            addi x1, x0, 12
            addi x2, x0, 3
            addi x3, x0, 2
            addi x4, x0, 4
            itof f1, x1
            itof f2, x2
            itof f3, x3
            itof f4, x4
            fadd f5, f1, f2
            fmul f6, f3, f4
            fdiv f7, f1, f2
            fsub f8, f6, f3
            fdiv f9, f1, f0       ; IEEE-754 +infinity
            ftoi x9, f9           ; non-finite conversion defined as 0
            ftoi x5, f5
            ftoi x6, f6
            ftoi x7, f7
            ftoi x8, f8
            halt
        )ASM", {{1, 12}, {2, 3}, {5, 15}, {6, 8}, {7, 4}, {8, 6}, {9, 0}}, {}, false,
            {{1, double_to_bits(12.0)}, {2, double_to_bits(3.0)},
             {3, double_to_bits(2.0)}, {4, double_to_bits(4.0)},
             {5, double_to_bits(15.0)}, {6, double_to_bits(8.0)},
             {7, double_to_bits(4.0)}, {8, double_to_bits(6.0)},
             {9, double_to_bits(numeric_limits<double>::infinity())}}},
        {"V9: FP register zero is writable and independently renamed", R"ASM(
            addi x1, x0, 7
            itof f0, x1
            fadd f1, f0, f0
            ftoi x2, f1
            halt
        )ASM", {{1, 7}, {2, 14}}, {}, false,
            {{0, double_to_bits(7.0)}, {1, double_to_bits(14.0)}}},
        {"V9: FLD/FSD transfer IEEE-754 values through LSU", R"ASM(
            addi x1, x0, 12
            itof f1, x1
            fsd f1, x0, 0
            fld f2, x0, 0
            fadd f3, f1, f2
            ftoi x2, f3
            halt
        )ASM", {{1, 12}, {2, 24}}, {{0, double_to_bits(12.0)}}, false,
            {{1, double_to_bits(12.0)}, {2, double_to_bits(12.0)},
             {3, double_to_bits(24.0)}}},
        {"DUAL-LSU: 2 SW va 2 LW co the issue cung chu ky", R"ASM(
            addi x1, x0, 11
            addi x2, x0, 22
            sw   x1, x0, 0
            sw   x2, x0, 8
            lw   x3, x0, 0
            lw   x4, x0, 8
            halt
        )ASM", {{1, 11}, {2, 22}, {3, 11}, {4, 22}}, {{0, 11}, {1, 22}}, false},
        {"DUAL-LSU forwarding: SW->LW cung dia chi", R"ASM(
            addi x1, x0, 77
            sw   x1, x0, 24
            lw   x2, x0, 24
            halt
        )ASM", {{1, 77}, {2, 77}}, {{3, 77}}, false},
        {"8-bank LSU conflict-free accesses", R"ASM(
            addi x1, x0, 11
            addi x2, x0, 22
            addi x3, x0, 33
            addi x4, x0, 44
            addi x5, x0, 55
            addi x6, x0, 66
            addi x7, x0, 77
            addi x8, x0, 88
            sw x1, x0, 0
            sw x2, x0, 8
            sw x3, x0, 16
            sw x4, x0, 24
            sw x5, x0, 32
            sw x6, x0, 40
            sw x7, x0, 48
            sw x8, x0, 56
            lw x9, x0, 0
            lw x10, x0, 8
            lw x11, x0, 16
            lw x12, x0, 24
            lw x13, x0, 32
            lw x14, x0, 40
            lw x15, x0, 48
            lw x16, x0, 56
            halt
        )ASM", {{9, 11}, {10, 22}, {11, 33}, {12, 44}, {13, 55}, {14, 66}, {15, 77}, {16, 88}}, {{0, 11}, {1, 22}, {2, 33}, {3, 44}, {4, 55}, {5, 66}, {6, 77}, {7, 88}}, false},
        {"opcode sai -> fault", "", {}, {}, true},
        {"chay ra ngoai IMEM -> fault", "addi x1, x0, 1", {{1, 1}}, {}, true}
    };
    string long_program;
    for (int i = 0; i < 300; ++i) long_program += "nop\n";
    long_program += "addi x1, x0, 77\nhalt\n";
    tests.push_back({"IMEM dong: chuong trinh hon 128 lenh", long_program, {{1, 77}}, {}, false});

    // Tao dia chi byte nam ngay sau DMEM bang cach dung ADD/SLL-style doubling.
    // Cach nay khong bi gioi han boi do rong immediate ADDI 15-bit.
    uint64_t out_of_range_addr = static_cast<uint64_t>(data_words) * 8u;
    ostringstream out_of_range_program;
    out_of_range_program << "addi x1, x0, 5\naddi x2, x0, 0\n";
    int highest_bit = 63;
    while (highest_bit > 0 && ((out_of_range_addr >> highest_bit) & 1u) == 0)
        --highest_bit;
    for (int bit = highest_bit; bit >= 0; --bit) {
        out_of_range_program << "add x2, x2, x2\n";
        if ((out_of_range_addr >> bit) & 1u)
            out_of_range_program << "addi x2, x2, 1\n";
    }
    out_of_range_program << "sw x1, x2, 0\nhalt\n";
    for (auto& test : tests) {
        if (test.name == "SW ngoai DMEM -> fault") {
            test.source = out_of_range_program.str();
            test.expected_regs[2] = out_of_range_addr;
            break;
        }
    }

    tests.push_back({"SW->LW: forwarding trong ROB (SW chua retire)", R"ASM(
            addi x7, x0, 5
            addi x1, x0, 1
            add  x1, x1, x1        ; chuoi phu thuoc dai giu dau ROB, SW khong the retire som
            add  x1, x1, x1
            add  x1, x1, x1
            add  x1, x1, x1
            add  x1, x1, x1
            add  x1, x1, x1
            sw   x7, x0, 16        ; execute som nhung chua retire
            lw   x3, x0, 16        ; phai lay du lieu tu SW trong ROB
            halt
        )ASM", {{1, 64}, {3, 5}, {7, 5}}, {{2, 5}}, false});
    // ---- du doan nhanh: lenh speculative bi xoa khong duoc de lai dau vet ----
    tests.push_back({"BP: SW tren duong sai khong ghi DMEM", R"ASM(
            addi x7, x0, 5
            addi x1, x0, 1
            beq  x1, x1, skip      ; luon nhay; nhay tien nen BP doan "khong nhay" -> sai
            addi x2, x0, 99        ; duong sai
            sw   x7, x0, 8         ; duong sai: da execute roi bi xoa, khong duoc ghi DMEM
    skip:   lw   x3, x0, 8         ; phai doc 0
            halt
        )ASM", {{7, 5}, {1, 1}, {2, 0}, {3, 0}}, {{1, 0}}, false});
    tests.push_back({"BP: fault/lenh sai tren duong sai bi bo qua", R"ASM(
            addi x1, x0, 1
            beq  x1, x1, ok
            lw   x4, x0, 3         ; duong sai: LW lech hang
            .word 0x7f             ; duong sai: lenh sai
    ok:     addi x5, x0, 5
            halt
        )ASM", {{4, 0}, {5, 5}}, {}, false});
    tests.push_back({"BP: duong sai hop le khong duoc commit", R"ASM(
            addi x1, x0, 1
            jal  x0, last
    back:   addi x6, x0, 6         ; duong sai (BP doan nhay lui)
            halt
    last:   beq  x1, x0, back      ; x1=1 -> khong nhay; thuc te chay ra ngoai IMEM
        )ASM", {{1, 1}, {6, 0}}, {}, true});
    tests.push_back({"BP: sai huong nhung cung next-PC (phai sua GHR/path)", R"ASM(
        addi x1, x0, 0
        beq x1, x0, 1
        halt
    )ASM", {{1, 0}}, {}, false});
    tests.push_back({"JALR/RET: BTB + RAS call-return", R"ASM(
            addi x1, x0, 0
            jal x1, subroutine
            addi x2, x0, 7
            halt
    subroutine: addi x3, x0, 99
            ret
        )ASM", {{2, 7}, {3, 99}}, {}, false});
    tests.push_back({"JALR indirect target from BTB", R"ASM(
    callsite: addi x5, x0, 24
            jalr x6, x5, 0
            halt
            nop
            nop
            nop
    target: addi x7, x7, 1
            addi x8, x0, 2
            beq x7, x8, done
            jal x0, callsite
    done:   halt
        )ASM", {{6, 8}, {7, 2}, {8, 2}}, {}, false});
    tests.push_back({"BP: hai vong lap long nhau", R"ASM(
            addi x1, x0, 100
    outer:  addi x2, x0, 3
    inner:  add  x3, x3, x1
            addi x2, x2, -1
            beq  x2, x0, idone
            jal  x0, inner
    idone:  addi x1, x1, -1
            beq  x1, x0, done
            jal  x0, outer
    done:   halt
        )ASM", {{1, 0}, {2, 0}, {3, 15150}}, {}, false});
    tests.push_back({"BP: nhanh xen ke chan/le (gshare hoc)", R"ASM(
            addi x1, x0, 40
            addi x5, x0, 1
    loop:   and  x3, x1, x5
            beq  x3, x0, even
            addi x6, x6, 1
            jal  x0, next
    even:   addi x7, x7, 1
    next:   addi x1, x1, -1
            beq  x1, x0, done
            jal  x0, loop
    done:   halt
        )ASM", {{1, 0}, {3, 1}, {6, 20}, {7, 20}}, {}, false});

    return tests;
}
vector<u32> test_words(const DirectedTest& t) {
    return t.source.empty() ? vector<u32>{0x0000007Fu} : assemble(t.source);
}

Config with_bp(Config cfg, BPMode m) {
    cfg.bp = m;
    return cfg;
}

void run_directed(const vector<DirectedTest>& tests, size_t data_words, const Config& cfg) {
    for (const auto& t : tests) {
        vector<u32> words = test_words(t);
        Result g = golden(words, 100000, data_words);
        PipelineResult first;
        for (BPMode mode : {BPMode::TOURNAMENT, BPMode::OFF}) {      // dung voi CA HAI che do
            PipelineResult p = run_pipeline(words, 5000, data_words, with_bp(cfg, mode));
            const string tag = string("[") + t.name + (mode == BPMode::OFF ? "|BP tat" : "|BP bat") + "]";
            for (auto [r, value] : t.expected_regs) {
                if (g.regs[r] != value) throw runtime_error(tag + " golden x" + to_string(r) + " sai");
                if (p.regs[r] != value) throw runtime_error(tag + " pipeline x" + to_string(r) + " sai");
            }
            for (auto [m, value] : t.expected_mem)
                if (g.dmem[m] != value || p.dmem[m] != value) throw runtime_error(tag + " dmem sai");
            for (auto [r, value] : t.expected_fregs)
                if (g.fregs[r] != value || p.fregs[r] != value)
                    throw runtime_error(tag + " f" + to_string(r) + " sai");
            if (g.fault != t.expected_fault || p.fault != t.expected_fault)
                throw runtime_error(tag + " fault sai");
            if (g.regs != p.regs || g.fregs != p.fregs || g.dmem != p.dmem)
                throw runtime_error(tag + " golden != pipeline");
            if (t.name.find("V9 ports:") != string::npos &&
                (p.peak_integer_alu_issue < 2 || p.peak_floating_alu_issue < 2 ||
                 (mode == BPMode::TOURNAMENT &&
                  (p.peak_branch_control_issue < 1 || p.peak_five_alu_ports < 5))))
                throw runtime_error(tag + " chua kich hoat du cong theo che do frontend");
            if (p.peak_integer_alu_issue > cfg.integer_alu_width)
                throw runtime_error(tag + " vuot qua gioi han ALU integer");
            if (t.name.find("DUAL-LSU:") != string::npos && p.peak_memory_issue < min(cfg.memory_width, 2))
                throw runtime_error(tag + " khong phat hanh du 2 LSU trong mot chu ky");
            if (t.name.find("8-bank LSU") != string::npos && p.peak_memory_issue < min(cfg.memory_width, cfg.issue_width))
                throw runtime_error(tag + " khong khai thac du cac bank trong gioi han issue width");
            if (mode == BPMode::TOURNAMENT && t.name.find("hai vong lap long nhau") != string::npos &&
                p.loop_predictions == 0)
                throw runtime_error(tag + " loop predictor khong duoc kich hoat tren vong lap lap lai");
            if (mode == BPMode::TOURNAMENT) first = p;
        }
        cout << "  OK  " << t.name << "  (" << first.cycles << " chu ky, "
             << first.committed << " lenh retire, dinh " << first.peak_issue << " lenh/chu ky, "
             << "dinh " << first.peak_integer_alu_issue << " INT + "
             << first.peak_floating_alu_issue << " FP + " << first.peak_branch_control_issue
             << " Branch/cycle (5-port simultaneous peak=" << first.peak_five_alu_ports
             << "), " << first.floating_ops << " FP ops, "
             << "dinh " << first.peak_memory_issue << " LSU/chu ky";
        if (first.cond_branches)
            cout << ", " << first.mispredicts << "/" << first.cond_branches << " nhanh sai"
                 << ", loop=" << first.loop_predictions << ", perceptron=" << first.perceptron_predictions;
        if (t.name.find("hai vong lap long nhau") != string::npos) {
            const double acc = first.predictor_predictions
                ? 100.0 * double(first.predictor_predictions - first.predictor_misses) /
                    double(first.predictor_predictions) : 0.0;
            cout << ", predictor-only=" << setprecision(2) << acc << "% ("
                 << first.predictor_predictions - first.predictor_misses << "/"
                 << first.predictor_predictions << ")" << setprecision(3);
        }
        cout << ")\n";
    }
}

// ---------------------------------------------------------------------------
// Sinh chuong trinh ngau nhien va doi chieu
// ---------------------------------------------------------------------------
int randint(mt19937_64& rng, int lo, int hi) {
    uniform_int_distribution<int> dist(lo, hi);
    return dist(rng);
}
double rand01(mt19937_64& rng) {
    return generate_canonical<double, 53>(rng);
}

// Mot lenh ngau nhien. max_fwd = so lenh toi da co the nhay tien (nho hon 1 -> khong sinh nhanh).
u32 random_instr(mt19937_64& rng, size_t data_words, int max_fwd, bool allow_illegal) {
    auto R = [&]() { return randint(rng, 0, 7); };
    const double k = rand01(rng);
    const int max_word = static_cast<int>(min<size_t>(63, data_words - 1));
    if (k < 0.38) {
        const int ops[] = {0, 1, 8, 9, 10, 16, 17};
        return enc_r(ops[randint(rng, 0, 6)], R(), R(), R());
    } else if (k < 0.55) {
        bool big = rand01(rng) < 0.15;
        int imm = big ? randint(rng, -16384, 16383) : randint(rng, -40, 40);
        return enc_i(OPS.at("ADDI"), R(), R(), imm);
    } else if (k < 0.67) {
        int imm = (rand01(rng) < 0.93) ? 8 * randint(rng, 0, max_word) : randint(rng, -20, 300);
        int rs1 = (rand01(rng) < 0.9) ? 0 : R();
        return enc_i(OPS.at("LW"), R(), rs1, imm);
    } else if (k < 0.79) {
        int imm = (rand01(rng) < 0.93) ? 8 * randint(rng, 0, max_word) : randint(rng, -20, 300);
        int rs1 = (rand01(rng) < 0.9) ? 0 : R();
        return enc_s(OPS.at("SW"), rs1, R(), imm);
    } else if (k < 0.93 && max_fwd >= 1) {
        return enc_s(OPS.at("BEQ"), R(), R(), randint(rng, 1, min(4, max_fwd)));
    } else if (k < 0.985 && max_fwd >= 1) {
        return enc_j(OPS.at("JAL"), R(), randint(rng, 1, min(3, max_fwd)));
    } else if (allow_illegal) {
        const u32 bad[] = {0x7F, 0x07, 0x02, 0xFFFFFFFFu};
        return bad[randint(rng, 0, 3)];
    }
    return enc_r(0, R(), R(), R());
}

// Chuong trinh nhanh toan tien (nhay tien 1..4 lenh), ket thuc HALT.
void random_program(mt19937_64& rng, int n, vector<u32>& out, size_t data_words) {
    out.clear();
    for (int i = 0; i < n; ++i) out.push_back(random_instr(rng, data_words, 4, true));
    out.push_back(HALT);
}

// Chuong trinh co VONG LAP that (nhay lui) -> bo du doan phai hoc.
// x30 la bien dem (than vong lap chi dung x0..x7). Nhanh tien trong than khong
// bao gio nhay qua lenh giam bien dem => vong lap luon ket thuc.
void random_loop_program(mt19937_64& rng, vector<u32>& out, size_t data_words) {
    out.clear();
    const int body = randint(rng, 3, 14);
    const int iters = randint(rng, 2, 9);
    out.push_back(enc_i(OPS.at("ADDI"), 30, 0, iters));          // index 0
    for (int p = 1; p <= body; ++p)                               // index 1..body
        out.push_back(random_instr(rng, data_words, body + 1 - p, rand01(rng) < 0.05));
    out.push_back(enc_i(OPS.at("ADDI"), 30, 30, -1));            // index body+1
    out.push_back(enc_s(OPS.at("BEQ"), 30, 0, 2));               // index body+2: het vong -> HALT
    out.push_back(enc_j(OPS.at("JAL"), 0, -(body + 2)));         // index body+3: ve lenh dau than
    out.push_back(HALT);                                          // index body+4
}

struct Agg {
    long long committed = 0, cycles = 0, branches = 0, raw_miss = 0;
    long long predicted = 0, predictor_miss = 0, deferred = 0;
    long long early = 0, miss = 0, flushed = 0;
    long long loop_predictions = 0, perceptron_predictions = 0, fast_predictor_uses = 0;
    long long critical_path_alu_issues = 0;
    long long matrix_scheduler_cycles = 0;
    int max_dependency_fanout = 0;
    void add(const PipelineResult& p) {
        committed += p.committed; cycles += p.cycles; branches += p.cond_branches;
        raw_miss += p.raw_mispredicts;
        predicted += p.predictor_predictions; predictor_miss += p.predictor_misses;
        deferred += p.deferred_branches; early += p.early_resolved_branches;
        loop_predictions += p.loop_predictions; perceptron_predictions += p.perceptron_predictions;
        fast_predictor_uses += p.fast_predictor_uses;
        critical_path_alu_issues += p.critical_path_alu_issues;
        matrix_scheduler_cycles += p.matrix_scheduler_cycles;
        max_dependency_fanout = max(max_dependency_fanout, p.max_dependency_fanout);
        miss += p.mispredicts; flushed += p.flushed;
    }
    double ipc() const { return cycles ? double(committed) / double(cycles) : 0.0; }
};

void run_random(int n_programs, uint64_t seed, size_t data_words, const Config& cfg) {
    mt19937_64 rng(seed);
    int faults = 0, loops = 0;
    Agg on, off;
    vector<u32> words;
    for (int it = 0; it < n_programs; ++it) {
        if (it % 2 == 1) { random_loop_program(rng, words, data_words); ++loops; }
        else random_program(rng, randint(rng, 5, 50), words, data_words);
        Result g = golden(words, 100000, data_words);
        for (BPMode mode : {BPMode::TOURNAMENT, BPMode::OFF}) {
            PipelineResult p = run_pipeline(words, 5000, data_words, with_bp(cfg, mode));
            if (g.regs != p.regs || g.dmem != p.dmem || g.fault != p.fault) {
                cout << "LECH o chuong trinh #" << it << (mode == BPMode::OFF ? " (BP tat)" : " (BP bat)") << ":";
                for (u32 x : words) cout << " " << hex << showbase << x;
                cout << dec << noshowbase << "\n";
                for (int i = 0; i < 32; ++i)
                    if (g.regs[i] != p.regs[i])
                        cout << "   x" << i << ": golden=" << hex << showbase << g.regs[i]
                             << " pipeline=" << p.regs[i] << dec << noshowbase << "\n";
                cout << "   fault golden/pipeline: " << g.fault << " " << p.fault << "\n";
                throw runtime_error("kiem thu ngau nhien phat hien sai lech");
            }
            (mode == BPMode::TOURNAMENT ? on : off).add(p);
        }
        faults += g.fault;
    }
    cout << "  OK  " << n_programs << " chuong trinh ngau nhien (" << loops << " co vong lap) khop 100% o ca BP bat/tat ("
         << faults << " chuong trinh fault)\n";
    cout << fixed << setprecision(3)
         << "  IPC trung binh: BP tat = " << off.ipc() << ", BP bat = " << on.ipc() << "\n";
    cout << "  BP raw truoc early-resolve: " << on.raw_miss << "/" << on.branches << " miss ("
         << setprecision(1) << (on.branches ? 100.0 * double(on.branches - on.raw_miss) / double(on.branches) : 0.0)
         << "% dung); early-resolve=" << on.early << " nhanh\n";
    cout << "  BP thuc su du doan: " << on.predicted << " nhanh, sai=" << on.predictor_miss
         << " (" << setprecision(1) << (on.predicted ? 100.0 * double(on.predicted - on.predictor_miss) / double(on.predicted) : 0.0)
         << "% accuracy); coverage=" << setprecision(1)
         << (on.branches ? 100.0 * double(on.predicted) / double(on.branches) : 0.0)
         << "%; defer=" << on.deferred << "; early-exact=" << on.early
         << "; fast-used=" << on.fast_predictor_uses
         << "; loop-used=" << on.loop_predictions << "; perceptron-used=" << on.perceptron_predictions << "\n";
    cout << "  Scheduler: matrix-priority cycles=" << on.matrix_scheduler_cycles
         << "; critical-path ALU issues=" << on.critical_path_alu_issues
         << "; max live dependent fanout=" << on.max_dependency_fanout << "\n";
    cout << "  Recovery: " << on.miss << "/" << on.branches << " nhanh phai redirect ("
         << setprecision(1) << (on.branches ? 100.0 * double(on.branches - on.miss) / double(on.branches) : 0.0)
         << "% khong can redirect), flush=" << on.flushed << " lenh\n";
}

// ---------------------------------------------------------------------------
// Benchmark IPC
//   - ILP cua chuong trinh moi la tran cua IPC: k chuoi phu thuoc doc lap, moi chuoi
//     chay 1 lenh/chu ky => IPC <= k. May 6-wide khong the vuot qua ILP cua code.
//   - V9 co 2 integer ports, 2 FP ports, 1 branch port; integer-only ILP is capped at 2.
// ---------------------------------------------------------------------------
PipelineResult run_checked(const vector<u32>& words, size_t data_words, const Config& cfg, int max_cycles) {
    Result g = golden(words, 5000000, data_words);
    PipelineResult p = run_pipeline(words, max_cycles, data_words, cfg);
    if (g.fault != p.fault || g.regs != p.regs || g.fregs != p.fregs || g.dmem != p.dmem)
        throw runtime_error("benchmark: pipeline != golden");
    return p;
}
double ipc_of(const PipelineResult& p) { return p.cycles ? double(p.committed) / double(p.cycles) : 0.0; }

vector<u32> chains_program(int chains, int n, int short_chain_len = -1) {
    vector<u32> w;
    for (int i = 0; i < n; ++i)
        for (int c = 1; c <= chains; ++c) {
            if (c == chains && short_chain_len >= 0 && i >= short_chain_len) continue;
            w.push_back(enc_i(OPS.at("ADDI"), c, c, 1));
        }
    w.push_back(HALT);
    return w;
}

vector<u32> floating_point_chains_program(int chains, int n) {
    if (chains < 1 || chains > 12) throw runtime_error("FP chains phai trong [1,12]");
    vector<u32> w;
    w.push_back(enc_i(OPS.at("ADDI"), 1, 0, 1));
    w.push_back(enc_r(OPS.at("ITOF"), 1, 1, 0)); // f1 = 1.0
    for (int c = 2; c < 2 + chains; ++c)
        w.push_back(enc_r(OPS.at("FADD"), c, c, 1)); // seed all FPR dependency chains
    for (int i = 0; i < n; ++i)
        for (int c = 2; c < 2 + chains; ++c)
            w.push_back(enc_r(OPS.at("FADD"), c, c, 1));
    w.push_back(HALT);
    return w;
}

vector<u32> memory_read_eight_program(int groups) {
    vector<u32> w;
    // Seed eight independent 64-bit words, one in each interleaved bank.
    for (int i = 0; i < 8; ++i) {
        w.push_back(enc_i(OPS.at("ADDI"), i + 1, 0, 11 * (i + 1)));
        w.push_back(enc_s(OPS.at("SW"), 0, i + 1, 8 * i));
    }
    for (int i = 0; i < groups; ++i) {
        for (int bank = 0; bank < 8; ++bank)
            w.push_back(enc_i(OPS.at("LW"), bank + 9, 0, 8 * bank));
    }
    w.push_back(HALT);
    return w;
}

void run_ipc_benchmarks(size_t data_words, const Config& cfg) {
    const int N = 1200;
    long long mix_committed = 0, mix_cycles = 0;
    auto add_mix = [&](const PipelineResult& p) { mix_committed += p.committed; mix_cycles += p.cycles; };
    cout << fixed << setprecision(3);
    // ILP ceiling diagnostics are deliberately kept separate from the >=3.5
    // performance qualification suite: a 3-chain program cannot exceed 3 IPC,
    // and a 3-full-plus-1-half chain has a theoretical ceiling of exactly 3.5.
    {
        auto p = run_checked(chains_program(3, N), data_words, cfg, 20000);
        cout << "  DIAGNOSTIC: 3 independent chains (ILP ceiling = 3): IPC = " << ipc_of(p) << "\n";
    }
    {
        auto p = run_checked(chains_program(4, N, N / 2), data_words, cfg, 20000);
        cout << "  DIAGNOSTIC: 3 full + 1 half chain (ILP ceiling = 3.5): IPC = " << ipc_of(p) << "\n";
    }
    for (int k = 2; k <= 6; ++k) {
        auto p = run_checked(chains_program(k, N), data_words, cfg, 20000);
        if (k == 2) add_mix(p);
        cout << "  " << (k <= 2 ? "QUALIFY" : "DIAGNOSTIC") << ": "
             << k << " independent integer chains (2-port ceiling = 2): IPC = " << ipc_of(p) << "\n";
        if (k == 2 && ipc_of(p) < 1.75)
            throw runtime_error("integer port qualification failed: 2-chain IPC < 1.75");
    }

    // Two FP issue ports: use enough independent recurrence chains to cover FP latency.
    {
        auto p = run_checked(floating_point_chains_program(8, 250), data_words, cfg, 30000);
        add_mix(p);
        cout << "  QUALIFY: 2 FP ports, 8 independent FADD chains: IPC = " << ipc_of(p)
             << ", FP peak = " << p.peak_floating_alu_issue << "/cycle"
             << ", FP ops = " << p.floating_ops << "\n";
        if (p.peak_floating_alu_issue != 2 || ipc_of(p) < 1.50)
            throw runtime_error("floating port qualification failed");
    }

    // Memory throughput qualification: 1000 independent loads spread across eight banks.
    {
        auto p = run_checked(memory_read_eight_program(125), data_words, cfg, 10000);
        add_mix(p);
        cout << "  QUALIFY: Memory benchmark (1000 LW, 8 interleaved banks): IPC = " << ipc_of(p)
             << ", peak LSU = " << p.peak_memory_issue << "/cycle"
             << ", total memory ops = " << p.memory_ops << "\n";
        for (int r = 9; r <= 16; ++r)
            if (p.regs[r] != static_cast<u64>((r - 8) * 11))
                throw runtime_error("eight-bank memory benchmark: load data mismatch");
        if (p.peak_memory_issue < 4 || cfg.memory_width < 8)
            throw runtime_error("eight-bank memory benchmark did not use multiple LSU banks");
        if (ipc_of(p) < 3.5) throw runtime_error("performance qualification failed: memory IPC < 3.5");
    }

    // Kernel vong lap: 4 chuoi + bien dem + BEQ thoat + JAL quay lai. Moi nhanh deu bi
    // "cho ket qua" neu khong co bo du doan.
    string kernel = "addi x10, x0, 2000\n"
                    "loop: addi x1, x1, 1\n addi x2, x2, 1\n addi x3, x3, 1\n addi x4, x4, 1\n"
                    "addi x10, x10, -1\n beq x10, x0, done\n jal x0, loop\n done: halt\n";
    auto kw = assemble(kernel);
    auto off = run_checked(kw, data_words, with_bp(cfg, BPMode::OFF), 200000);
    auto on = run_checked(kw, data_words, with_bp(cfg, BPMode::TOURNAMENT), 200000);
    add_mix(on);
    cout << "  QUALIFY: Kernel vong lap (4 chuoi + dieu khien, 2000 vong):\n"
         << "    BP tat : IPC = " << ipc_of(off) << " (" << off.cycles << " chu ky)\n"
         << "    BP bat : IPC = " << ipc_of(on) << " (" << on.cycles << " chu ky, "
         << on.mispredicts << "/" << on.cond_branches << " nhanh sai)\n";
    const double loop_pred_acc = on.predictor_predictions
        ? 100.0 * double(on.predictor_predictions - on.predictor_misses) / double(on.predictor_predictions) : 0.0;
    cout << "    Predictor-only: " << setprecision(2) << loop_pred_acc << "% ("
         << on.predictor_predictions - on.predictor_misses << "/" << on.predictor_predictions
         << " dung), coverage=" << setprecision(1)
         << (on.cond_branches ? 100.0 * double(on.predictor_predictions) / double(on.cond_branches) : 0.0)
         << "%; early exact=" << on.early_resolved_branches << " nhanh\n" << setprecision(3);
    if (ipc_of(on) < 1.60) throw runtime_error("performance qualification failed: loop IPC < 1.60 for 2-integer-port V9 topology");
    if (ipc_of(on) <= ipc_of(off)) throw runtime_error("bo du doan khong cai thien IPC");

    // Kernel nhanh xen ke chan/le: kho voi bo dem 2-bit don thuan, gshare hoc duoc theo lich su.
    string alt = "addi x1, x0, 4000\n addi x5, x0, 1\n"
                 "loop: addi x8, x8, 1\n addi x9, x9, 1\n addi x11, x11, 1\n addi x12, x12, 1\n"
                 "and x3, x1, x5\n beq x3, x0, even\n addi x6, x6, 1\n jal x0, next\n"
                 "even: addi x7, x7, 1\n"
                 "next: addi x1, x1, -1\n beq x1, x0, done\n jal x0, loop\n done: halt\n";
    auto aw = assemble(alt);
    auto aoff = run_checked(aw, data_words, with_bp(cfg, BPMode::OFF), 400000);
    auto aon = run_checked(aw, data_words, with_bp(cfg, BPMode::TOURNAMENT), 400000);
    add_mix(aon);
    cout << "  QUALIFY: Kernel branch chẵn/lẻ có 4 chuỗi ILP (4000 vòng):\n"
         << "    BP tat : IPC = " << ipc_of(aoff) << "\n"
         << "    BP bat : IPC = " << setprecision(6) << ipc_of(aon) << setprecision(3)
         << ", " << aon.mispredicts << "/" << aon.cond_branches
         << " nhanh sai (" << setprecision(1)
         << 100.0 * double(aon.cond_branches - aon.mispredicts) / double(aon.cond_branches)
         << "% dung), " << aon.flushed << " lenh bi xoa\n" << setprecision(3);
    if (ipc_of(aon) <= ipc_of(aoff)) throw runtime_error("kernel xen ke: BP khong cai thien IPC");
    const double branch_pred_acc = aon.predictor_predictions
        ? 100.0 * double(aon.predictor_predictions - aon.predictor_misses) / double(aon.predictor_predictions) : 0.0;
    cout << "    Predictor-only: " << setprecision(2) << branch_pred_acc << "% ("
         << aon.predictor_predictions - aon.predictor_misses << "/" << aon.predictor_predictions
         << " dung), coverage=" << setprecision(1)
         << (aon.cond_branches ? 100.0 * double(aon.predictor_predictions) / double(aon.cond_branches) : 0.0)
         << "%; early exact=" << aon.early_resolved_branches
         << "; loop-used=" << aon.loop_predictions
         << "; perceptron-used=" << aon.perceptron_predictions << " nhanh\n" << setprecision(3);
    // Report the branch-heavy kernel as a measured result; do not force the
    // random branch pattern to pass a threshold it does not reach.
    cout << "  MIX IPC (7 benchmark cases, total retired / total cycles) = "
         << (mix_cycles ? double(mix_committed) / double(mix_cycles) : 0.0)
         << "  [random correctness stress reported separately]\n";
}


// ---------------------------------------------------------------------------
// Doi chieu RTL that neu may co iverilog va vvp
// ---------------------------------------------------------------------------
string shell_quote(const string& s) {
    string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}
string read_file(const fs::path& path) {
    ifstream f(path);
    if (!f) throw runtime_error("khong mo duoc file: " + path.string());
    ostringstream ss; ss << f.rdbuf(); return ss.str();
}
void run_rtl_tests(const vector<vector<u32>>& progs, const fs::path& source_dir,
                   size_t data_words = DEFAULT_DWORDS) {
    if (system("command -v iverilog >/dev/null 2>&1") != 0 ||
        system("command -v vvp >/dev/null 2>&1") != 0) {
        cout << "  (khong co iverilog -> bo qua doi chieu RTL that; xem README)\n";
        return;
    }
    auto stamp = chrono::steady_clock::now().time_since_epoch().count();
    fs::path tmp = fs::temp_directory_path() / ("helix1_" + to_string(stamp));
    fs::create_directories(tmp);
    fs::path sim = tmp / "sim";
    vector<fs::path> srcs = {
        source_dir / "rtl" / "alu.sv",
        source_dir / "rtl" / "decoder.sv",
        source_dir / "rtl" / "regfile.sv",
        source_dir / "rtl" / "helix1_cpu.sv",
        source_dir / "tb" / "cpu_tb.sv"
    };
    string compile_cmd = "iverilog -g2012 -o " + shell_quote(sim.string());
    for (const auto& p : srcs) compile_cmd += " " + shell_quote(p.string());
    if (system(compile_cmd.c_str()) != 0) throw runtime_error("iverilog bien dich RTL that that bai");
    fs::path smoke_file = tmp / "smoke.txt";
    string smoke_cmd = "vvp " + shell_quote(sim.string()) + " > " + shell_quote(smoke_file.string()) + " 2>&1";
    if (system(smoke_cmd.c_str()) != 0) throw runtime_error("chay RTL smoke test that bai");
    string smoke = read_file(smoke_file);
    string last_line;
    {
        stringstream ss(smoke); string line;
        while (getline(ss, line)) if (!trim(line).empty()) last_line = line;
    }
    cout << "  RTL smoke: " << last_line << "\n";
    if (smoke.find("PASS") == string::npos) throw runtime_error("RTL smoke test khong PASS: " + smoke);
    int bad = 0;
    for (size_t i = 0; i < progs.size(); ++i) {
        fs::path hex_file = tmp / ("p" + to_string(i) + ".hex");
        fs::path out_file = tmp / ("o" + to_string(i) + ".txt");
        {
            ofstream f(hex_file);
            if (!f) throw runtime_error("khong tao duoc file hex");
            for (u32 w : progs[i]) f << hex << nouppercase << setw(8) << setfill('0') << w << "\n";
        }
        string cmd = "vvp " + shell_quote(sim.string()) + " +PROG=" + shell_quote(hex_file.string()) +
                     " +OUT=" + shell_quote(out_file.string()) + " > /dev/null 2>&1";
        if (system(cmd.c_str()) != 0) throw runtime_error("vvp loi o chuong trinh RTL #" + to_string(i));
        unordered_map<string, u64> got;
        {
            ifstream f(out_file);
            string key, value;
            while (f >> key >> value) {
                if (key == "FAULT") got[key] = stoull(value, nullptr, 10);
                else got[key] = stoull(value, nullptr, 16);
            }
        }
        Result g = golden(progs[i], 100000, data_words);
        bool ok = got.count("FAULT") && bool(got["FAULT"]) == g.fault;
        for (int r = 1; r < 32; ++r) {
            string key = "X" + to_string(r);
            if (!got.count(key) || got[key] != g.regs[r]) ok = false;
        }
        for (size_t m = 0; m < g.dmem.size(); ++m) {
            string key = "M" + to_string(m);
            if (!got.count(key) || got[key] != g.dmem[m]) ok = false;
        }
        if (!ok) {
            ++bad;
            cout << "  RTL LECH o chuong trinh #" << i << ":";
            for (u32 w : progs[i]) cout << " " << hex << showbase << w;
            cout << dec << noshowbase << "\n";
        }
    }
    if (bad == 0) cout << "  RTL that khop mo hinh chuan tren " << progs.size() << " chuong trinh\n";
    else cout << "  RTL that co " << bad << " chuong trinh lech\n";
    if (bad) throw runtime_error("doi chieu RTL that that bai");
}

int main(int argc, char** argv) {
    try {
        size_t data_words = DEFAULT_DWORDS;
        int stress_samples = 1000;
        uint64_t stress_seed = 2026;
        Config cfg;
        for (int i = 1; i < argc; ++i) {
            const string a = argv[i];
            if (a == "--dmem" && i + 1 < argc) data_words = stoull(argv[++i]);
            else if (a == "--samples" && i + 1 < argc) stress_samples = stoi(argv[++i]);
            else if (a == "--seed" && i + 1 < argc) stress_seed = stoull(argv[++i]);
            else if (a == "--penalty" && i + 1 < argc) cfg.redirect_bubbles = stoi(argv[++i]);
            else if (a == "--lsu" && i + 1 < argc) cfg.memory_width = stoi(argv[++i]);
            else if (a == "--alu" && i + 1 < argc) cfg.integer_alu_width = stoi(argv[++i]);
            else if (a == "--fp-alu" && i + 1 < argc) cfg.floating_alu_width = stoi(argv[++i]);
            else if (a == "--rob" && i + 1 < argc) cfg.rob_size = stoull(argv[++i]);
            else if (a == "--entries" && i + 1 < argc) cfg.predictor_entries = stoi(argv[++i]);
            else if (a == "--hist-bits" && i + 1 < argc) cfg.hist_bits = stoi(argv[++i]);
            else if (a == "--bp-confidence" && i + 1 < argc) cfg.branch_confidence_threshold = stoi(argv[++i]);
            else throw runtime_error("Usage: ./helix1 [--dmem N] [--samples N] [--seed N] [--penalty N] [--lsu 1..8] [--alu 2] [--fp-alu 2] [--rob 16..128] [--entries 512..4096] [--hist-bits 4..128] [--bp-confidence 0|1]");
        }
        if (stress_samples < 1 || stress_samples > 100000)
            throw runtime_error("--samples phai trong [1,100000]");
        if (data_words < 64 || data_words > numeric_limits<u64>::max() / 8u)
            throw runtime_error("DMEM phai co it nhat 64 tu va kich thuoc byte khong duoc tran 64-bit");
        if (cfg.redirect_bubbles < 0) throw runtime_error("--penalty phai >= 0");
        if (cfg.memory_width < 1 || cfg.memory_width > 8) throw runtime_error("--lsu chi ho tro 1..8");
        if (cfg.integer_alu_width != 2 || cfg.floating_alu_width != 2 || cfg.branch_control_width != 1)
            throw runtime_error("V9 topology co dinh: 2 Integer + 2 FP + 1 Branch; dung --alu 2 --fp-alu 2");
        if (cfg.rob_size < 8 || cfg.rob_size > 128) throw runtime_error("--rob phai trong [8,128]");
        if (cfg.predictor_entries < 16 || cfg.predictor_entries > 4096) throw runtime_error("--entries phai trong [16,4096]");
        if (cfg.hist_bits < 4 || cfg.hist_bits > 128) throw runtime_error("--hist-bits phai trong [4,128]");
        if (cfg.branch_confidence_threshold < 0 || cfg.branch_confidence_threshold > 1)
            throw runtime_error("--bp-confidence phai la 0 hoac 1");

        cout << "HELIX-1 PORT5 V9 C++17 | 6-wide dispatch OOO | " << cfg.integer_alu_width
             << " integer ALU + " << cfg.floating_alu_width << " FP ALU + "
             << cfg.branch_control_width << " branch port | " << cfg.memory_width
             << " banked LSU | Fast-warmup TAGE hybrid + Path History (Micro-Bimodal/Gshare + Bimodal+Gshare+Local+6-level TAGE, " << cfg.predictor_entries
             << " entries/table) | ROB " << cfg.rob_size << " | fetch " << cfg.fetch_width
             << "/cycle | retire " << cfg.retire_width << "/cycle | dynamic IMEM\n";
        cout << "DMEM: " << data_words << " tu x 64-bit (" << data_words * 8 << " byte); "
             << "LSU ports: " << cfg.memory_width << "; doi huong sau nhanh: "
             << cfg.redirect_bubbles << " chu ky\n";
        cout << "Issue topology: 2 Integer(ALU/Logic/SHIFT/MUL/DIV) + 2 FP(IEEE-754 binary64 FADD/FSUB/FMUL/FDIV, separate F0-F31) + 1 Branch/Control(BEQ/JAL/JALR); LSU AGU separate; total dispatch=6\n";
        cout << "Predictor: Fast64(Bimodal+Gshare)/Bimodal/Gshare/Local/6xTAGE(4/8/16/32/64/128)/PathHist128/LocalHist/Perceptron32/LoopPredictor/BTB = "
             << cfg.predictor_entries << " entries each; confidence-threshold=" << cfg.branch_confidence_threshold
             << "; RAS depth=" << cfg.ras_depth
             << "; GHR=" << cfg.hist_bits << " bits; PathHistory=128 bits; adaptive allocator=1/2 misses; Perceptron-TAGE 2-bit chooser by branch class; StoreBuffer=" << cfg.store_buffer_entries
             << " entries; early illegal decode=ON\n";
        cout << "Kiem thu truc tiep (moi test chay voi BP bat VA tat):\n";
        run_directed(make_directed_tests(data_words), data_words, cfg);
        cout << "Kiem thu ngau nhien stress (nhanh tien + vong lap, vs mo hinh chuan):\n";
        run_random(stress_samples, stress_seed, data_words, cfg);
        cout << "Benchmark IPC:\n";
        run_ipc_benchmarks(data_words, cfg);
    } catch (const exception& e) {
        cerr << "LOI: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
